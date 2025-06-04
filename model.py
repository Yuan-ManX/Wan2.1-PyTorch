# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

__all__ = ['WanModel']

T5_CONTEXT_TOKEN_NUMBER = 512
FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2


def sinusoidal_embedding_1d(dim, position):
    """
    生成一维正弦位置嵌入。

    该函数生成正弦和余弦位置编码，用于为输入序列添加位置信息。

    参数:
        dim (int): 嵌入的维度，必须是偶数。
        position (torch.Tensor): 位置张量，形状为 [L]，其中 L 是序列长度。

    返回:
        torch.Tensor: 正弦位置嵌入，形状为 [L, dim]。
    """
    assert dim % 2 == 0

    # 计算嵌入维度的一半
    half = dim // 2
    # 将位置张量转换为 float64 类型
    position = position.type(torch.float64)

    # 计算正弦和余弦嵌入
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))  # 计算正弦和余弦的指数项
    # 将正弦和余弦值拼接起来
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    # 返回正弦位置嵌入
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    """
    生成旋转位置编码（RoPE）参数。

    该函数生成用于旋转位置编码的参数，这些参数用于在自注意力机制中引入位置信息。

    参数:
        max_seq_len (int): 序列的最大长度。
        dim (int): 嵌入的维度，必须是偶数。
        theta (float, 可选): 旋转角度的基数，默认为10000。

    返回:
        torch.Tensor: RoPE 参数，形状为 [max_seq_len, dim // 2]。
    """
    assert dim % 2 == 0
    
    # 计算频率
    freqs = torch.outer(
        torch.arange(max_seq_len),  # 生成序列长度的范围
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))  # 计算频率
    # 将频率转换为极坐标形式
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    # 返回 RoPE 参数
    return freqs


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    """
    应用旋转位置编码（RoPE）到输入张量。

    该函数将 RoPE 应用于输入张量，以在自注意力机制中引入位置信息。

    参数:
        x (torch.Tensor): 输入张量，形状为 [B, L, N, C]。
        grid_sizes (torch.Tensor): 网格大小张量，形状为 [B, 3]，其中第二维包含 (F, H, W)。
        freqs (torch.Tensor): RoPE 参数，形状为 [max_seq_len, C // 2]。

    返回:
        torch.Tensor: 应用了 RoPE 后的张量，形状为 [B, L, N, C]。
    """
    # 获取序列长度和通道数的一半
    n, c = x.size(2), x.size(3) // 2

    # 将 RoPE 参数分割为三部分
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # 遍历每个样本
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        # 计算序列长度
        seq_len = f * h * w

        # 预计算乘数
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))  # 将输入张量转换为复数张量
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),  # 扩展频率张量以匹配网格大小
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)  # 拼接并重塑频率张量

        # 应用旋转位置编码  
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)  # 应用 RoPE 并展平
        x_i = torch.cat([x_i, x[i, seq_len:]])  # 拼接处理后的张量和剩余部分

        # 添加到输出列表
        output.append(x_i)
    
    # 堆叠并返回结果
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):
    """
    WanRMSNorm 类实现了 RMS 层归一化。

    该类实现了 RMS 层归一化，并在归一化后应用可学习的权重参数。
    """
    def __init__(self, dim, eps=1e-5):
        """
        初始化 WanRMSNorm。

        参数:
            dim (int): 输入张量的维度。
            eps (float, 可选): 用于数值稳定性的小常数，默认为1e-5。
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        # 可学习的权重参数，初始化为全1
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        前向传播过程，应用 RMS 层归一化。

        参数:
            x (torch.Tensor): 输入张量，形状为 [B, L, C]。

        返回:
            torch.Tensor: 应用了 RMS 层归一化后的张量，形状为 [B, L, C]。
        """
        # 应用 RMS 层归一化，并应用可学习的权重参数
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        """
        计算 RMS 层归一化的内部计算。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 归一化后的张量。
        """
        # 计算 RMS 层归一化
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    """
    WanLayerNorm 类实现了层归一化。

    该类实现了标准的层归一化，并在前向传播过程中将输入转换为 float 进行计算，然后转换回原始类型。
    """
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        """
        初始化 WanLayerNorm。

        参数:
            dim (int): 输入张量的维度。
            eps (float, 可选): 用于数值稳定性的小常数，默认为1e-6。
            elementwise_affine (bool, 可选): 是否使用可学习的权重和偏置，默认为 False。
        """
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        """
        前向传播过程，应用层归一化。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 应用了层归一化后的张量，类型与输入相同。
        """
        # 将输入转换为 float 进行层归一化，然后转换回原始类型
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):
    """
    WanSelfAttention 类实现了自注意力机制，扩展了标准的多头自注意力机制，添加了 RMS 层归一化、旋转位置编码（RoPE）以及窗口局部注意力。
    """
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        """
        初始化 WanSelfAttention。

        参数:
            dim (int): 注意力机制的维度。
            num_heads (int): 注意力头的数量。
            window_size (Tuple[int, int], 可选): 滑动窗口局部注意力的窗口大小，默认为 (-1, -1)。
            qk_norm (bool, 可选): 是否对查询和键进行 RMS 层归一化，默认为 True。
            eps (float, 可选): 层归一化的 epsilon，默认为1e-6。
        """
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # 定义线性层
        self.q = nn.Linear(dim, dim)  # 查询线性层
        self.k = nn.Linear(dim, dim)  # 键线性层
        self.v = nn.Linear(dim, dim)  # 值线性层
        self.o = nn.Linear(dim, dim)  # 输出线性层

        # RMS 层归一化
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()  # 查询归一化
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()  # 键归一化

    def forward(self, x, seq_lens, grid_sizes, freqs):
        """
        前向传播过程，应用自注意力机制。

        参数:
            x (torch.Tensor): 输入张量，形状为 [B, L, num_heads, C / num_heads]。
            seq_lens (torch.Tensor): 序列长度张量，形状为 [B]。
            grid_sizes (torch.Tensor): 网格大小张量，形状为 [B, 3]，第二维包含 (F, H, W)。
            freqs (torch.Tensor): RoPE 参数，形状为 [1024, C / num_heads / 2]。

        返回:
            torch.Tensor: 应用了自注意力机制后的张量，形状为 [B, L, C]。
        """
        # 获取张量的维度信息
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # 查询、键和值函数
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)  # 计算查询并进行归一化
            k = self.norm_k(self.k(x)).view(b, s, n, d)  # 计算键并进行归一化
            v = self.v(x).view(b, s, n, d)  # 计算值
            return q, k, v

        # 计算查询、键和值
        q, k, v = qkv_fn(x)

        # 应用旋转位置编码
        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),  # 应用 RoPE 到查询
            k=rope_apply(k, grid_sizes, freqs),  # 应用 RoPE 到键
            v=v,  # 值
            k_lens=seq_lens,  # 键序列长度
            window_size=self.window_size)  # 窗口大小

        # 展平张量
        x = x.flatten(2)
        # 应用输出线性层
        x = self.o(x)
        # 返回输出张量
        return x


class WanT2VCrossAttention(WanSelfAttention):
    """
    WanT2VCrossAttention 类继承自 WanSelfAttention，用于实现文本到视频（Text-to-Video, T2V）的交叉注意力机制。

    该类扩展了自注意力机制，添加了处理文本到视频的交叉注意力功能。
    """
    def forward(self, x, context, context_lens):
        """
        前向传播过程，应用文本到视频的交叉注意力机制。

        参数:
            x (torch.Tensor): 查询输入张量，形状为 [B, L1, C]。
            context (torch.Tensor): 上下文输入张量，形状为 [B, L2, C]。
            context_lens (torch.Tensor): 每个上下文序列的实际长度，形状为 [B]。

        返回:
            torch.Tensor: 应用了文本到视频交叉注意力机制后的张量，形状为 [B, L1, C]。
        """
        # 获取批量大小、头的数量和每个头的维度
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # 计算查询 (q)、键 (k) 和值 (v)
        q = self.norm_q(self.q(x)).view(b, -1, n, d)  # 计算查询并进行归一化
        k = self.norm_k(self.k(context)).view(b, -1, n, d)  # 计算键并进行归一化
        v = self.v(context).view(b, -1, n, d)  # 计算值

        # 应用 Flash Attention 进行交叉注意力计算
        x = flash_attention(q, k, v, k_lens=context_lens)

        # 展平张量
        x = x.flatten(2)
        # 应用输出线性层
        x = self.o(x)
        # 返回输出张量
        return x


class WanI2VCrossAttention(WanSelfAttention):
    """
    WanI2VCrossAttention 类继承自 WanSelfAttention，用于实现图像到视频（Image-to-Video, I2V）的交叉注意力机制。

    该类扩展了自注意力机制，添加了处理图像到视频的交叉注意力功能，并引入了图像上下文。
    """
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        """
        初始化 WanI2VCrossAttention。

        参数:
            dim (int): 模型的维度。
            num_heads (int): 注意力头的数量。
            window_size (Tuple[int, int], 可选): 滑动窗口局部注意力的窗口大小，默认为 (-1, -1)。
            qk_norm (bool, 可选): 是否对查询和键进行归一化，默认为 True。
            eps (float, 可选): 层归一化的 epsilon，默认为1e-6。
        """
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        # 图像键线性层
        self.k_img = nn.Linear(dim, dim)
        # 图像值线性层
        self.v_img = nn.Linear(dim, dim)
        # 可学习的参数（未使用）
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        # 图像键归一化
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        """
        前向传播过程，应用图像到视频的交叉注意力机制。

        参数:
            x (torch.Tensor): 查询输入张量，形状为 [B, L1, C]。
            context (torch.Tensor): 上下文输入张量，形状为 [B, L2, C]。
            context_lens (torch.Tensor): 每个上下文序列的实际长度，形状为 [B]。

        返回:
            torch.Tensor: 应用了图像到视频交叉注意力机制后的张量，形状为 [B, L1, C]。
        """
        # 计算图像上下文长度
        image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
        # 分离图像上下文
        context_img = context[:, :image_context_length]
        # 分离文本上下文
        context = context[:, image_context_length:]
        # 获取批量大小、头的数量和每个头的维度
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # 计算查询 (q)、键 (k) 和值 (v)
        q = self.norm_q(self.q(x)).view(b, -1, n, d)  # 计算查询并进行归一化
        # 计算键并进行归一化
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        # 计算值
        v = self.v(context).view(b, -1, n, d)
        # 计算图像键并进行归一化
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        # 计算图像值
        v_img = self.v_img(context_img).view(b, -1, n, d)
        # 使用 Flash Attention 进行图像注意力计算
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # 使用 Flash Attention 进行文本注意力计算
        x = flash_attention(q, k, v, k_lens=context_lens)

        # 展平张量
        x = x.flatten(2)
        # 展平图像注意力输出
        img_x = img_x.flatten(2)
        # 将图像注意力输出与文本注意力输出相加
        x = x + img_x
        # 应用输出线性层
        x = self.o(x)
        # 返回输出张量
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,  # 文本到视频交叉注意力类
    'i2v_cross_attn': WanI2VCrossAttention,  # 图像到视频交叉注意力类
}


class WanAttentionBlock(nn.Module):
    """
    WanAttentionBlock 类实现了包含自注意力和交叉注意力的注意力块。

    该类结合了自注意力机制和交叉注意力机制，并支持不同的归一化位置和归一化类型。
    """
    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        """
        初始化 WanAttentionBlock。

        参数:
            cross_attn_type (str): 交叉注意力的类型。
            dim (int): 模型的维度。
            ffn_dim (int): 前馈神经网络中间层的维度。
            num_heads (int): 注意力头的数量。
            window_size (Tuple[int, int], 可选): 滑动窗口局部注意力的窗口大小，默认为 (-1, -1)。
            qk_norm (bool, 可选): 是否对查询和键进行归一化，默认为 True。
            cross_attn_norm (bool, 可选): 是否对交叉注意力进行归一化，默认为 False。
            eps (float, 可选): 层归一化的 epsilon，默认为1e-6。
        """
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # 自注意力归一化
        self.norm1 = WanLayerNorm(dim, eps)
        # 自注意力机制
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,  # 交叉注意力归一化
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        # 交叉注意力机制
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        # FFN 归一化
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),  # 应用 GELU 激活函数
            nn.Linear(ffn_dim, dim))

        # 调制参数，形状为 [1, 6, dim]
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        """
        前向传播过程，应用自注意力和交叉注意力机制。

        参数:
            x (torch.Tensor): 输入张量，形状为 [B, L, C]。
            e (torch.Tensor): 调制张量，形状为 [B, 6, C]。
            seq_lens (torch.Tensor): 序列长度张量，形状为 [B]。
            grid_sizes (torch.Tensor): 网格大小张量，形状为 [B, 3]。
            freqs (torch.Tensor): 旋转位置编码参数，形状为 [1024, C / num_heads / 2]。
            context (torch.Tensor): 上下文张量，形状为 [B, L', C]。
            context_lens (torch.Tensor): 上下文序列长度，形状为 [B]。

        返回:
            torch.Tensor: 输出张量，形状为 [B, L, C]。
        """
        # 确保调制张量的数据类型为 float32
        assert e.dtype == torch.float32
        # 使用自动混合精度
        with amp.autocast(dtype=torch.float32):
            # 将调制张量拆分为6部分
            e = (self.modulation + e).chunk(6, dim=1)
        # 确保拆分后的张量数据类型为 float32
        assert e[0].dtype == torch.float32

        # 自注意力机制
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes,  # 应用归一化、调制和偏置
            freqs)
        with amp.autocast(dtype=torch.float32):  # 使用自动混合精度
            x = x + y * e[2]  # 应用残差连接和调制
 
        # 交叉注意力与前馈神经网络
        def cross_attn_ffn(x, context, context_lens, e):
            # 应用交叉注意力
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            # 应用前馈神经网络
            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            # 使用自动混合精度
            with amp.autocast(dtype=torch.float32):
                # 应用残差连接和调制
                x = x + y * e[5]
            return x

        # 应用交叉注意力与前馈神经网络
        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):
    """
    Head 类实现了模型的头部，用于将模型的输出映射到目标维度，并应用调制（modulation）。

    该头部通过线性层将输入映射到目标维度，并在映射过程中应用调制参数以增强模型的表达能力。
    """
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        """
        初始化 Head。

        参数:
            dim (int): 输入的维度。
            out_dim (int): 输出的维度。
            patch_size (Tuple[int, int, int]): 3D 图像块的尺寸 (t_patch, h_patch, w_patch)。
            eps (float, 可选): 层归一化的 epsilon，默认为1e-6。
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # 计算输出维度为 patch维度与图像块尺寸的乘积
        out_dim = math.prod(patch_size) * out_dim
        # 层归一化
        self.norm = WanLayerNorm(dim, eps)
        # 线性层，将输入映射到输出维度
        self.head = nn.Linear(dim, out_dim)

        # 调制参数，形状为 [1, 2, dim]
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        """
        前向传播过程，应用头部并进行调制。

        参数:
            x (torch.Tensor): 输入张量，形状为 [B, L1, C]。
            e (torch.Tensor): 调制张量，形状为 [B, C]。

        返回:
            torch.Tensor: 应用了头部和调制后的张量，形状为 [B, L1, out_dim]。
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):  # 使用自动混合精度
            # 将调制张量拆分为两部分
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            # 应用归一化、调制和头部
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):
    """
    MLPProj 类实现了多层感知机（MLP）投影，用于处理图像嵌入。

    该类通过一系列线性层和激活函数将图像嵌入映射到目标维度，并可选择性地添加位置嵌入。
    """
    def __init__(self, in_dim, out_dim, flf_pos_emb=False):
        """
        初始化 MLPProj。

        参数:
            in_dim (int): 输入的维度。
            out_dim (int): 输出的维度。
            flf_pos_emb (bool, 可选): 是否使用首尾帧位置嵌入，默认为 False。
        """
        super().__init__()

        # 定义多层感知机
        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))
        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            # 首尾帧位置嵌入
            self.emb_pos = nn.Parameter(
                torch.zeros(1, FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER, 1280))  # 首尾帧位置嵌入参数

    def forward(self, image_embeds):
        """
        前向传播过程，应用 MLP 投影。

        参数:
            image_embeds (torch.Tensor): 输入的图像嵌入张量。

        返回:
            torch.Tensor: 投影后的图像嵌入张量。
        """
        if hasattr(self, 'emb_pos'):
            # 获取批量大小、序列长度和维度
            bs, n, d = image_embeds.shape
            # 重塑张量形状
            image_embeds = image_embeds.view(-1, 2 * n, d)
            # 添加位置嵌入
            image_embeds = image_embeds + self.emb_pos
        # 应用多层感知机
        clip_extra_context_tokens = self.proj(image_embeds)
        # 返回投影后的图像嵌入
        return clip_extra_context_tokens


class WanModel(ModelMixin, ConfigMixin):
    """
    WanModel 类实现了扩散模型的主干，支持文本到视频（Text-to-Video, T2V）、图像到视频（Image-to-Video, I2V）以及首尾帧到视频（First-Last-Frame-to-Video, FLF2V）等模式。
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ] # 在配置中忽略的参数
    _no_split_modules = ['WanAttentionBlock']  # 不进行分片的模块列表
    
    # 注册到配置中
    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        """
        初始化扩散模型主干。

        参数:
            model_type (str, 可选): 模型类型，'t2v' (文本到视频), 'i2v' (图像到视频), 'flf2v' (首尾帧到视频), 'vace'，默认为 't2v'。
            patch_size (Tuple[int, int, int], 可选): 3D 图像块尺寸 (t_patch, h patch, w patch)，默认为 (1, 2, 2)。
            text_len (int, 可选): 文本嵌入的固定长度，默认为512。
            in_dim (int, 可选): 输入视频通道数 (C_in)，默认为16。
            dim (int, 可选): 变换器的隐藏维度，默认为2048。
            ffn_dim (int, 可选): 前馈神经网络中间层的维度，默认为8192。
            freq_dim (int, 可选): 正弦时间嵌入的维度，默认为256。
            text_dim (int, 可选): 文本嵌入的输入维度，默认为4096。
            out_dim (int, 可选): 输出视频通道数 (C_out)，默认为16。
            num_heads (int, 可选): 注意力头的数量，默认为16。
            num_layers (int, 可选): 变换器层的数量，默认为32。
            window_size (Tuple[int, int], 可选): 局部注意力的窗口大小 (-1 表示全局注意力)，默认为 (-1, -1)。
            qk_norm (bool, 可选): 是否启用查询/键归一化，默认为 True。
            cross_attn_norm (bool, 可选): 是否启用交叉注意力归一化，默认为 False。
            eps (float, 可选): 归一化层的 epsilon，默认为1e-6。
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'flf2v', 'vace']
        # 设置模型类型
        self.model_type = model_type

        self.patch_size = patch_size  # 3D 图像块尺寸
        self.text_len = text_len  # 文本嵌入长度
        self.in_dim = in_dim  # 输入视频通道数
        self.dim = dim  # 变换器隐藏维度
        self.ffn_dim = ffn_dim  # 前馈神经网络中间层维度
        self.freq_dim = freq_dim  # 正弦时间嵌入维度
        self.text_dim = text_dim  # 文本嵌入输入维度
        self.out_dim = out_dim  # 输出视频通道数
        self.num_heads = num_heads  # 注意力头数量
        self.num_layers = num_layers  # 变换器层数量
        self.window_size = window_size  # 局部注意力窗口大小
        self.qk_norm = qk_norm  # 是否启用查询/键归一化
        self.cross_attn_norm = cross_attn_norm  # 是否启用交叉注意力归一化
        self.eps = eps  # 归一化层 epsilon

        # 嵌入层
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)  # 3D 卷积嵌入
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),  # 线性层，将文本嵌入映射到模型维度
            nn.Linear(dim, dim))  # 线性层，将模型维度映射回模型维度

        # 时间嵌入
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        # 对时间嵌入进行投影
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # 根据模型类型选择交叉注意力类型
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,  # 初始化注意力块列表
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # 初始化头部
        self.head = Head(dim, out_dim, patch_size, eps)

        # 确保维度可以被头的数量整除，并且每个头的维度是偶数
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        # 计算每个头的维度
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),  # 生成旋转位置编码参数
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        if model_type == 'i2v' or model_type == 'flf2v':
            # 初始化图像嵌入 MLP
            self.img_emb = MLPProj(1280, dim, flf_pos_emb=model_type == 'flf2v')

        # 初始化模型权重
        self.init_weights()

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
    ):
        """
        扩散模型的前向传播过程。

        参数:
            x (List[torch.Tensor]): 输入的视频张量列表，每个张量形状为 [C_in, F, H, W]。
            t (torch.Tensor): 扩散时间步张量，形状为 [B]。
            context (List[torch.Tensor]): 文本嵌入列表，每个张量形状为 [L, C]。
            seq_len (int): 序列的最大长度，用于位置编码。
            clip_fea (Optional[torch.Tensor], 可选): CLIP 图像特征，用于图像到视频模式或首尾帧到视频模式。
            y (Optional[List[torch.Tensor]], 可选): 条件视频输入，用于图像到视频模式，形状与 x 相同。

        返回:
            List[torch.Tensor]: 去噪后的视频张量列表，原始输入形状为 [C_out, F, H / 8, W / 8]。
        """
        if self.model_type == 'i2v' or self.model_type == 'flf2v':
            # 如果模型类型为 'i2v' 或 'flf2v'，则确保提供了 CLIP 图像特征和条件视频输入
            assert clip_fea is not None and y is not None
        # **设备设置**: 获取设备信息，并将频率张量移动到设备
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            # **条件视频输入处理**: 如果提供了条件视频输入，则将它们与输入视频张量拼接
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # **嵌入层**: 对每个输入视频张量应用 3D 卷积嵌入
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        # **网格大小**: 获取每个视频帧的网格大小
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        # **重塑与转置**: 将嵌入后的张量展平并转置
        x = [u.flatten(2).transpose(1, 2) for u in x]
        # **序列长度**: 计算每个视频帧的序列长度
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        # **填充**: 确保序列长度不超过最大序列长度
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # **时间嵌入**: 生成时间嵌入并进行投影
        with amp.autocast(dtype=torch.float32):
            # 生成时间嵌入
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            # 对时间嵌入进行投影
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            # 确保数据类型为 float32
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # **文本嵌入**: 对文本嵌入进行嵌入层处理
        context_lens = None  # 初始化文本长度
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            # **CLIP 图像特征处理**: 如果提供了 CLIP 图像特征，则对它们进行嵌入，并与文本嵌入拼接
            context_clip = self.img_emb(clip_fea)  # bs x 257 (x2) x dim
            context = torch.concat([context_clip, context], dim=1)  # 将 CLIP 图像特征与文本嵌入拼接

        # **参数处理**: 设置关键字参数，包括时间嵌入、序列长度、网格大小、频率、文本嵌入和文本长度
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)

        # **应用注意力块**: 遍历每个注意力块，应用注意力块
        for block in self.blocks:
            x = block(x, **kwargs)

        # **头部**: 应用头部
        x = self.head(x, e)

        # **去块化**: 对嵌入后的张量进行去块化
        x = self.unpatchify(x, grid_sizes)
        # 返回去噪后的视频张量列表
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        """
        从图像块嵌入中重构视频张量。

        参数:
            x (List[torch.Tensor]): 图像块特征列表，每个张量形状为 [L, C_out * prod(patch_size)]。
            grid_sizes (torch.Tensor): 原始的空间-时间网格尺寸，在分块之前，形状为 [B, 3]（3 个维度对应 F_patches, H_patches, W_patches）。

        返回:
            List[torch.Tensor]: 重构后的视频张量，形状为 [C_out, F, H / 8, W / 8]。
        """
        # 获取输出维度
        c = self.out_dim
        # 初始化输出列表
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            # **重塑张量**: 将张量重塑为 [F_patches, H_patches, W_patches, patch_size, patch_size, C_out]
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            # **重排维度**: 使用爱因斯坦求和约定重新排列维度
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            # **重塑张量**: 将张量重塑为 [C_out, F, H, W]
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            # 将重构后的张量添加到输出列表中
            out.append(u)
        # 返回重构后的视频张量列表
        return out

    def init_weights(self):
        """
        使用 Xavier 初始化方法初始化模型参数。
        """
        # **基础初始化**: 对所有线性层进行 Xavier 初始化，偏置初始化为零
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # **嵌入层初始化**: 对嵌入层进行 Xavier 初始化
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                # 对文本嵌入线性层的权重进行正态初始化
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                # 对时间嵌入线性层的权重进行正态初始化
                nn.init.normal_(m.weight, std=.02)

        # **输出层初始化**: 对输出层的权重初始化为零
        nn.init.zeros_(self.head.head.weight)

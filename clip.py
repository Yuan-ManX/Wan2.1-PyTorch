# Modified from ``https://github.com/openai/CLIP'' and ``https://github.com/mlfoundations/open_clip''
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from .attention import flash_attention
from .tokenizers import HuggingfaceTokenizer
from .xlm_roberta import XLMRoberta

__all__ = [
    'XLMRobertaCLIP',
    'clip_xlm_roberta_vit_h_14',
    'CLIPModel',
]


def pos_interpolate(pos, seq_len):
    """
    对位置编码进行插值，以匹配目标序列长度。

    参数:
        pos (torch.Tensor): 输入的位置编码张量，形状为 [B, N, C]，其中 N 是当前序列长度。
        seq_len (int): 目标序列长度。

    返回:
        torch.Tensor: 插值后的位置编码张量，形状为 [B, seq_len, C]。
    """
    if pos.size(1) == seq_len:
        # 如果当前序列长度已经等于目标序列长度，则直接返回位置编码
        return pos
    else:
        # 计算当前网格大小（假设位置编码是二维网格）
        src_grid = int(math.sqrt(pos.size(1)))
        # 计算目标网格大小
        tar_grid = int(math.sqrt(seq_len))
        # 计算需要移除的元素数量，以使位置编码可以被网格大小整除
        n = pos.size(1) - src_grid * src_grid

        # 对位置编码进行插值处理
        return torch.cat([
            pos[:, :n],   # 保留前 n 个位置编码
            F.interpolate(
                pos[:, n:].float().reshape(1, src_grid, src_grid, -1).permute(
                    0, 3, 1, 2),  # 重塑并转置张量以适应插值函数
                size=(tar_grid, tar_grid),  # 设置目标网格大小
                mode='bicubic',  # 使用双三次插值方法
                align_corners=False).flatten(2).transpose(1, 2)  # 展平并转置回原始形状
        ],
                         dim=1)   # 在序列长度维度上拼接


class QuickGELU(nn.Module):
    """
    QuickGELU 激活函数实现。

    QuickGELU 是 GELU（高斯误差线性单元）的一种快速近似实现，通过对输入进行缩放和 sigmoid 激活来实现。
    """
    def forward(self, x):
        """
        对输入张量应用 QuickGELU 激活函数。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 应用了 QuickGELU 激活后的张量。
        """
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    """
    自定义的 LayerNorm 实现，支持将输入转换为 float 进行计算，然后转换回原始类型。

    这对于处理混合精度训练或不同数据类型时非常有用。
    """
    def forward(self, x):
        """
        对输入张量应用层归一化。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 应用了层归一化后的张量，类型与输入相同。
        """
        # 将输入转换为 float 进行层归一化，然后转换回原始类型
        return super().forward(x.float()).type_as(x)


class SelfAttention(nn.Module):
    """
    自注意力机制的实现。

    该类实现了多头自注意力机制，支持因果掩码（causal mask）和 Dropout。
    """
    def __init__(self,
                 dim,
                 num_heads,
                 causal=False,
                 attn_dropout=0.0,
                 proj_dropout=0.0):
        """
        初始化自注意力机制。

        参数:
            dim (int): 注意力机制的维度。
            num_heads (int): 注意力头的数量。
            causal (bool, 可选): 是否应用因果掩码，默认为 False。
            attn_dropout (float, 可选): 注意力 Dropout 概率，默认为 0.0。
            proj_dropout (float, 可选): 输出投影 Dropout 概率，默认为 0.0。
        """
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        self.attn_dropout = attn_dropout
        self.proj_dropout = proj_dropout

        # 定义线性层，将输入转换为查询、键和值
        self.to_qkv = nn.Linear(dim, dim * 3)
        # 定义输出投影线性层
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        前向传播过程，计算自注意力。

        参数:
            x (torch.Tensor): 输入张量，形状为 [B, L, C]，其中:
                - B: 批量大小
                - L: 序列长度
                - C: 通道数

        返回:
            torch.Tensor: 注意力计算后的输出张量，形状为 [B, L, C]。
        """
        # 获取张量的维度信息
        b, s, c, n, d = *x.size(), self.num_heads, self.head_dim

        # 计算查询 (q)、键 (k) 和值 (v)
        q, k, v = self.to_qkv(x).view(b, s, 3, n, d).unbind(2)  # 将线性层输出重塑并拆分为 q, k, v

        # 应用注意力机制
        # 设置 Dropout 概率，如果处于训练模式，则使用指定的概率，否则为0
        p = self.attn_dropout if self.training else 0.0
        # 使用 Flash Attention v2 进行注意力计算
        x = flash_attention(q, k, v, dropout_p=p, causal=self.causal, version=2)
        # 重塑张量形状为 [B, L, C]
        x = x.reshape(b, s, c)

        # 应用输出投影
        x = self.proj(x)
        # 应用 Dropout
        x = F.dropout(x, self.proj_dropout, self.training)
        return x


class SwiGLU(nn.Module):
    """
    SwiGLU (Swish-Gated Linear Unit) 激活函数实现。

    SwiGLU 是 GLU（Gated Linear Unit）的一种变体，使用 Swish 激活函数代替传统的 ReLU 激活函数。
    它通过门控机制来控制信息的流动，从而提高模型的表达能力。
    """

    def __init__(self, dim, mid_dim):
        """
        初始化 SwiGLU 模块。

        参数:
            dim (int): 输入和输出的维度。
            mid_dim (int): 中间层的维度，通常设置为 dim 的倍数。
        """
        super().__init__()
        self.dim = dim
        self.mid_dim = mid_dim

        # 定义线性层，将输入映射到中间维度
        self.fc1 = nn.Linear(dim, mid_dim)
        # 定义第二个线性层，同样将输入映射到中间维度
        self.fc2 = nn.Linear(dim, mid_dim)
        # 定义输出线性层，将中间维度映射回原始维度
        self.fc3 = nn.Linear(mid_dim, dim)

    def forward(self, x):
        """
        前向传播过程，应用 SwiGLU 激活函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 [B, L, dim]。

        返回:
            torch.Tensor: 应用了 SwiGLU 激活后的张量，形状为 [B, L, dim]。
        """
        # 对输入进行第一个线性变换，并应用 Swish 激活函数
        x = F.silu(self.fc1(x)) * self.fc2(x)  # 将两个结果相乘，实现门控机制
        # 对结果进行输出线性变换，恢复到原始维度
        x = self.fc3(x) # [B, L, dim]
        return x


class AttentionBlock(nn.Module):
    """
    注意力块（Attention Block）实现。

    该模块结合了自注意力机制和前馈神经网络（MLP），并支持不同的归一化位置和激活函数。
    """
    def __init__(self,
                 dim,
                 mlp_ratio,
                 num_heads,
                 post_norm=False,
                 causal=False,
                 activation='quick_gelu',
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 norm_eps=1e-5):
        """
        初始化注意力块。

        参数:
            dim (int): 注意力块的维度。
            mlp_ratio (float): MLP 中间层的缩放比例。
            num_heads (int): 注意力头的数量。
            post_norm (bool, 可选): 是否在注意力机制和 MLP 之后应用归一化，默认为 False。
            causal (bool, 可选): 是否应用因果掩码，默认为 False。
            activation (str, 可选): 激活函数类型，默认为 'quick_gelu'。可选 'quick_gelu', 'gelu', 'swi_glu'。
            attn_dropout (float, 可选): 注意力 Dropout 概率，默认为 0.0。
            proj_dropout (float, 可选): 输出投影 Dropout 概率，默认为 0.0。
            norm_eps (float, 可选): 层归一化的 epsilon，默认为 1e-5。
        """
        assert activation in ['quick_gelu', 'gelu', 'swi_glu']
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.post_norm = post_norm
        self.causal = causal
        self.norm_eps = norm_eps

        # 定义层归一化
        self.norm1 = LayerNorm(dim, eps=norm_eps)
        # 定义自注意力机制
        self.attn = SelfAttention(dim, num_heads, causal, attn_dropout,
                                  proj_dropout)
        # 定义第二个层归一化
        self.norm2 = LayerNorm(dim, eps=norm_eps)
        # 根据激活函数类型，定义 MLP
        if activation == 'swi_glu':
            self.mlp = SwiGLU(dim, int(dim * mlp_ratio))
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim, int(dim * mlp_ratio)),
                QuickGELU() if activation == 'quick_gelu' else nn.GELU(),
                nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(proj_dropout))

    def forward(self, x):
        """
        前向传播过程，应用注意力块。

        参数:
            x (torch.Tensor): 输入张量，形状为 [B, L, dim]。

        返回:
            torch.Tensor: 应用了注意力块后的张量，形状为 [B, L, dim]。
        """
        if self.post_norm:
            # 如果启用后归一化，则先进行注意力计算，再归一化，再进行 MLP，再归一化
            x = x + self.norm1(self.attn(x))  # 注意力机制 + 归一化
            x = x + self.norm2(self.mlp(x))  # MLP + 归一化
        else:
            # 如果未启用后归一化，则先归一化，再进行注意力计算，再进行 MLP
            x = x + self.attn(self.norm1(x))  # 归一化 + 注意力机制
            x = x + self.mlp(self.norm2(x))  # MLP + 归一化
        return x


class AttentionPool(nn.Module):
    """
    注意力池化（Attention Pool）实现。

    该模块使用自注意力机制对输入序列进行池化，通常用于生成全局表示。
    """
    def __init__(self,
                 dim,
                 mlp_ratio,
                 num_heads,
                 activation='gelu',
                 proj_dropout=0.0,
                 norm_eps=1e-5):
        """
        初始化注意力池化模块。

        参数:
            dim (int): 注意力池化的维度。
            mlp_ratio (float): MLP 中间层的缩放比例。
            num_heads (int): 注意力头的数量。
            activation (str, 可选): 激活函数类型，默认为 'gelu'。可选 'quick_gelu', 'gelu'。
            proj_dropout (float, 可选): 输出投影 Dropout 概率，默认为 0.0。
            norm_eps (float, 可选): 层归一化的 epsilon，默认为 1e-5。
        """
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.proj_dropout = proj_dropout
        self.norm_eps = norm_eps

        # 定义增益，用于初始化分类器嵌入
        gain = 1.0 / math.sqrt(dim)
        # 分类器嵌入参数
        self.cls_embedding = nn.Parameter(gain * torch.randn(1, 1, dim))
        # 定义查询线性层
        self.to_q = nn.Linear(dim, dim)
        # 定义键和值线性层
        self.to_kv = nn.Linear(dim, dim * 2)
        # 定义输出投影线性层
        self.proj = nn.Linear(dim, dim)
        # 定义层归一化
        self.norm = LayerNorm(dim, eps=norm_eps)
        # 定义 MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            QuickGELU() if activation == 'quick_gelu' else nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(proj_dropout))

    def forward(self, x):
        """
        前向传播过程，应用注意力池化。

        参数:
            x (torch.Tensor): 输入张量，形状为 [B, L, dim]。

        返回:
            torch.Tensor: 池化后的张量，形状为 [B, dim]。
        """
        # 获取张量的维度信息
        b, s, c, n, d = *x.size(), self.num_heads, self.head_dim

        # 计算查询 (q)、键 (k) 和值 (v)
        q = self.to_q(self.cls_embedding).view(1, 1, n, d).expand(b, -1, -1, -1) # 扩展分类器嵌入以匹配批量大小
        # 拆分键和值
        k, v = self.to_kv(x).view(b, s, 2, n, d).unbind(2)

        # 应用注意力机制
        x = flash_attention(q, k, v, version=2)  # 使用 Flash Attention v2
        x = x.reshape(b, 1, c)  # 重塑张量形状为 [B, 1, dim]

        # 应用输出投影
        x = self.proj(x)
        # 应用 Dropout
        x = F.dropout(x, self.proj_dropout, self.training)

        # 应用 MLP 和归一化
        x = x + self.mlp(self.norm(x))
        # 返回分类器嵌入的输出
        return x[:, 0]


class VisionTransformer(nn.Module):
    """
    视觉变换器（Vision Transformer，ViT）实现。

    该类实现了基于变换器的图像分类模型，通过将图像分割成小块（patch），然后应用变换器编码器进行处理。
    """
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 dim=768,
                 mlp_ratio=4,
                 out_dim=512,
                 num_heads=12,
                 num_layers=12,
                 pool_type='token',
                 pre_norm=True,
                 post_norm=False,
                 activation='quick_gelu',
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 embedding_dropout=0.0,
                 norm_eps=1e-5):
        """
        初始化视觉变换器。

        参数:
            image_size (int, 可选): 输入图像的尺寸（高度和宽度），默认为224。
            patch_size (int, 可选): 图像块的大小，默认为16。
            dim (int, 可选): 模型的维度，默认为768。
            mlp_ratio (float, 可选): MLP 中间层的缩放比例，默认为4。
            out_dim (Optional[int], 可选): 输出维度。如果为None，则使用 dim，默认为512。
            num_heads (int, 可选): 注意力头的数量，默认为12。
            num_layers (int, 可选): 变换器层的数量，默认为12。
            pool_type (str, 可选): 池化类型，可选 'token', 'token_fc', 'attn_pool'，默认为 'token'。
            pre_norm (bool, 可选): 是否在变换器层之前应用层归一化，默认为True。
            post_norm (bool, 可选): 是否在变换器层之后应用层归一化，默认为False。
            activation (str, 可选): 激活函数类型，可选 'quick_gelu', 'gelu', 'swi_glu'，默认为 'quick_gelu'。
            attn_dropout (float, 可选): 注意力 Dropout 概率，默认为0.0。
            proj_dropout (float, 可选): 输出投影 Dropout 概率，默认为0.0。
            embedding_dropout (float, 可选): 嵌入 Dropout 概率，默认为0.0。
            norm_eps (float, 可选): 层归一化的 epsilon，默认为1e-5。
        """
        # 检查图像尺寸是否可以被块大小整除
        if image_size % patch_size != 0:
            print(
                '[WARNING] image_size is not divisible by patch_size',
                flush=True)
        # 确保池化类型有效
        assert pool_type in ('token', 'token_fc', 'attn_pool')
        # 如果未提供输出维度，则使用模型维度
        out_dim = out_dim or dim
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size)**2  # 计算图像块的数量
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pool_type = pool_type
        self.post_norm = post_norm
        self.norm_eps = norm_eps

        # 嵌入层
        gain = 1.0 / math.sqrt(dim)
        # 使用卷积层将图像分割成块，并进行线性投影
        self.patch_embedding = nn.Conv2d(
            3,  # 输入通道数（RGB）
            dim,  # 输出通道数
            kernel_size=patch_size,  # 卷积核大小
            stride=patch_size,  # 卷积步幅
            bias=not pre_norm)   # 如果未启用预归一化，则使用偏置
        
        # 如果池化类型为 'token' 或 'token_fc'，则添加分类器嵌入
        if pool_type in ('token', 'token_fc'):
            self.cls_embedding = nn.Parameter(gain * torch.randn(1, 1, dim))
        # 添加位置嵌入
        self.pos_embedding = nn.Parameter(gain * torch.randn(
            1, self.num_patches +
            (1 if pool_type in ('token', 'token_fc') else 0), dim))
        # 添加嵌入 Dropout
        self.dropout = nn.Dropout(embedding_dropout)

        # 变换器编码器
        # 如果启用预归一化，则添加层归一化
        self.pre_norm = LayerNorm(dim, eps=norm_eps) if pre_norm else None
        # 构建变换器层序列
        self.transformer = nn.Sequential(*[
            AttentionBlock(dim, mlp_ratio, num_heads, post_norm, False,
                           activation, attn_dropout, proj_dropout, norm_eps)
            for _ in range(num_layers)
        ])
        # 添加后归一化
        self.post_norm = LayerNorm(dim, eps=norm_eps)

        # 头部
        if pool_type == 'token':
            # 如果池化类型为 'token'，则使用线性投影作为头部
            self.head = nn.Parameter(gain * torch.randn(dim, out_dim))
        elif pool_type == 'token_fc':
            # 如果池化类型为 'token_fc'，则使用线性层作为头部
            self.head = nn.Linear(dim, out_dim)
        elif pool_type == 'attn_pool':
            # 如果池化类型为 'attn_pool'，则使用注意力池化作为头部
            self.head = AttentionPool(dim, mlp_ratio, num_heads, activation,
                                      proj_dropout, norm_eps)

    def forward(self, x, interpolation=False, use_31_block=False):
        """
        前向传播过程。

        参数:
            x (torch.Tensor): 输入图像张量，形状为 [B, C, H, W]。
            interpolation (bool, 可选): 是否对位置嵌入进行插值，默认为 False。
            use_31_block (bool, 可选): 是否只使用前31个变换器块，默认为 False。

        返回:
            torch.Tensor: 输出张量，形状为 [B, out_dim]。
        """
        # 获取批量大小
        b = x.size(0)

        # embeddings
        # 将图像分割成块并进行线性投影
        x = self.patch_embedding(x).flatten(2).permute(0, 2, 1)
        if self.pool_type in ('token', 'token_fc'):
            # 如果池化类型为 'token' 或 'token_fc'，则在序列开头添加分类器嵌入
            x = torch.cat([self.cls_embedding.expand(b, -1, -1), x], dim=1)
        if interpolation:
            # 如果启用了插值，则对位置嵌入进行插值以匹配序列长度
            e = pos_interpolate(self.pos_embedding, x.size(1))
        else:
            # 否则，直接使用位置嵌入
            e = self.pos_embedding
        # 添加位置嵌入并进行 Dropout
        x = self.dropout(x + e)
        if self.pre_norm is not None:
            # 应用预归一化
            x = self.pre_norm(x)

        # transformer
        if use_31_block:
            # 如果只使用前31个变换器块，则只应用前31个块
            x = self.transformer[:-1](x)
            return x
        else:
            # 否则，应用所有变换器块
            x = self.transformer(x)
            return x


class XLMRobertaWithHead(XLMRoberta):
    """
    XLM-RoBERTa 模型扩展，添加了一个自定义头部用于特定任务。
    """
    def __init__(self, **kwargs):
        """
        初始化 XLMRobertaWithHead 模型。

        参数:
            **kwargs: 其他参数，包括输出维度（out_dim）。
        """
        # 获取输出维度
        self.out_dim = kwargs.pop('out_dim')
        super().__init__(**kwargs)

        # head
        # 计算中间维度
        mid_dim = (self.dim + self.out_dim) // 2
        self.head = nn.Sequential(  
            nn.Linear(self.dim, mid_dim, bias=False), nn.GELU(),  # 线性层，映射到中间维度
            nn.Linear(mid_dim, self.out_dim, bias=False))  # 线性层，映射到输出维度

    def forward(self, ids):
        """
        前向传播过程。

        参数:
            ids (torch.Tensor): 输入的 token ID 张量。

        返回:
            torch.Tensor: 输出张量，形状为 [B, out_dim]。
        """
        # xlm-roberta
        x = super().forward(ids)

        # average pooling
        mask = ids.ne(self.pad_id).unsqueeze(-1).to(x)  # 创建掩码，忽略填充 token
        x = (x * mask).sum(dim=1) / mask.sum(dim=1)  # 对 token 进行加权平均

        # head
        # 应用自定义头部
        x = self.head(x)
        return x


class XLMRobertaCLIP(nn.Module):
    """
    XLMRobertaCLIP 类实现了 CLIP（Contrastive Language-Image Pre-Training）架构，
    结合了视觉变换器（ViT）和扩展的 XLM-RoBERTa 模型，用于联合学习图像和文本表示。
    """
    def __init__(self,
                 embed_dim=1024,
                 image_size=224,
                 patch_size=14,
                 vision_dim=1280,
                 vision_mlp_ratio=4,
                 vision_heads=16,
                 vision_layers=32,
                 vision_pool='token',
                 vision_pre_norm=True,
                 vision_post_norm=False,
                 activation='gelu',
                 vocab_size=250002,
                 max_text_len=514,
                 type_size=1,
                 pad_id=1,
                 text_dim=1024,
                 text_heads=16,
                 text_layers=24,
                 text_post_norm=True,
                 text_dropout=0.1,
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 embedding_dropout=0.0,
                 norm_eps=1e-5):
        """
        初始化 XLMRobertaCLIP 模型。

        参数:
            embed_dim (int, 可选): 嵌入向量的维度，默认为1024。
            image_size (int, 可选): 输入图像的尺寸（高度和宽度），默认为224。
            patch_size (int, 可选): 图像块的大小，默认为14。
            vision_dim (int, 可选): 视觉模型的维度，默认为1280。
            vision_mlp_ratio (float, 可选): 视觉模型中 MLP 的缩放比例，默认为4。
            vision_heads (int, 可选): 视觉模型中注意力头的数量，默认为16。
            vision_layers (int, 可选): 视觉模型中变换器层的数量，默认为32。
            vision_pool (str, 可选): 视觉模型的池化类型，可选 'token', 'token_fc', 'attn_pool'，默认为 'token'。
            vision_pre_norm (bool, 可选): 视觉模型是否在变换器层之前应用层归一化，默认为True。
            vision_post_norm (bool, 可选): 视觉模型是否在变换器层之后应用层归一化，默认为False。
            activation (str, 可选): 激活函数类型，可选 'quick_gelu', 'gelu', 'swi_glu'，默认为 'gelu'。
            vocab_size (int, 可选): 词汇表大小，默认为250002。
            max_text_len (int, 可选): 文本的最大长度，默认为514。
            type_size (int, 可选): 类型嵌入的大小，默认为1。
            pad_id (int, 可选): 填充 token 的 ID，默认为1。
            text_dim (int, 可选): 文本模型的维度，默认为1024。
            text_heads (int, 可选): 文本模型中注意力头的数量，默认为16。
            text_layers (int, 可选): 文本模型中变换器层的数量，默认为24。
            text_post_norm (bool, 可选): 文本模型是否在变换器层之后应用层归一化，默认为True。
            text_dropout (float, 可选): 文本模型的 Dropout 概率，默认为0.1。
            attn_dropout (float, 可选): 注意力 Dropout 概率，默认为0.0。
            proj_dropout (float, 可选): 输出投影 Dropout 概率，默认为0.0。
            embedding_dropout (float, 可选): 嵌入 Dropout 概率，默认为0.0。
            norm_eps (float, 可选): 层归一化的 epsilon，默认为1e-5。
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_dim = vision_dim
        self.vision_mlp_ratio = vision_mlp_ratio
        self.vision_heads = vision_heads
        self.vision_layers = vision_layers
        self.vision_pre_norm = vision_pre_norm
        self.vision_post_norm = vision_post_norm
        self.activation = activation
        self.vocab_size = vocab_size
        self.max_text_len = max_text_len
        self.type_size = type_size
        self.pad_id = pad_id
        self.text_dim = text_dim
        self.text_heads = text_heads
        self.text_layers = text_layers
        self.text_post_norm = text_post_norm
        self.norm_eps = norm_eps

        # 视觉模型
        self.visual = VisionTransformer(
            image_size=image_size,  # 输入图像尺寸
            patch_size=patch_size,  # 图像块大小
            dim=vision_dim,  # 视觉模型维度
            mlp_ratio=vision_mlp_ratio,  # MLP 缩放比例
            out_dim=embed_dim,  # 输出维度
            num_heads=vision_heads,  # 注意力头数量
            num_layers=vision_layers,  # 变换器层数量
            pool_type=vision_pool,  # 池化类型
            pre_norm=vision_pre_norm,  # 是否预归一化
            post_norm=vision_post_norm,  # 是否后归一化
            activation=activation,  # 激活函数类型
            attn_dropout=attn_dropout,  # 注意力 Dropout 概率
            proj_dropout=proj_dropout,  # 输出投影 Dropout 概率
            embedding_dropout=embedding_dropout,  # 嵌入 Dropout 概率
            norm_eps=norm_eps) # 层归一化 epsilon
        
        # 文本模型
        self.textual = XLMRobertaWithHead(
            vocab_size=vocab_size,  # 词汇表大小
            max_seq_len=max_text_len,  # 最大序列长度
            type_size=type_size,  # 类型嵌入大小
            pad_id=pad_id,  # 填充 token ID
            dim=text_dim,  # 文本模型维度
            out_dim=embed_dim,  # 输出维度
            num_heads=text_heads,  # 文本模型中注意力头数量
            num_layers=text_layers,  # 文本模型中变换器层数量
            post_norm=text_post_norm,  # 是否后归一化 
            dropout=text_dropout)  # Dropout 概率
        
        # 对数尺度参数
        self.log_scale = nn.Parameter(math.log(1 / 0.07) * torch.ones([])) # 对数尺度参数，用于对比学习

    def forward(self, imgs, txt_ids):
        """
        前向传播过程，计算图像和文本的嵌入表示。

        参数:
            imgs (torch.Tensor): 输入图像张量，形状为 [B, 3, H, W]，数据类型为 torch.float32。
                                 - 均值: [0.48145466, 0.4578275, 0.40821073]
                                 - 标准差: [0.26862954, 0.26130258, 0.27577711]
            txt_ids (torch.Tensor): 输入文本 ID 张量，形状为 [B, L]，数据类型为 torch.long。
                                    由 data.CLIPTokenizer 编码。

        返回:
            Tuple[torch.Tensor, torch.Tensor]: 图像和文本的嵌入表示，形状均为 [B, embed_dim]。
        """

        xi = self.visual(imgs)  # 使用视觉模型编码图像
        xt = self.textual(txt_ids)  # 使用文本模型编码文本
        # 返回图像和文本的嵌入表示
        return xi, xt

    def param_groups(self):
        """
        定义参数组，用于优化器的参数分组。

        返回:
            List[Dict[str, Any]]: 参数组列表，包含两个组:
                - 一组包含所有归一化层和偏置参数，权重衰减为0.0。
                - 另一组包含其他参数，权重衰减为默认值。
        """
        groups = [{
            'params': [
                p for n, p in self.named_parameters()
                if 'norm' in n or n.endswith('bias')  # 选择归一化层和偏置参数
            ],
            'weight_decay': 0.0   # 设置权重衰减为0.0
        }, {
            'params': [
                p for n, p in self.named_parameters() 
                if not ('norm' in n or n.endswith('bias'))  # 选择其他参数
            ]
        }]
        # 返回参数组列表
        return groups


def _clip(pretrained=False,
          pretrained_name=None,
          model_cls=XLMRobertaCLIP,
          return_transforms=False,
          return_tokenizer=False,
          tokenizer_padding='eos',
          dtype=torch.float32,
          device='cpu',
          **kwargs):
    """
    初始化 CLIP 模型，并根据参数返回相应的组件。

    参数:
        pretrained (bool, 可选): 是否加载预训练模型，默认为 False。
        pretrained_name (Optional[str], 可选): 预训练模型的名称，如果为 None，则不加载预训练权重。
        model_cls (class, 可选): 用于实例化模型的类，默认为 XLMRobertaCLIP。
        return_transforms (bool, 可选): 是否返回图像变换，默认为 False。
        return_tokenizer (bool, 可选): 是否返回分词器，默认为 False。
        tokenizer_padding (str, 可选): 分词器的填充策略，可选 'eos', 'max_length'，默认为 'eos'。
        dtype (torch.dtype, 可选): 模型的数据类型，默认为 torch.float32。
        device (str, 可选): 设备类型，可选 'cpu', 'cuda'，默认为 'cpu'。
        **kwargs: 其他关键字参数，用于传递给 model_cls。

    返回:
        Any: 根据参数返回相应的组件。如果 return_transforms 和 return_tokenizer 都为 False，则返回模型；否则，返回包含模型、变换和分词器的元组。
    """
    # 初始化模型
    with torch.device(device):
        model = model_cls(**kwargs)  # 使用提供的参数实例化模型类

    # 将模型移动到指定设备和设置数据类型
    model = model.to(dtype=dtype, device=device)
    output = (model,)

    # 初始化图像变换
    if return_transforms:
        # 设置均值和标准差
        if 'siglip' in pretrained_name.lower():
            # 如果预训练名称包含 'siglip'，则使用均值为0.5，标准差为0.5
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        else:
            mean = [0.48145466, 0.4578275, 0.40821073]  # 否则，使用默认的均值
            std = [0.26862954, 0.26130258, 0.27577711]  # 使用默认的标准差

        # 定义图像变换
        transforms = T.Compose([
            T.Resize((model.image_size, model.image_size),   # 调整图像大小到模型指定的尺寸
                     interpolation=T.InterpolationMode.BICUBIC),  # 使用双三次插值方法
            T.ToTensor(), # 将图像转换为张量
            T.Normalize(mean=mean, std=std) # 标准化图像数据
        ])
        output += (transforms,)  # 将变换添加到输出元组中
    return output[0] if len(output) == 1 else output


def clip_xlm_roberta_vit_h_14(
        pretrained=False,
        pretrained_name='open-clip-xlm-roberta-large-vit-huge-14',
        **kwargs):
    """
    配置并初始化 CLIP 模型，使用 XLM-RoBERTa 作为文本编码器，ViT-Huge-14 作为视觉编码器。

    参数:
        pretrained (bool, 可选): 是否加载预训练模型，默认为 False。
        pretrained_name (str, 可选): 预训练模型的名称，默认为 'open-clip-xlm-roberta-large-vit-huge-14'。
        **kwargs: 其他关键字参数，用于配置模型。

    返回:
        XLMRobertaCLIP: 配置好的 CLIP 模型实例。
    """
    cfg = dict(
        embed_dim=1024,  # 嵌入维度
        image_size=224,  # 图像尺寸
        patch_size=14,  # 图像块大小
        vision_dim=1280,  # 视觉模型维度
        vision_mlp_ratio=4,  # 视觉模型 MLP 缩放比例
        vision_heads=16,  # 视觉模型注意力头数量
        vision_layers=32,  # 视觉模型变换器层数量
        vision_pool='token',  # 视觉模型池化类型
        activation='gelu',  # 激活函数类型
        vocab_size=250002,  # 词汇表大小
        max_text_len=514,  # 最大文本长度
        type_size=1,  # 类型嵌入大小
        pad_id=1,  # 填充 token ID
        text_dim=1024,  # 文本模型维度
        text_heads=16,  # 文本模型注意力头数量
        text_layers=24,  # 文本模型变换器层数量
        text_post_norm=True,  # 文本模型是否后归一化
        text_dropout=0.1,  # 文本模型 Dropout 概率
        attn_dropout=0.0,  # 注意力 Dropout 概率
        proj_dropout=0.0,  # 输出投影 Dropout 概率
        embedding_dropout=0.0) # 嵌入 Dropout 概率
    # 更新配置参数
    cfg.update(**kwargs)
    # 调用 _clip 函数进行初始化
    return _clip(pretrained, pretrained_name, XLMRobertaCLIP, **cfg)


class CLIPModel:
    """
    CLIPModel 类用于加载和推理 CLIP 模型，包括视觉和文本编码器。
    """
    def __init__(self, dtype, device, checkpoint_path, tokenizer_path):
        """
        初始化 CLIPModel。

        参数:
            dtype (torch.dtype): 模型的数据类型。
            device (str): 设备类型，如 'cpu', 'cuda'。
            checkpoint_path (str): 模型检查点路径。
            tokenizer_path (str): 分词器路径。
        """
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        # 初始化模型和图像变换
        self.model, self.transforms = clip_xlm_roberta_vit_h_14(
            pretrained=False,
            return_transforms=True,
            return_tokenizer=False,
            dtype=dtype,
            device=device)
        # 设置模型为评估模式，不计算梯度
        self.model = self.model.eval().requires_grad_(False)
        # 输出日志信息，指示正在加载模型检查点
        logging.info(f'loading {checkpoint_path}')
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location='cpu'))  # 加载模型权重

        # 初始化分词器
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path,  # 分词器名称或路径
            seq_len=self.model.max_text_len - 2,  # 序列长度，减去2用于特殊 token
            clean='whitespace')  # 清理策略，这里使用空白字符清理

    def visual(self, videos):
        """
        对视频帧进行预处理，并计算视觉嵌入表示。

        参数:
            videos (List[torch.Tensor]): 输入的视频帧列表，每个视频为一个张量。

        返回:
            torch.Tensor: 视觉嵌入表示，形状为 [B, embed_dim]。
        """
        # 预处理
        size = (self.model.image_size,) * 2  # 设置图像尺寸
        videos = torch.cat([
            F.interpolate(
                u.transpose(0, 1),  # 转置张量以适应插值函数
                size=size,  # 设置插值后的尺寸
                mode='bicubic',  # 使用双三次插值方法
                align_corners=False) for u in videos  # 对每个视频帧进行插值
        ])
        videos = self.transforms.transforms[-1](videos.mul_(0.5).add_(0.5))  # 应用标准化变换

        # 前向传播
        with torch.cuda.amp.autocast(dtype=self.dtype):   # 使用自动混合精度
            # 使用视觉模型计算嵌入表示，只使用前31个变换器块
            out = self.model.visual(videos, use_31_block=True)
            # 返回视觉嵌入表示
            return out

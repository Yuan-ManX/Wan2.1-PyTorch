# Modified from transformers.models.t5.modeling_t5
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizers import HuggingfaceTokenizer

__all__ = [
    'T5Model',
    'T5Encoder',
    'T5Decoder',
    'T5EncoderModel',
]


def fp16_clamp(x):
    """
    对半精度浮点数（float16）进行裁剪，以防止出现无穷大值。

    参数:
        x (torch.Tensor): 输入的张量。

    返回:
        torch.Tensor: 裁剪后的张量。如果输入不是 float16 类型，则直接返回原张量。
    """
    if x.dtype == torch.float16 and torch.isinf(x).any():
        # 获取 float16 类型的最大有限值，并减去1000以避免极端值
        clamp = torch.finfo(x.dtype).max - 1000
        # 对张量进行裁剪，限制在 [-clamp, clamp] 范围内
        x = torch.clamp(x, min=-clamp, max=clamp)
    # 返回裁剪后的张量
    return x


def init_weights(m):
    """
    对模型的不同层进行权重初始化。

    参数:
        m (nn.Module): 需要初始化权重的模型层。
    """
    if isinstance(m, T5LayerNorm):
        # 如果是 T5LayerNorm 层，则将权重初始化为全1
        nn.init.ones_(m.weight)
    elif isinstance(m, T5Model):
        # 如果是 T5Model 模型，则将 token 嵌入权重初始化为均值为0，标准差为1.0的正态分布
        nn.init.normal_(m.token_embedding.weight, std=1.0)
    elif isinstance(m, T5FeedForward):
        # 如果是 T5FeedForward 层，则对不同的线性层进行权重初始化
        # gate 线性层的权重，标准差为 dim 的 -0.5 次方
        nn.init.normal_(m.gate[0].weight, std=m.dim**-0.5)
        # fc1 线性层的权重，标准差为 dim 的 -0.5 次方
        nn.init.normal_(m.fc1.weight, std=m.dim**-0.5)
        # fc2 线性层的权重，标准差为 dim_ffn 的 -0.5 次方
        nn.init.normal_(m.fc2.weight, std=m.dim_ffn**-0.5)
    elif isinstance(m, T5Attention):
        # 如果是 T5Attention 层，则对不同的线性层进行权重初始化
        # 查询 (q) 线性层的权重，标准差为 (dim * dim_attn) 的 -0.5 次方
        nn.init.normal_(m.q.weight, std=(m.dim * m.dim_attn)**-0.5)
        # 键 (k) 线性层的权重，标准差为 dim 的 -0.5 次方
        nn.init.normal_(m.k.weight, std=m.dim**-0.5)
        # 值 (v) 线性层的权重，标准差为 dim 的 -0.5 次方
        nn.init.normal_(m.v.weight, std=m.dim**-0.5)
        # 输出 (o) 线性层的权重，标准差为 (num_heads * dim_attn) 的 -0.5 次方
        nn.init.normal_(m.o.weight, std=(m.num_heads * m.dim_attn)**-0.5)
    elif isinstance(m, T5RelativeEmbedding):
        # 如果是 T5RelativeEmbedding 层，则对嵌入权重进行初始化
        nn.init.normal_(
            # 嵌入权重的标准差为 (2 * num_buckets * num_heads) 的 -0.5 次方
            m.embedding.weight, std=(2 * m.num_buckets * m.num_heads)**-0.5)


class GELU(nn.Module):
    """
    GELU（高斯误差线性单元）激活函数实现。
    """
    def forward(self, x):
        """
        对输入张量应用 GELU 激活函数。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 应用了 GELU 激活后的张量。
        """
        # 计算 GELU 激活函数
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class T5LayerNorm(nn.Module):
    """
    T5 风格的层归一化实现。

    T5LayerNorm 对输入张量进行归一化处理，并应用可学习的权重参数。
    """
    def __init__(self, dim, eps=1e-6):
        """
        初始化 T5LayerNorm。

        参数:
            dim (int): 输入张量的维度。
            eps (float, 可选): 用于数值稳定性的小常数，默认为 1e-6。
        """
        super(T5LayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的权重参数，初始化为全1

    def forward(self, x):
        """
        前向传播过程，应用层归一化。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 应用了层归一化后的张量。
        """
        # 计算均方根
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) +
                            self.eps)
        # 如果权重是半精度类型，则将 x 转换为与权重相同的类型
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            x = x.type_as(self.weight)
        # 应用权重参数
        return self.weight * x


class T5Attention(nn.Module):
    """
    T5 风格的注意力机制实现。

    该类实现了多头自注意力机制，支持上下文和位置偏置。
    """
    def __init__(self, dim, dim_attn, num_heads, dropout=0.1):
        """
        初始化 T5Attention。

        参数:
            dim (int): 输入和输出的维度。
            dim_attn (int): 注意力机制的维度。
            num_heads (int): 注意力头的数量。
            dropout (float, 可选): Dropout 概率，默认为0.1。
        """
        assert dim_attn % num_heads == 0
        super(T5Attention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads

        # 定义线性层，将输入转换为查询 (q)、键 (k)、值 (v) 和输出 (o)
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, pos_bias=None):
        """
        前向传播过程，应用注意力机制。

        参数:
            x (torch.Tensor): 查询输入张量，形状为 [B, L1, C]。
            context (Optional[torch.Tensor]): 上下文输入张量，形状为 [B, L2, C]。如果为 None，则使用 x 作为上下文。
            mask (Optional[torch.Tensor]): 注意力掩码，形状为 [B, L2] 或 [B, L1, L2]。如果为 None，则不应用掩码。
            pos_bias (Optional[torch.Tensor]): 位置偏置，形状为 [B, N, L1, L2]。如果为 None，则不应用位置偏置。

        返回:
            torch.Tensor: 注意力计算后的输出张量，形状为 [B, L1, C]。
        """
        # 参数处理
        # 如果未提供上下文，则使用查询作为上下文
        context = x if context is None else context
        # 获取批量大小、头的数量和每个头的维度
        b, n, c = x.size(0), self.num_heads, self.head_dim

        # 计算查询 (q)、键 (k) 和值 (v)
        q = self.q(x).view(b, -1, n, c)  # 计算查询并重塑形状为 [B, L1, N, C]
        k = self.k(context).view(b, -1, n, c)  # 计算键并重塑形状为 [B, L2, N, C]
        v = self.v(context).view(b, -1, n, c)  # 计算值并重塑形状为 [B, L2, N, C]

        # 计算注意力偏置
        attn_bias = x.new_zeros(b, n, q.size(1), k.size(1))  # 初始化注意力偏置为零张量
        if pos_bias is not None:
            # 如果提供了位置偏置，则将其添加到注意力偏置中
            attn_bias += pos_bias
        if mask is not None:
            # 确保掩码的维度为2或3
            assert mask.ndim in [2, 3]
            # 重塑掩码形状以适应注意力计算
            mask = mask.view(b, 1, 1,
                             -1) if mask.ndim == 2 else mask.unsqueeze(1)
            # 对掩码为0的位置进行填充，设置为负无穷大
            attn_bias.masked_fill_(mask == 0, torch.finfo(x.dtype).min)

        # 计算注意力权重
        attn = torch.einsum('binc,bjnc->bnij', q, k) + attn_bias  # 计算 QK^T + 偏置
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)  # 应用 softmax 激活函数
        x = torch.einsum('bnij,bjnc->binc', attn, v)  # 计算最终的注意力输出

        # output
        x = x.reshape(b, -1, n * c)  # 重塑张量形状为 [B, L1, N*C]
        x = self.o(x)  # 应用输出线性层
        x = self.dropout(x)  # 应用 Dropout
        # 返回输出张量
        return x


class T5FeedForward(nn.Module):
    """
    T5 模型中的前馈神经网络（Feed-Forward Network，FFN）实现。

    T5 的 FFN 通常由两个线性层和一个激活函数组成，用于对注意力层的输出进行非线性变换。
    """

    def __init__(self, dim, dim_ffn, dropout=0.1):
        """
        初始化 T5FeedForward 模块。

        参数:
            dim (int): 输入和输出的维度。
            dim_ffn (int): 中间层的维度，通常设置为 dim 的倍数。
            dropout (float, 可选): Dropout 概率，默认为0.1。
        """
        super(T5FeedForward, self).__init__()
        self.dim = dim
        self.dim_ffn = dim_ffn

        # 定义前馈层的层结构
        # 定义门控部分
        self.gate = nn.Sequential(nn.Linear(dim, dim_ffn, bias=False), GELU())
        # 第一个线性层，将输入映射到中间维度
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        # 第二个线性层，将中间维度映射回原始维度
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)
        # Dropout 层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播过程，应用前馈神经网络。

        参数:
            x (torch.Tensor): 输入张量，形状为 [B, L, dim]。

        返回:
            torch.Tensor: 应用了前馈神经网络后的张量，形状为 [B, L, dim]。
        """
        # 第一个线性层输出与门控输出相乘
        x = self.fc1(x) * self.gate(x)
        # 应用 Dropout
        x = self.dropout(x)
        # 第二个线性层输出
        x = self.fc2(x)
        # 应用 Dropout
        x = self.dropout(x)
        return x


class T5SelfAttention(nn.Module):
    """
    T5 模型中的自注意力机制实现。

    该模块实现了多头自注意力机制，并结合了位置偏置和前馈神经网络（FFN）。
    """
    def __init__(self,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        """
        初始化 T5SelfAttention 模块。

        参数:
            dim (int): 输入和输出的维度。
            dim_attn (int): 注意力机制的维度。
            dim_ffn (int): 前馈神经网络中中间层的维度。
            num_heads (int): 注意力头的数量。
            num_buckets (int): 位置偏置桶的数量。
            shared_pos (bool, 可选): 是否共享位置嵌入，默认为 True。
            dropout (float, 可选): Dropout 概率，默认为0.1。
        """
        super(T5SelfAttention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # 层归一化
        self.norm1 = T5LayerNorm(dim)
        # 自注意力机制
        self.attn = T5Attention(dim, dim_attn, num_heads, dropout)
        # 层归一化
        self.norm2 = T5LayerNorm(dim)
        # 前馈神经网络
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = None if shared_pos else T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=True)  # 位置嵌入，如果未共享，则初始化

    def forward(self, x, mask=None, pos_bias=None):
        """
        前向传播过程，应用自注意力机制。

        参数:
            x (torch.Tensor): 输入张量，形状为 [B, L, C]。
            mask (Optional[torch.Tensor]): 注意力掩码，形状为 [B, L] 或 [B, L, L]，默认为 None。
            pos_bias (Optional[torch.Tensor]): 位置偏置，形状为 [B, N, L, L]，默认为 None。

        返回:
            torch.Tensor: 应用了自注意力机制后的张量，形状为 [B, L, C]。
        """
        # 如果共享位置嵌入，则使用传入的偏置；否则，计算位置嵌入
        e = pos_bias if self.shared_pos else self.pos_embedding(
            x.size(1), x.size(1))
        # 应用归一化、自注意力和残差连接，并进行裁剪
        x = fp16_clamp(x + self.attn(self.norm1(x), mask=mask, pos_bias=e))
        # 应用归一化、前馈神经网络和残差连接，并进行裁剪
        x = fp16_clamp(x + self.ffn(self.norm2(x)))
        return x


class T5CrossAttention(nn.Module):
    """
    T5 模型中的交叉注意力机制实现。

    该模块实现了多头交叉注意力机制，并结合了自注意力机制和前馈神经网络（FFN）。
    """
    def __init__(self,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        """
        初始化 T5CrossAttention 模块。

        参数:
            dim (int): 输入和输出的维度。
            dim_attn (int): 注意力机制的维度。
            dim_ffn (int): 前馈神经网络中中间层的维度。
            num_heads (int): 注意力头的数量。
            num_buckets (int): 位置偏置桶的数量。
            shared_pos (bool, 可选): 是否共享位置嵌入，默认为 True。
            dropout (float, 可选): Dropout 概率，默认为0.1。
        """
        super(T5CrossAttention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # 层归一化
        self.norm1 = T5LayerNorm(dim)
        # 自注意力机制
        self.self_attn = T5Attention(dim, dim_attn, num_heads, dropout)
        # 层归一化
        self.norm2 = T5LayerNorm(dim)
        # 交叉注意力机制
        self.cross_attn = T5Attention(dim, dim_attn, num_heads, dropout)
        # 层归一化
        self.norm3 = T5LayerNorm(dim)
        # 前馈神经网络
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = None if shared_pos else T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=False)  # 位置嵌入，如果未共享，则初始化

    def forward(self,
                x,
                mask=None,
                encoder_states=None,
                encoder_mask=None,
                pos_bias=None):
        """
        前向传播过程，应用交叉注意力机制。

        参数:
            x (torch.Tensor): 输入张量，形状为 [B, L, C]。
            mask (Optional[torch.Tensor]): 自注意力掩码，形状为 [B, L] 或 [B, L, L]，默认为 None。
            encoder_states (Optional[torch.Tensor]): 编码器状态，形状为 [B, L', C]，默认为 None。
            encoder_mask (Optional[torch.Tensor]): 交叉注意力掩码，形状为 [B, L']，默认为 None。
            pos_bias (Optional[torch.Tensor]): 位置偏置，形状为 [B, N, L, L]，默认为 None。

        返回:
            torch.Tensor: 应用了交叉注意力机制后的张量，形状为 [B, L, C]。
        """
        # 如果共享位置嵌入，则使用传入的偏置；否则，计算位置嵌入
        e = pos_bias if self.shared_pos else self.pos_embedding(
            x.size(1), x.size(1))
        # 应用归一化、自注意力和残差连接，并进行裁剪
        x = fp16_clamp(x + self.self_attn(self.norm1(x), mask=mask, pos_bias=e))
        # 应用归一化、交叉注意力和残差连接，并进行裁剪
        x = fp16_clamp(x + self.cross_attn(
            self.norm2(x), context=encoder_states, mask=encoder_mask))
        # 应用归一化、前馈神经网络和残差连接，并进行裁剪
        x = fp16_clamp(x + self.ffn(self.norm3(x)))
        return x


class T5RelativeEmbedding(nn.Module):
    """
    T5 模型中的相对位置编码实现。

    该类实现了相对位置编码，用于在注意力机制中引入位置信息。
    """
    def __init__(self, num_buckets, num_heads, bidirectional, max_dist=128):
        """
        初始化 T5RelativeEmbedding。

        参数:
            num_buckets (int): 位置桶的数量，用于离散化相对位置。
            num_heads (int): 注意力头的数量。
            bidirectional (bool): 是否使用双向相对位置编码。如果为 True，则考虑双向位置关系；否则，只考虑单向。
            max_dist (int, 可选): 最大相对距离，默认为128。
        """
        super(T5RelativeEmbedding, self).__init__()
        # 位置桶的数量
        self.num_buckets = num_buckets
        # 注意力头的数量
        self.num_heads = num_heads
        # 是否双向
        self.bidirectional = bidirectional
        # 最大相对距离
        self.max_dist = max_dist

        # 初始化嵌入层，嵌入维度为 num_heads
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def forward(self, lq, lk):
        """
        前向传播过程，生成相对位置编码。

        参数:
            lq (int): 查询序列长度。
            lk (int): 键序列长度。

        返回:
            torch.Tensor: 相对位置编码张量，形状为 [1, N, Lq, Lk]。
        """
        device = self.embedding.weight.device
        # 计算相对位置矩阵
        rel_pos = torch.arange(lk, device=device).unsqueeze(0) - \
            torch.arange(lq, device=device).unsqueeze(1)
        # 将相对位置映射到相对位置桶
        rel_pos = self._relative_position_bucket(rel_pos)
        # 获取相对位置嵌入
        rel_pos_embeds = self.embedding(rel_pos)
        # 将嵌入张量重塑为 [1, N, Lq, Lk]
        rel_pos_embeds = rel_pos_embeds.permute(2, 0, 1).unsqueeze(
            0)  # [1, N, Lq, Lk]
        # 返回连续的嵌入张量
        return rel_pos_embeds.contiguous()

    def _relative_position_bucket(self, rel_pos):
        """
        将相对位置映射到相对位置桶。

        参数:
            rel_pos (torch.Tensor): 相对位置张量，形状为 [Lq, Lk]。

        返回:
            torch.Tensor: 映射后的相对位置桶，形状为 [Lq, Lk]。
        """
        if self.bidirectional:
            # 如果是双向，则桶数量减半
            num_buckets = self.num_buckets // 2
            # 计算正负位置
            rel_buckets = (rel_pos > 0).long() * num_buckets
            # 取绝对值
            rel_pos = torch.abs(rel_pos)
        else:
            # 单向，桶数量不变
            num_buckets = self.num_buckets
            # 初始化桶索引为0
            rel_buckets = 0
            # 取负值并与0比较，取较大值
            rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))

        # embeddings for small and large positions
        # 计算精确桶的数量
        max_exact = num_buckets // 2
        # 对相对位置进行对数映射
        rel_pos_large = max_exact + (torch.log(rel_pos.float() / max_exact) /
                                     math.log(self.max_dist / max_exact) *
                                     (num_buckets - max_exact)).long()
        # 确保桶索引不超过最大桶索引
        rel_pos_large = torch.min(
            rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1))
        # 根据相对位置的大小选择桶索引
        rel_buckets += torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)
        # 返回映射后的桶索引
        return rel_buckets


class T5Encoder(nn.Module):
    """
    T5 模型中的编码器实现。

    该类实现了 T5 编码器，包括嵌入层、位置嵌入、注意力块以及归一化层。
    """
    def __init__(self,
                 vocab,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_layers,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        """
        初始化 T5 编码器。

        参数:
            vocab (int): 词汇表大小。
            dim (int): 模型的维度。
            dim_attn (int): 注意力机制的维度。
            dim_ffn (int): 前馈神经网络中中间层的维度。
            num_heads (int): 注意力头的数量。
            num_layers (int): 变换器层的数量。
            num_buckets (int): 位置桶的数量。
            shared_pos (bool, 可选): 是否共享位置嵌入，默认为 True。
            dropout (float, 可选): Dropout 概率，默认为0.1。
        """
        super(T5Encoder, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # 词汇表嵌入
        self.token_embedding = vocab if isinstance(vocab, nn.Embedding) \
            else nn.Embedding(vocab, dim)
        # 位置嵌入
        self.pos_embedding = T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=True) if shared_pos else None
        # Dropout 层
        self.dropout = nn.Dropout(dropout)
        # 构建编码器层列表
        self.blocks = nn.ModuleList([
            T5SelfAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets,
                            shared_pos, dropout) for _ in range(num_layers)
        ])
        # 层归一化
        self.norm = T5LayerNorm(dim)

        # 应用权重初始化
        self.apply(init_weights)

    def forward(self, ids, mask=None):
        """
        前向传播过程。

        参数:
            ids (torch.Tensor): 输入的 token ID 张量，形状为 [B, L]。
            mask (Optional[torch.Tensor]): 注意力掩码，形状为 [B, L]，默认为 None。

        返回:
            torch.Tensor: 编码器输出张量，形状为 [B, L, dim]。
        """
        # 词汇表嵌入
        x = self.token_embedding(ids)
        # 应用 Dropout
        x = self.dropout(x)
        # 获取位置嵌入
        e = self.pos_embedding(x.size(1),
                               x.size(1)) if self.shared_pos else None
        for block in self.blocks:
            # 应用每个编码器层
            x = block(x, mask, pos_bias=e)
        # 应用层归一化
        x = self.norm(x)
        # 应用 Dropout
        x = self.dropout(x)
        return x


class T5Decoder(nn.Module):
    """
    T5 模型中的解码器实现。

    该类实现了 T5 解码器，包括嵌入层、位置嵌入、自注意力和交叉注意力块以及归一化层。
    """
    def __init__(self,
                 vocab,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_layers,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        """
        初始化 T5 解码器。

        参数:
            vocab (int): 词汇表大小。
            dim (int): 模型的维度。
            dim_attn (int): 注意力机制的维度。
            dim_ffn (int): 前馈神经网络中中间层的维度。
            num_heads (int): 注意力头的数量。
            num_layers (int): 变换器层的数量。
            num_buckets (int): 位置桶的数量。
            shared_pos (bool, 可选): 是否共享位置嵌入，默认为 True。
            dropout (float, 可选): Dropout 概率，默认为0.1。
        """
        super(T5Decoder, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # 词汇表嵌入
        self.token_embedding = vocab if isinstance(vocab, nn.Embedding) \
            else nn.Embedding(vocab, dim)
        # 位置嵌入
        self.pos_embedding = T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=False) if shared_pos else None
        # Dropout 层
        self.dropout = nn.Dropout(dropout)
        # 构建解码器层列表
        self.blocks = nn.ModuleList([
            T5CrossAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets,
                             shared_pos, dropout) for _ in range(num_layers)
        ])
        # 层归一化
        self.norm = T5LayerNorm(dim)

        # 应用权重初始化
        self.apply(init_weights)

    def forward(self, ids, mask=None, encoder_states=None, encoder_mask=None):
        """
        前向传播过程。

        参数:
            ids (torch.Tensor): 输入的 token ID 张量，形状为 [B, L]。
            mask (Optional[torch.Tensor]): 自注意力掩码，形状为 [B, L] 或 [B, L, L]，默认为 None。
            encoder_states (Optional[torch.Tensor]): 编码器状态，形状为 [B, L', C]，默认为 None。
            encoder_mask (Optional[torch.Tensor]): 交叉注意力掩码，形状为 [B, L']，默认为 None。

        返回:
            torch.Tensor: 解码器输出张量，形状为 [B, L, dim]。
        """
        # 获取批量大小和序列长度
        b, s = ids.size()

        # 如果未提供掩码，则生成因果掩码
        if mask is None:
            mask = torch.tril(torch.ones(1, s, s).to(ids.device))
        elif mask.ndim == 2:
            # 如果掩码是二维的，则扩展为三维因果掩码
            mask = torch.tril(mask.unsqueeze(1).expand(-1, s, -1))

        # 嵌入层
        # 词汇表嵌入
        x = self.token_embedding(ids)
        # 应用 Dropout
        x = self.dropout(x)
        # 获取位置嵌入
        e = self.pos_embedding(x.size(1),
                               x.size(1)) if self.shared_pos else None
        for block in self.blocks:
            # 应用每个解码器层
            x = block(x, mask, encoder_states, encoder_mask, pos_bias=e)
        # 应用层归一化
        x = self.norm(x)
        # 应用 Dropout
        x = self.dropout(x)
        # 返回解码器输出
        return x


class T5Model(nn.Module):
    """
    T5 模型实现。

    该类实现了 T5 模型的整体架构，包括编码器、解码器、嵌入层、位置编码以及输出头。
    """
    def __init__(self,
                 vocab_size,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 encoder_layers,
                 decoder_layers,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        """
        初始化 T5 模型。

        参数:
            vocab_size (int): 词汇表大小。
            dim (int): 模型的维度。
            dim_attn (int): 注意力机制的维度。
            dim_ffn (int): 前馈神经网络中中间层的维度。
            num_heads (int): 注意力头的数量。
            encoder_layers (int): 编码器层的数量。
            decoder_layers (int): 解码器层的数量。
            num_buckets (int): 位置桶的数量。
            shared_pos (bool, 可选): 是否共享位置嵌入，默认为 True。
            dropout (float, 可选): Dropout 概率，默认为0.1。
        """
        super(T5Model, self).__init__()
        # 词汇表大小
        self.vocab_size = vocab_size
        # 模型维度
        self.dim = dim
        # 注意力机制维度
        self.dim_attn = dim_attn
        # 前馈神经网络中间层维度
        self.dim_ffn = dim_ffn
        # 注意力头数量
        self.num_heads = num_heads
        # 编码器层数量
        self.encoder_layers = encoder_layers
        # 解码器层数量
        self.decoder_layers = decoder_layers
        # 位置桶数量
        self.num_buckets = num_buckets

        # 词汇表嵌入
        self.token_embedding = nn.Embedding(vocab_size, dim)

        # 定义编码器和解码器
        self.encoder = T5Encoder(self.token_embedding, dim, dim_attn, dim_ffn,
                                 num_heads, encoder_layers, num_buckets,
                                 shared_pos, dropout)
        self.decoder = T5Decoder(self.token_embedding, dim, dim_attn, dim_ffn,
                                 num_heads, decoder_layers, num_buckets,
                                 shared_pos, dropout)
        # 输出线性层，将维度映射回词汇表大小
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # 应用权重初始化
        self.apply(init_weights)

    def forward(self, encoder_ids, encoder_mask, decoder_ids, decoder_mask):
        """
        前向传播过程。

        参数:
            encoder_ids (torch.Tensor): 编码器的输入 token ID 张量，形状为 [B, L1]。
            encoder_mask (torch.Tensor): 编码器的注意力掩码，形状为 [B, L1]。
            decoder_ids (torch.Tensor): 解码器的输入 token ID 张量，形状为 [B, L2]。
            decoder_mask (torch.Tensor): 解码器的注意力掩码，形状为 [B, L2]。

        返回:
            torch.Tensor: 模型输出张量，形状为 [B, L2, vocab_size]。
        """
        # 计算编码器输出
        x = self.encoder(encoder_ids, encoder_mask)
        # 计算解码器输出，使用编码器输出作为上下文
        x = self.decoder(decoder_ids, decoder_mask, x, encoder_mask)
        # 应用输出线性层
        x = self.head(x)
        return x


def _t5(name,
        encoder_only=False,
        decoder_only=False,
        return_tokenizer=False,
        tokenizer_kwargs={},
        dtype=torch.float32,
        device='cpu',
        **kwargs):
    """
    初始化 T5 模型，根据参数选择是仅使用编码器、解码器还是完整的模型。

    参数:
        name (str): 模型名称。
        encoder_only (bool, 可选): 是否仅使用编码器，默认为 False。
        decoder_only (bool, 可选): 是否仅使用解码器，默认为 False。
        return_tokenizer (bool, 可选): 是否返回分词器，默认为 False。
        tokenizer_kwargs (dict, 可选): 分词器的关键字参数。
        dtype (torch.dtype, 可选): 模型的数据类型，默认为 torch.float32。
        device (str, 可选): 设备类型，默认为 'cpu'。
        **kwargs: 其他关键字参数。

    返回:
        Any: 根据参数返回相应的模型和/或分词器。
    """
    assert not (encoder_only and decoder_only)

    # 参数处理
    if encoder_only:
        # 如果仅使用编码器，则模型类为 T5Encoder
        model_cls = T5Encoder
        # 将 'vocab_size' 关键字参数重命名为 'vocab'
        kwargs['vocab'] = kwargs.pop('vocab_size')
        # 将 'encoder_layers' 关键字参数重命名为 'num_layers'
        kwargs['num_layers'] = kwargs.pop('encoder_layers')
        # 移除 'decoder_layers' 关键字参数
        _ = kwargs.pop('decoder_layers')
    elif decoder_only:
        # 如果仅使用解码器，则模型类为 T5Decoder
        model_cls = T5Decoder
        # 将 'vocab_size' 关键字参数重命名为 'vocab'
        kwargs['vocab'] = kwargs.pop('vocab_size')
        # 将 'decoder_layers' 关键字参数重命名为 'num_layers'
        kwargs['num_layers'] = kwargs.pop('decoder_layers')
        # 移除 'encoder_layers' 关键字参数
        _ = kwargs.pop('encoder_layers')
    else:
        # 否则，模型类为 T5Model
        model_cls = T5Model

    # 模型初始化
    with torch.device(device):
        # 使用提供的参数实例化模型类
        model = model_cls(**kwargs)

    # 将模型移动到指定设备和设置数据类型
    model = model.to(dtype=dtype, device=device)

    # 导入分词器类
    if return_tokenizer:
        from .tokenizers import HuggingfaceTokenizer
        # 初始化分词器
        tokenizer = HuggingfaceTokenizer(f'google/{name}', **tokenizer_kwargs)
        # 返回模型和分词器
        return model, tokenizer
    else:
        # 仅返回模型
        return model


def umt5_xxl(**kwargs):
    """
    初始化 UMT5-XXL 模型。

    参数:
        **kwargs: 其他关键字参数，用于配置模型。

    返回:
        XLMRobertaCLIP: 配置好的 UMT5-XXL 模型实例。
    """
    cfg = dict(
        vocab_size=256384,  # 词汇表大小
        dim=4096,  # 模型维度
        dim_attn=4096,  # 注意力机制维度
        dim_ffn=10240,  # 前馈神经网络中间层维度
        num_heads=64,  # 注意力头数量
        encoder_layers=24,  # 编码器层数量
        decoder_layers=24,  # 解码器层数量
        num_buckets=32,  # 位置桶数量
        shared_pos=False,  # 是否共享位置嵌入
        dropout=0.1) # Dropout 概率
    cfg.update(**kwargs)  # 更新配置参数
    # 调用 _t5 函数进行初始化
    return _t5('umt5-xxl', **cfg)


class T5EncoderModel:
    """
    T5 编码器模型封装类。

    该类封装了 T5 编码器模型，提供了加载模型、加载检查点以及文本编码的功能。
    """
    def __init__(
        self,
        text_len,
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
        checkpoint_path=None,
        tokenizer_path=None,
        shard_fn=None,
    ):
        """
        初始化 T5 编码器模型。

        参数:
            text_len (int): 文本长度。
            dtype (torch.dtype, 可选): 模型的数据类型，默认为 torch.bfloat16。
            device (torch.device, 可选): 设备类型，默认为当前 CUDA 设备。
            checkpoint_path (Optional[str], 可选): 模型检查点路径。
            tokenizer_path (Optional[str], 可选): 分词器路径。
            shard_fn (Optional[Callable], 可选): 分片函数。
        """
        # 文本长度
        self.text_len = text_len
        # 数据类型
        self.dtype = dtype
        self.device = device
        # 检查点路径
        self.checkpoint_path = checkpoint_path
        # 分词器路径
        self.tokenizer_path = tokenizer_path

        # 模型初始化
        model = umt5_xxl(
            encoder_only=True,  # 仅使用编码器
            return_tokenizer=False,  # 不返回分词器
            dtype=dtype,  # 设置数据类型
            device=device).eval().requires_grad_(False)  # 设置为评估模式，不计算梯度
        # 输出日志信息，指示正在加载检查点
        logging.info(f'loading {checkpoint_path}')
        # 加载模型权重
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        # 赋值给模型属性
        self.model = model
        if shard_fn is not None:
            # 如果提供了分片函数，则应用分片
            self.model = shard_fn(self.model, sync_module_states=False)
        else:
            # 否则，将模型移动到指定设备
            self.model.to(self.device)
        # 分词器初始化
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=text_len, clean='whitespace')

    def __call__(self, texts, device):
        """
        对输入文本进行编码。

        参数:
            texts (List[str]): 输入的文本列表。
            device (torch.device): 设备类型。

        返回:
            List[torch.Tensor]: 编码后的文本嵌入列表。
        """
        # 文本编码
        ids, mask = self.tokenizer(
            # 对文本进行分词，并获取 token ID 和掩码
            texts, return_mask=True, add_special_tokens=True)
        # 将 token ID 移动到指定设备
        ids = ids.to(device)
        mask = mask.to(device)
        # 计算每个序列的实际长度
        seq_lens = mask.gt(0).sum(dim=1).long()
        # 对输入序列进行编码
        context = self.model(ids, mask)
        # 返回每个序列的实际长度对应的编码结果
        return [u[:v] for u, v in zip(context, seq_lens)]

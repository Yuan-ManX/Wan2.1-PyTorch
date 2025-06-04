# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    Flash Attention 实现的高效注意力计算。

    参数:
        q (torch.Tensor): 查询张量，形状为 [B, Lq, Nq, C1]。
        k (torch.Tensor): 键张量，形状为 [B, Lk, Nk, C1]。
        v (torch.Tensor): 值张量，形状为 [B, Lk, Nk, C2]。Nq 必须能被 Nk 整除。
        q_lens (Optional[torch.Tensor]): 每个查询序列的实际长度，形状为 [B]。
        k_lens (Optional[torch.Tensor]): 每个键序列的实际长度，形状为 [B]。
        dropout_p (float, 可选): Dropout 概率，默认为0。
        softmax_scale (Optional[float], 可选): 在应用 softmax 之前对 QK^T 进行缩放的因子，默认为 None。
        q_scale (Optional[float], 可选): 对查询进行缩放的因子，默认为 None。
        causal (bool, 可选): 是否应用因果注意力掩码，默认为 False。
        window_size (Tuple[int, int], 可选): 滑动窗口局部注意力的窗口大小，默认为 (-1, -1)。如果为 (-1, -1)，则不应用滑动窗口。
        deterministic (bool, 可选): 如果为 True，则结果更确定，但速度稍慢且使用更多内存，默认为 False。
        dtype (torch.dtype, 可选): 当 q/k/v 的 dtype 不是 float16/bfloat16 时应用的类型，默认为 torch.bfloat16。
        version (Optional[int], 可选): 指定使用的 Flash Attention 版本。如果为 None，则自动选择，默认为 None。

    返回:
        torch.Tensor: 注意力计算结果。
    """
    # 定义半精度数据类型
    half_dtypes = (torch.float16, torch.bfloat16)
    # 确保 dtype 是半精度类型
    assert dtype in half_dtypes
    # 确保查询张量在 CUDA 上，并且通道数不超过256
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # 获取张量的基本维度
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    # 定义一个辅助函数，用于将张量转换为半精度类型（如果尚未是）
    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # 预处理查询张量
    if q_lens is None:
        # 如果未提供查询长度，则假设所有查询序列长度相同
        q = half(q.flatten(0, 1))  # 将前两个维度展平
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)  # 生成每个查询序列的长度列表
    else:
        # 如果提供了查询长度，则根据长度截断每个查询序列
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))  # 截断并拼接查询序列

    # 预处理键和值张量
    if k_lens is None:
        # 如果未提供键长度，则假设所有键序列长度相同
        k = half(k.flatten(0, 1))  # 将前两个维度展平
        v = half(v.flatten(0, 1))  # 将前两个维度展平
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)  # 生成每个键序列的长度列表
    else:
        # 如果提供了键长度，则根据长度截断每个键序列
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))   # 截断并拼接键序列
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))   # 截断并拼接值序列

    # 将查询和键的 dtype 转换为与值相同的类型
    q = q.to(v.dtype)
    k = k.to(v.dtype)

    # 如果提供了查询缩放因子，则对查询进行缩放
    if q_scale is not None:
        q = q * q_scale

    # 版本选择与警告
    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # 应用注意力机制
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # 如果未指定版本或指定为3，并且支持 Flash Attention v3，则使用 v3
        # 注意：当前 Flash Attention v3 不支持 dropout_p 和 window_size 参数
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),  # 计算累积序列长度
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),  # 计算累积序列长度
            seqused_q=None,  # 序列使用的查询长度（未使用）
            seqused_k=None,  # 序列使用的键长度（未使用）
            max_seqlen_q=lq,  # 最大查询序列长度
            max_seqlen_k=lk,  # 最大键序列长度
            softmax_scale=softmax_scale,  # softmax 缩放因子
            causal=causal,  # 是否应用因果掩码
            deterministic=deterministic)[0].unflatten(0, (b, lq))  # 是否使用确定性计算
    else:
        # 否则，使用 Flash Attention v2
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),  # 计算累积序列长度
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),  # 计算累积序列长度
            max_seqlen_q=lq,  # 最大查询序列长度
            max_seqlen_k=lk,  # 最大键序列长度
            dropout_p=dropout_p,  # Dropout 概率
            softmax_scale=softmax_scale,   # softmax 缩放因子
            causal=causal,   # 是否应用因果掩码
            window_size=window_size,  # 滑动窗口局部注意力窗口大小
            deterministic=deterministic).unflatten(0, (b, lq))   # 是否使用确定性计算

    # 将结果转换为指定的输出类型
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    """
    通用的注意力机制实现，优先使用 Flash Attention 算法（如果可用），否则退回到使用 PyTorch 内置的 scaled_dot_product_attention 方法。

    参数:
        q (torch.Tensor): 查询张量，形状为 [B, Lq, Nq, C1]。
        k (torch.Tensor): 键张量，形状为 [B, Lk, Nk, C1]。
        v (torch.Tensor): 值张量，形状为 [B, Lk, Nk, C2]。Nq 必须能被 Nk 整除。
        q_lens (Optional[torch.Tensor]): 每个查询序列的实际长度，形状为 [B]，默认为 None。
        k_lens (Optional[torch.Tensor]): 每个键序列的实际长度，形状为 [B]，默认为 None。
        dropout_p (float, 可选): Dropout 概率，默认为0。
        softmax_scale (Optional[float], 可选): 在应用 softmax 之前对 QK^T 进行缩放的因子，默认为 None。
        q_scale (Optional[float], 可选): 对查询进行缩放的因子，默认为 None。
        causal (bool, 可选): 是否应用因果注意力掩码，默认为 False。
        window_size (Tuple[int, int], 可选): 滑动窗口局部注意力的窗口大小，默认为 (-1, -1)。如果为 (-1, -1)，则不应用滑动窗口。
        deterministic (bool, 可选): 如果为 True，则使用确定性计算，默认为 False。
        dtype (torch.dtype, 可选): 当 q/k/v 的 dtype 不是 float16/bfloat16 时应用的类型，默认为 torch.bfloat16。
        fa_version (Optional[int], 可选): 指定使用的 Flash Attention 版本，默认为 None。

    返回:
        torch.Tensor: 注意力计算结果。
    """
    # 如果 Flash Attention v2 或 v3 可用，则使用 Flash Attention
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        # 如果不支持 Flash Attention，则使用 PyTorch 内置的 scaled_dot_product_attention 方法
        if q_lens is not None or k_lens is not None:
            # 如果提供了查询或键长度，则发出警告，因为 scaled_dot_product_attention 不支持填充掩码
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        # 将查询、键和值张量从 (B, L, N, C) 转换为 (B, N, L, C)，以适应 scaled_dot_product_attention 的输入格式
        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        # 使用 PyTorch 内置的 scaled_dot_product_attention 进行注意力计算
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        # 将输出张量从 (B, N, L, C) 转回 (B, L, N, C)
        out = out.transpose(1, 2).contiguous()
        return out

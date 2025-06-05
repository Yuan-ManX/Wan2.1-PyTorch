# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = [
    'WanVAE',
]

CACHE_T = 2


class CausalConv3d(nn.Conv3d):
    """
    因果3D卷积（Causal 3D Convolution）实现。

    因果卷积确保在时间维度上，输出仅依赖于当前和过去的输入，而不依赖于未来的输入。这在处理序列数据（如视频帧）时非常重要。
    """

    def __init__(self, *args, **kwargs):
        """
        初始化 CausalConv3d。

        参数:
            *args: 传递给 nn.Conv3d 的位置参数。
            **kwargs: 传递给 nn.Conv3d 的关键字参数。
        """
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        # 重置填充参数，避免在 forward 中重复填充
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        """
        前向传播过程，应用因果3D卷积。

        参数:
            x (torch.Tensor): 输入张量，形状为 [B, C, T, H, W]。
            cache_x (Optional[torch.Tensor], 可选): 缓存张量，用于存储前一个时间步的输入，默认为 None。

        返回:
            torch.Tensor: 应用了因果3D卷积后的张量，形状为 [B, C_out, T, H, W]。
        """
        # 将填充参数转换为列表，以便后续修改
        padding = list(self._padding)
        # **处理缓存**: 如果提供了缓存张量，并且时间填充大于0，则将缓存张量与输入张量拼接
        if cache_x is not None and self._padding[4] > 0:
            # 将缓存张量移动到与输入相同的设备
            cache_x = cache_x.to(x.device)
            # 在时间维度上拼接缓存和输入
            x = torch.cat([cache_x, x], dim=2)
            # 调整时间填充，以考虑缓存张量的长度
            padding[4] -= cache_x.shape[2]
        
        # 对输入张量进行填充，以实现因果卷积
        x = F.pad(x, padding)

        return super().forward(x)


class RMS_norm(nn.Module):
    """
    RMS 层归一化实现。

    RMS 层归一化对输入张量进行归一化处理，并应用可学习的缩放和偏置参数。
    """
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        """
        初始化 RMS_norm。

        参数:
            dim (int): 输入张量的维度。
            channel_first (bool, 可选): 是否为通道优先格式，默认为 True。
            images (bool, 可选): 是否为图像数据，默认为 True。
            bias (bool, 可选): 是否使用偏置，默认为 False。
        """
        super().__init__()
        # 如果不是图像数据，则广播维度为 (1, 1, 1)
        # 如果是图像数据，则广播维度为 (1, 1)
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)

        # 形状为 (dim, 1, 1) 或 (dim, 1, 1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        # 设置通道优先标志
        self.channel_first = channel_first
        # 计算缩放因子
        self.scale = dim**0.5
        # 可学习的缩放参数，初始化为全1
        self.gamma = nn.Parameter(torch.ones(shape))
        # 可学习的偏置参数，初始化为全0；如果不使用偏置，则为0
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        """
        前向传播过程，应用 RMS 层归一化。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 应用了 RMS 层归一化后的张量。
        """
        # 应用归一化、缩放和偏置
        return F.normalize(
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):
    """
    上采样实现，修复了 bfloat16 类型的最近邻插值支持问题。
    """
    def forward(self, x):
        """
        前向传播过程，应用上采样。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 应用了上采样后的张量，类型与输入相同。
        """
        # 将输入转换为 float 进行上采样，然后转换回原始类型
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):
    """
    重采样模块，支持2D和3D的上采样和下采样。

    该模块通过一系列卷积和插值操作实现重采样。
    """
    def __init__(self, dim, mode):
        """
        初始化 Resample。

        参数:
            dim (int): 输入和输出的维度。
            mode (str): 重采样模式，可选 'none', 'upsample2d', 'upsample3d', 'downsample2d', 'downsample3d'。
        """
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode

        # **定义重采样层**
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),  # 2D 上采样，使用最近邻插值
                nn.Conv2d(dim, dim // 2, 3, padding=1))  # 2D 卷积，减少通道数
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),  # 2D 上采样，使用最近邻插值
                nn.Conv2d(dim, dim // 2, 3, padding=1))  # 2D 卷积，减少通道数
            self.time_conv = CausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))  # 3D 因果卷积，增加通道数

        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),  # 2D 零填充
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))  # 2D 卷积，下采样
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),  # 2D 零填充
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))  # 2D 卷积，下采样
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))  # 3D 因果卷积，下采样

        else:
            self.resample = nn.Identity()  # 无重采样操作

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        """
        前向传播过程，应用重采样。

        参数:
            x (torch.Tensor): 输入张量，形状为 [B, C, T, H, W]。
            feat_cache (Optional[torch.Tensor], 可选): 特征缓存，默认为 None。
            feat_idx (List[int], 可选): 特征索引列表，默认为 [0]。

        返回:
            torch.Tensor: 重采样后的张量。
        """
        # 获取输入张量的维度信息
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                # 获取当前特征索引
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'  # 如果缓存为空，则标记为 'Rep'
                    feat_idx[0] += 1  # 增加特征索引
                else:
                    # **缓存处理**: 如果提供了缓存，则将当前输入的最后几帧与缓存拼接
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] != 'Rep':
                        # 如果缓存长度小于2，则将缓存的最后一帧与当前输入拼接
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device), cache_x
                        ],
                                            dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] == 'Rep':
                        # 如果缓存长度小于2，并且标记为 'Rep'，则用零张量与当前输入拼接
                        cache_x = torch.cat([
                            torch.zeros_like(cache_x).to(cache_x.device),
                            cache_x
                        ],
                                            dim=2)
                    if feat_cache[idx] == 'Rep':
                        # 应用3D因果卷积
                        x = self.time_conv(x)
                    else:
                        # 应用3D因果卷积，并使用缓存
                        x = self.time_conv(x, feat_cache[idx])
                    # 更新缓存
                    feat_cache[idx] = cache_x
                    # 增加特征索引
                    feat_idx[0] += 1

                    # 重塑张量形状
                    x = x.reshape(b, 2, c, t, h, w)
                    # 堆叠张量
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                                    3)
                    # 重塑回原始形状
                    x = x.reshape(b, c, t * 2, h, w)
        # 获取时间维度长度
        t = x.shape[2]
        # 重排张量维度
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        # 应用重采样操作
        x = self.resample(x)
        # 重排回原始维度
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if self.mode == 'downsample3d':
            if feat_cache is not None:
                # 获取当前特征索引
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    # 如果缓存为空，则复制当前输入到缓存
                    feat_cache[idx] = x.clone()
                    # 增加特征索引
                    feat_idx[0] += 1
                else:
                    # **缓存处理**: 如果提供了缓存，则将当前输入与缓存的最后几帧拼接
                    cache_x = x[:, :, -1:, :, :].clone()
                    # if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx]!='Rep':
                    #     # cache last frame of last two chunk
                    #     cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

                    # 应用3D因果卷积
                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    # 更新缓存
                    feat_cache[idx] = cache_x
                    # 增加特征索引
                    feat_idx[0] += 1
        # 返回重采样后的张量
        return x

    def init_weight(self, conv):
        """
        初始化3D卷积权重。

        参数:
            conv (nn.Conv3d): 需要初始化的3D卷积层。
        """
        # 获取卷积权重
        conv_weight = conv.weight
        # 将权重初始化为零
        nn.init.zeros_(conv_weight)
        # 获取权重维度
        c1, c2, t, h, w = conv_weight.size()
        # 生成单位矩阵
        one_matrix = torch.eye(c1, c2)
        # 初始化矩阵
        init_matrix = one_matrix
        # 将权重初始化为零
        nn.init.zeros_(conv_weight)
        #conv_weight.data[:,:,-1,1,1] = init_matrix * 0.5
        # 设置权重矩阵的对角线元素
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  #* 0.5
        # 复制权重到卷积层
        conv.weight.data.copy_(conv_weight)
        # 将偏置初始化为零
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        """
        初始化3D卷积权重（另一种方式）。

        参数:
            conv (nn.Conv3d): 需要初始化的3D卷积层。
        """
        # 获取卷积权重数据
        conv_weight = conv.weight.data
        # 将权重初始化为零
        nn.init.zeros_(conv_weight)
        # 获取权重维度
        c1, c2, t, h, w = conv_weight.size()
        # 生成单位矩阵
        init_matrix = torch.eye(c1 // 2, c2)
        #init_matrix = repeat(init_matrix, 'o ... -> (o 2) ...').permute(1,0,2).contiguous().reshape(c1,c2)
        # 设置权重矩阵的对角线元素
        conv_weight[:c1 // 2, :, -1, 0, 0] = init_matrix
        # 设置权重矩阵的对角线元素
        conv_weight[c1 // 2:, :, -1, 0, 0] = init_matrix
        # 复制权重到卷积层
        conv.weight.data.copy_(conv_weight)
        # 将偏置初始化为零
        nn.init.zeros_(conv.bias.data)


class ResidualBlock(nn.Module):
    """
    残差块（Residual Block）实现。

    该残差块包含两个因果3D卷积层，以及残差连接和跳跃连接（shortcut）。
    """
    def __init__(self, in_dim, out_dim, dropout=0.0):
        """
        初始化 ResidualBlock。

        参数:
            in_dim (int): 输入的维度。
            out_dim (int): 输出的维度。
            dropout (float, 可选): Dropout 概率，默认为0.0。
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # **定义残差路径**
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),  # RMS 层归一化
            CausalConv3d(in_dim, out_dim, 3, padding=1),  # 因果3D卷积，卷积核大小为3，填充为1
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),  # RMS 层归一化
            CausalConv3d(out_dim, out_dim, 3, padding=1))  # 因果3D卷积，卷积核大小为3，填充为1
        # **定义跳跃连接**
        # 如果输入和输出维度不同，则使用1x1卷积作为跳跃连接；否则，使用恒等映射
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        """
        前向传播过程，应用残差块。

        参数:
            x (torch.Tensor): 输入张量，形状为 [B, C, T, H, W]。
            feat_cache (Optional[torch.Tensor], 可选): 特征缓存，默认为 None。
            feat_idx (List[int], 可选): 特征索引列表，默认为 [0]。

        返回:
            torch.Tensor: 应用了残差块后的张量，形状为 [B, out_dim, T, H, W]。
        """
        # 应用跳跃连接
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                # **处理因果卷积层**: 如果当前层是因果卷积层，并且提供了特征缓存，则进行以下处理
                idx = feat_idx[0]  # 获取当前特征索引
                # 复制输入张量的最后几帧作为缓存
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # **缓存处理**: 如果缓存长度小于2，并且缓存不为空，则将缓存的最后一帧与当前输入拼接
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                # 应用因果卷积层，并使用缓存
                x = layer(x, feat_cache[idx])
                # 更新缓存
                feat_cache[idx] = cache_x
                # 增加特征索引
                feat_idx[0] += 1
            else:
                # 否则，直接应用当前层
                x = layer(x)
        # 应用残差连接
        return x + h


class AttentionBlock(nn.Module):
    """
    因果自注意力块实现，包含单个注意力头。

    该块实现了因果自注意力机制，用于捕捉序列数据中的依赖关系。
    """

    def __init__(self, dim):
        """
        初始化 AttentionBlock。

        参数:
            dim (int): 注意力机制的维度。
        """
        super().__init__()
        self.dim = dim

        # **定义层**
        self.norm = RMS_norm(dim)  # RMS 层归一化
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)  # 1x1卷积，将输入映射到查询、键和值
        self.proj = nn.Conv2d(dim, dim, 1)  # 1x1卷积，用于输出投影

        # 将输出投影层的权重初始化为零
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        """
        前向传播过程，应用因果自注意力。

        参数:
            x (torch.Tensor): 输入张量，形状为 [B, C, T, H, W]。

        返回:
            torch.Tensor: 应用了因果自注意力后的张量，形状为 [B, C, T, H, W]。
        """
        # 保存输入张量以应用残差连接
        identity = x
        # 获取输入张量的维度信息
        b, c, t, h, w = x.size()
        # 重排张量维度以适应注意力计算
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        # 应用 RMS 层归一化
        x = self.norm(x)
        # **计算查询 (q)、键 (k) 和值 (v)**
        # 重塑并分割为查询、键和值
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3,
                                         -1).permute(0, 1, 3,
                                                     2).contiguous().chunk(
                                                         3, dim=-1)

        # **应用注意力**
        # 应用缩放点积注意力
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )
        # 重塑张量形状
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # 应用输出投影
        x = self.proj(x)
        # 重排回原始维度
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        # 应用残差连接
        return x + identity


class Encoder3d(nn.Module):
    """
    3D编码器实现，用于处理视频数据。

    该编码器结合了残差块、注意力块以及重采样操作，实现多层次的特征提取。
    """
    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        """
        初始化 Encoder3d。

        参数:
            dim (int, 可选): 模型的维度，默认为128。
            z_dim (int, 可选): 潜在空间的维度，默认为4。
            dim_mult (List[int], 可选): 维度倍增列表，默认为 [1, 2, 4, 4]。
            num_res_blocks (int, 可选): 每个下采样阶段的残差块数量，默认为2。
            attn_scales (List[float], 可选): 应用注意力的尺度列表，默认为 []。
            temperal_downsample (List[bool], 可选): 是否在时间维度上下采样，默认为 [True, True, False]。
            dropout (float, 可选): Dropout 概率，默认为0.0。
        """
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # 计算每个阶段的维度
        dims = [dim * u for u in [1] + dim_mult]
        # 初始化尺度
        scale = 1.0

        # 3D因果卷积，将输入通道数从3转换为 dims[0]
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # **下采样块**
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # **残差块与注意力块**
            for _ in range(num_res_blocks):
                # 添加残差块
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    # 如果当前尺度在 attn_scales 中，则添加注意力块
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # **重采样块**
            if i != len(dim_mult) - 1:
                # 根据是否在时间维度上下采样选择模式
                mode = 'downsample3d' if temperal_downsample[
                    i] else 'downsample2d'
                # 添加重采样块
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        # 构建下采样序列
        self.downsamples = nn.Sequential(*downsamples)

         # **中间块**
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout))

        # **输出块**
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),  
            CausalConv3d(out_dim, z_dim, 3, padding=1))  # 3D因果卷积，将输出通道数转换为 z_dim
  
    def forward(self, x, feat_cache=None, feat_idx=[0]):
        """
        前向传播过程，应用3D编码器。

        参数:
            x (torch.Tensor): 输入张量，形状为 [B, C, T, H, W]。
            feat_cache (Optional[torch.Tensor], 可选): 特征缓存，默认为 None。
            feat_idx (List[int], 可选): 特征索引列表，默认为 [0]。

        返回:
            torch.Tensor: 编码后的张量，形状为 [B, z_dim, T, H, W]。
        """
        if feat_cache is not None:
            # 获取当前特征索引
            idx = feat_idx[0]
            # 复制输入张量的最后几帧作为缓存
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # **缓存处理**: 如果缓存长度小于2，并且缓存不为空，则将缓存的最后一帧与当前输入拼接
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            # 应用3D因果卷积，并使用缓存
            x = self.conv1(x, feat_cache[idx])
            # 更新缓存
            feat_cache[idx] = cache_x
            # 增加特征索引
            feat_idx[0] += 1
        else:
            # 否则，直接应用3D因果卷积
            x = self.conv1(x)

        ## **下采样阶段**
        for layer in self.downsamples:
            if feat_cache is not None:
                # 应用下采样块，并使用缓存
                x = layer(x, feat_cache, feat_idx)
            else:
                # 否则，直接应用下采样块
                x = layer(x)

        ## **中间阶段**
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                # 应用残差块，并使用缓存
                x = layer(x, feat_cache, feat_idx)
            else:
                # 否则，直接应用当前层
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                # 获取当前特征索引
                idx = feat_idx[0]
                # 复制输入张量的最后几帧作为缓存
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # **缓存处理**: 如果缓存长度小于2，并且缓存不为空，则将缓存的最后一帧与当前输入拼接
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                # 应用3D因果卷积，并使用缓存
                x = layer(x, feat_cache[idx])
                # 更新缓存
                feat_cache[idx] = cache_x
                # 增加特征索引
                feat_idx[0] += 1
            else:
                # 否则，直接应用当前层
                x = layer(x)
        # 返回编码后的张量
        return x


class Decoder3d(nn.Module):
    """
    3D解码器实现，用于视频数据的解码和重建。

    该解码器结合了残差块、注意力块以及上采样操作，实现多层次的特征生成。
    """
    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[False, True, True],
                 dropout=0.0):
        """
        初始化 Decoder3d。

        参数:
            dim (int, 可选): 模型的维度，默认为128。
            z_dim (int, 可选): 潜在空间的维度，默认为4。
            dim_mult (List[int], 可选): 维度倍增列表，默认为 [1, 2, 4, 4]。
            num_res_blocks (int, 可选): 每个上采样阶段的残差块数量，默认为2。
            attn_scales (List[float], 可选): 应用注意力的尺度列表，默认为 []。
            temperal_upsample (List[bool], 可选): 是否在时间维度上上采样，默认为 [False, True, True]。
            dropout (float, 可选): Dropout 概率，默认为0.0。
        """
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # 计算每个阶段的维度
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        # 计算初始尺度
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # 3D因果卷积，将输入通道数从 z_dim 转换为 dims[0]
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # 中间块
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout))

        # 上采样块
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                # 如果是特定阶段，则调整输入维度
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                # 添加残差块
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    # 如果当前尺度在 attn_scales 中，则添加注意力块
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # 重采样块
            if i != len(dim_mult) - 1:
                # 根据是否在时间维度上上采样选择模式
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                # 添加重采样块
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0 # 调整尺度
        # 构建上采样序列
        self.upsamples = nn.Sequential(*upsamples)

        # 输出块
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1))  # 3D因果卷积，将输出通道数转换为3

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        """
        前向传播过程，应用3D解码器。

        参数:
            x (torch.Tensor): 输入张量，形状为 [B, C, T, H, W]。
            feat_cache (Optional[torch.Tensor], 可选): 特征缓存，默认为 None。
            feat_idx (List[int], 可选): 特征索引列表，默认为 [0]。

        返回:
            torch.Tensor: 解码后的张量，形状为 [B, 3, T, H, W]。
        """
        # 应用卷积层
        if feat_cache is not None:
            # 获取当前特征索引
            idx = feat_idx[0]
            # 复制输入张量的最后几帧作为缓存
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # **缓存处理**: 如果缓存长度小于2，并且缓存不为空，则将缓存的最后一帧与当前输入拼接
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            # 应用3D因果卷积，并使用缓存
            x = self.conv1(x, feat_cache[idx])
            # 更新缓存
            feat_cache[idx] = cache_x
            # 增加特征索引
            feat_idx[0] += 1
        else:
            # 否则，直接应用3D因果卷积
            x = self.conv1(x)

        # 中间块
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                # 应用残差块，并使用缓存
                x = layer(x, feat_cache, feat_idx)
            else:
                # 否则，直接应用当前层
                x = layer(x)

        # 上采样块
        for layer in self.upsamples:
            if feat_cache is not None:
                # 应用上采样块，并使用缓存
                x = layer(x, feat_cache, feat_idx)
            else:
                # 否则，直接应用上采样块
                x = layer(x)

        # 输出块
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                # 获取当前特征索引
                idx = feat_idx[0]
                # 复制输入张量的最后几帧作为缓存
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # **缓存处理**: 如果缓存长度小于2，并且缓存不为空，则将缓存的最后一帧与当前输入拼接
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                # 应用3D因果卷积，并使用缓存
                x = layer(x, feat_cache[idx])
                # 更新缓存
                feat_cache[idx] = cache_x
                # 增加特征索引
                feat_idx[0] += 1
            else:
                # 否则，直接应用当前层
                x = layer(x)
        # 返回解码后的张量
        return x


def count_conv3d(model):
    """
    计算模型中 CausalConv3d 层的数量。

    参数:
        model (nn.Module): 需要计数的模型。

    返回:
        int: CausalConv3d 层的数量。
    """
    count = 0  # 初始化计数器
    # 遍历模型的所有模块
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            # 如果模块是 CausalConv3d，则计数加1
            count += 1
    return count


class WanVAE_(nn.Module):
    """
    WanVAE_ 类实现了3D变分自编码器（VAE），用于视频数据的编码和解码。

    该模型结合了3D编码器和解码器，并通过重参数化技巧实现潜在空间的采样。
    """
    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        """
        初始化 WanVAE_。

        参数:
            dim (int, 可选): 模型的维度，默认为128。
            z_dim (int, 可选): 潜在空间的维度，默认为4。
            dim_mult (List[int], 可选): 维度倍增列表，默认为 [1, 2, 4, 4]。
            num_res_blocks (int, 可选): 每个阶段的残差块数量，默认为2。
            attn_scales (List[float], 可选): 应用注意力的尺度列表，默认为 []。
            temperal_downsample (List[bool], 可选): 是否在时间维度上下采样，默认为 [True, True, False]。
            dropout (float, 可选): Dropout 概率，默认为0.0。
        """
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]  # 反转时间上采样列表

        # 定义模块
        # 3D编码器，输出维度为 z_dim * 2
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_downsample, dropout)
        # 1x1因果卷积
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        # 1x1因果卷积
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        # 3D解码器，输入维度为 z_dim
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_upsample, dropout)

    def forward(self, x):
        """
        前向传播过程，应用3D VAE。

        参数:
            x (torch.Tensor): 输入视频张量，形状为 [B, C, T, H, W]。

        返回:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 返回重建视频张量、均值和方差。
        """
        # 对输入进行编码，得到均值和方差
        mu, log_var = self.encode(x)
        # 对潜在空间进行重参数化采样
        z = self.reparameterize(mu, log_var)
        # 对采样后的潜在空间进行解码，得到重建视频
        x_recon = self.decode(z)
        # 返回重建视频、均值和方差
        return x_recon, mu, log_var

    def encode(self, x, scale):
        """
        对输入视频进行编码，得到潜在空间的均值和方差。

        参数:
            x (torch.Tensor): 输入视频张量，形状为 [B, C, T, H, W]。
            scale (Tuple[torch.Tensor, torch.Tensor]): 缩放参数，用于调整均值和方差。

        返回:
            Tuple[torch.Tensor, torch.Tensor]: 返回潜在空间的均值和方差。
        """
        # 清除缓存
        self.clear_cache()
        # 缓存处理
        t = x.shape[2] # 获取时间维度长度
        iter_ = 1 + (t - 1) // 4
        # 对encode输入的x，按时间拆分为1、4、4、4....
        for i in range(iter_):
            self._enc_conv_idx = [0] # 重置编码卷积索引
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],  # 对前1帧进行编码
                    feat_cache=self._enc_feat_map,  # 使用特征缓存
                    feat_idx=self._enc_conv_idx)  # 使用特征索引
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],  # 对后续4帧进行编码
                    feat_cache=self._enc_feat_map,  # 使用特征缓存
                    feat_idx=self._enc_conv_idx)  # 使用特征索引
                out = torch.cat([out, out_], 2) # 将编码结果拼接起来
        # 将输出拆分为均值和方差
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            # 对均值进行缩放
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1)
        else:
            # 对均值进行缩放
            mu = (mu - scale[0]) * scale[1]
        # 清除缓存
        self.clear_cache()
        return mu

    def decode(self, z, scale):
        """
        对潜在空间的采样进行解码，得到重建视频。

        参数:
            z (torch.Tensor): 潜在空间的采样，形状为 [B, C, T, H, W]。
            scale (Tuple[torch.Tensor, torch.Tensor]): 缩放参数，用于调整潜在空间。

        返回:
            torch.Tensor: 重建视频，形状为 [B, C, T, H, W]。
        """
        self.clear_cache()
        # **重参数化处理**
        if isinstance(scale[0], torch.Tensor):
            # 对潜在空间进行缩放
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            # 对潜在空间进行缩放
            z = z / scale[1] + scale[0]
        # 获取时间维度长度
        iter_ = z.shape[2]
        # 应用1x1因果卷积
        x = self.conv2(z)
        for i in range(iter_):
            # 重置解码卷积索引
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i:i + 1, :, :],  # 对第i帧进行解码
                    feat_cache=self._feat_map,  # 使用特征缓存
                    feat_idx=self._conv_idx)  # 使用特征索引
            else:
                out_ = self.decoder(
                    x[:, :, i:i + 1, :, :],  # 对第i帧进行解码
                    feat_cache=self._feat_map,  # 使用特征缓存
                    feat_idx=self._conv_idx)  # 使用特征索引
                # 将解码结果拼接起来
                out = torch.cat([out, out_], 2)
        self.clear_cache()
        return out

    def reparameterize(self, mu, log_var):
        """
        对潜在空间进行重参数化采样。

        参数:
            mu (torch.Tensor): 均值。
            log_var (torch.Tensor): 对数方差。

        返回:
            torch.Tensor: 重参数化后的采样。
        """
        std = torch.exp(0.5 * log_var)  # 计算标准差
        eps = torch.randn_like(std)  # 生成标准正态分布的随机数
        # 返回重参数化后的采样
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        """
        对输入图像进行采样。

        参数:
            imgs (torch.Tensor): 输入图像，形状为 [B, C, T, H, W]。
            deterministic (bool, 可选): 是否为确定性采样，默认为 False。

        返回:
            torch.Tensor: 采样结果。
        """
        # 对输入图像进行编码
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu  # 如果是确定性采样，则返回均值
        # 计算标准差，并限制对数方差范围
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        # 返回采样结果
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        """
        清除缓存。
        """
        # 计算解码器中 CausalConv3d 层的数量
        self._conv_num = count_conv3d(self.decoder)
        # 重置解码卷积索引
        self._conv_idx = [0]
        # 初始化特征缓存
        self._feat_map = [None] * self._conv_num
        # 缓存编码
        # 计算编码器中 CausalConv3d 层的数量
        self._enc_conv_num = count_conv3d(self.encoder)
        # 重置编码卷积索引
        self._enc_conv_idx = [0]
        # 初始化特征缓存
        self._enc_feat_map = [None] * self._enc_conv_num


def _video_vae(pretrained_path=None, z_dim=None, device='cpu', **kwargs):
    """
    初始化视频变分自编码器（Video VAE），适配自 Stable Diffusion 1.x, 2.x 和 XL。

    参数:
        pretrained_path (Optional[str], 可选): 预训练模型路径。
        z_dim (Optional[int], 可选): 潜在空间的维度。
        device (str, 可选): 设备类型，默认为 'cpu'。
        **kwargs: 其他关键字参数，用于配置模型。

    返回:
        WanVAE_: 配置好的视频 VAE 模型实例。
    """
    # 参数配置
    cfg = dict(
        dim=96,  # 模型维度，默认为96
        z_dim=z_dim,  # 潜在空间维度
        dim_mult=[1, 2, 4, 4],  # 维度倍增列表，默认为 [1, 2, 4, 4]
        num_res_blocks=2,  # 每个阶段的残差块数量，默认为2
        attn_scales=[],  # 应用注意力的尺度列表，默认为空列表
        temperal_downsample=[False, True, True],  # 时间维度上下采样标志，默认为 [False, True, True]
        dropout=0.0) # Dropout 概率，默认为0.0
    
    # 更新配置参数
    cfg.update(**kwargs)

    # 模型初始化
    with torch.device('meta'):
        model = WanVAE_(**cfg)

    # 加载预训练权重
    logging.info(f'loading {pretrained_path}')
    model.load_state_dict(
        torch.load(pretrained_path, map_location=device), assign=True)

    return model


class WanVAE:
    """
    WanVAE 类封装了视频变分自编码器（Video VAE），提供了编码和解码功能。

    该类使用预训练的 WanVAE_ 模型进行视频的压缩和解压缩。
    """
    def __init__(self,
                 z_dim=16,
                 vae_pth='cache/vae_step_411000.pth',
                 dtype=torch.float,
                 device="cuda"):
        """
        初始化 WanVAE。

        参数:
            z_dim (int, 可选): 潜在空间的维度，默认为16。
            vae_pth (str, 可选): 预训练模型路径，默认为 'cache/vae_step_411000.pth'。
            dtype (torch.dtype, 可选): 模型的数据类型，默认为 torch.float。
            device (str, 可选): 设备类型，默认为 "cuda"。
        """
        self.dtype = dtype
        self.device = device

        # 设置均值和标准差
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]  # 预训练模型的均值列表
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]  # 预训练模型的标准差列表

        # 将均值列表转换为张量
        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        # 将标准差列表转换为张量
        self.std = torch.tensor(std, dtype=dtype, device=device)
        # 计算缩放参数，用于调整输入和输出
        self.scale = [self.mean, 1.0 / self.std]

        # 初始化模型
        self.model = _video_vae(
            pretrained_path=vae_pth,  # 设置预训练模型路径
            z_dim=z_dim,  # 设置潜在空间维度
        ).eval().requires_grad_(False).to(device)  # 设置为评估模式，关闭梯度计算，并移动到指定设备

    def encode(self, videos):
        """
        对输入视频进行编码，得到潜在空间的表示。

        参数:
            videos (List[torch.Tensor]): 输入的视频列表，每个视频形状为 [C, T, H, W]。

        返回:
            List[torch.Tensor]: 编码后的潜在空间表示列表。
        """
        # 使用自动混合精度
        with amp.autocast(dtype=self.dtype):
            return [
                # 对每个视频进行编码，并去除批量维度
                self.model.encode(u.unsqueeze(0), self.scale).float().squeeze(0)
                for u in videos
            ]

    def decode(self, zs):
        """
        对潜在空间的表示进行解码，得到重建视频。

        参数:
            zs (List[torch.Tensor]): 潜在空间的表示列表。

        返回:
            List[torch.Tensor]: 重建的视频列表，每个视频形状为 [C, T, H, W]。
        """
        # 使用自动混合精度
        with amp.autocast(dtype=self.dtype):
            return [
                self.model.decode(u.unsqueeze(0),  # 对每个潜在空间表示进行解码，并限制输出范围在 [-1, 1]
                                  self.scale).float().clamp_(-1, 1).squeeze(0)
                for u in zs
            ]

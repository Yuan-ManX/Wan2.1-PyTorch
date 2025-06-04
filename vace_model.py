# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import register_to_config

from .model import WanAttentionBlock, WanModel, sinusoidal_embedding_1d


class VaceWanAttentionBlock(WanAttentionBlock):
    """
    VaceWanAttentionBlock 类继承自 WanAttentionBlock，用于实现带有前/后投影层的注意力块。

    该类在 WanAttentionBlock 的基础上，增加了前投影层和后投影层，用于在处理特定块时调整特征的维度。
    """
    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 block_id=0):
        """
        初始化 VaceWanAttentionBlock。

        参数:
            cross_attn_type (str): 交叉注意力的类型。
            dim (int): 模型的维度。
            ffn_dim (int): 前馈神经网络中间层的维度。
            num_heads (int): 注意力头的数量。
            window_size (Tuple[int, int], 可选): 滑动窗口局部注意力的窗口大小，默认为 (-1, -1)。
            qk_norm (bool, 可选): 是否对查询和键进行归一化，默认为 True。
            cross_attn_norm (bool, 可选): 是否对交叉注意力进行归一化，默认为 False。
            eps (float, 可选): 层归一化的 epsilon，默认为1e-6。
            block_id (int, 可选): 块的编号，用于标识特定块，默认为0。
        """
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size,
                         qk_norm, cross_attn_norm, eps)
        # 设置块的编号
        self.block_id = block_id
        if block_id == 0:
            # 如果块编号为0，则添加前投影层，用于调整特征的维度
            self.before_proj = nn.Linear(self.dim, self.dim)
            # 初始化前投影层的权重为全0
            nn.init.zeros_(self.before_proj.weight)
            # 初始化前投影层的偏置为全0
            nn.init.zeros_(self.before_proj.bias)
        # 添加后投影层，用于调整特征的维度
        self.after_proj = nn.Linear(self.dim, self.dim)
        # 初始化后投影层的权重为全0
        nn.init.zeros_(self.after_proj.weight)
        # 初始化后投影层的偏置为全0
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x, **kwargs):
        """
        前向传播过程，应用注意力块。

        参数:
            c (torch.Tensor): 上下文输入张量，形状为 [B, L, dim]。
            x (torch.Tensor): 输入张量，形状为 [B, L, dim]。
            **kwargs: 其他关键字参数。

        返回:
            Tuple[torch.Tensor, torch.Tensor]: 返回包含应用了注意力机制后的张量 (c) 和后投影后的张量 (c_skip)。
        """
        if self.block_id == 0:
            # 如果块编号为0，则将前投影层的输出与输入张量 x 相加
            c = self.before_proj(c) + x

        # 应用注意力机制和 FFN
        c = super().forward(c, **kwargs)
        # 应用后投影层
        c_skip = self.after_proj(c)
        return c, c_skip


class BaseWanAttentionBlock(WanAttentionBlock):
    """
    BaseWanAttentionBlock 类继承自 WanAttentionBlock，用于实现带有提示（hints）的注意力块。

    该类在 WanAttentionBlock 的基础上，增加了对提示（hints）的处理，用于在特定块中引入额外的上下文信息。
    """
    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 block_id=None):
        """
        初始化 BaseWanAttentionBlock。

        参数:
            cross_attn_type (str): 交叉注意力的类型。
            dim (int): 模型的维度。
            ffn_dim (int): 前馈神经网络中间层的维度。
            num_heads (int): 注意力头的数量。
            window_size (Tuple[int, int], 可选): 滑动窗口局部注意力的窗口大小，默认为 (-1, -1)。
            qk_norm (bool, 可选): 是否对查询和键进行归一化，默认为 True。
            cross_attn_norm (bool, 可选): 是否对交叉注意力进行归一化，默认为 False。
            eps (float, 可选): 层归一化的 epsilon，默认为1e-6。
            block_id (Optional[int], 可选): 块的编号，用于标识特定块，默认为 None。
        """
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size,
                         qk_norm, cross_attn_norm, eps)
        # 设置块的编号
        self.block_id = block_id

    def forward(self, x, hints, context_scale=1.0, **kwargs):
        """
        前向传播过程，应用带有提示的注意力块。

        参数:
            x (torch.Tensor): 输入张量，形状为 [B, L, dim]。
            hints (List[torch.Tensor]): 提示列表，每个元素为一个张量，形状为 [B, L, dim]。
            context_scale (float, 可选): 上下文缩放因子，默认为1.0。
            **kwargs: 其他关键字参数。

        返回:
            torch.Tensor: 应用了注意力块后的张量，形状为 [B, L, dim]。
        """
        x = super().forward(x, **kwargs)
        if self.block_id is not None:
            # 如果块编号不为 None，则将提示与上下文缩放因子相乘，并加到输出中
            x = x + hints[self.block_id] * context_scale
        # 返回最终输出
        return x


class VaceWanModel(WanModel):
    """
    VaceWanModel 类继承自 WanModel，用于实现 VaceWan 模型。

    该模型结合了视觉和文本数据，通过多个注意力块处理多模态数据。
    """
    @register_to_config
    def __init__(self,
                 vace_layers=None,
                 vace_in_dim=None,
                 model_type='vace',
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
        初始化 VaceWanModel。

        参数:
            vace_layers (Optional[List[int]], 可选): Vace 层编号列表，默认为 None。
            vace_in_dim (Optional[int], 可选): Vace 层的输入维度，默认为 None。
            model_type (str, 可选): 模型类型，默认为 'vace'。
            patch_size (Tuple[int, int, int], 可选): 图像块大小，默认为 (1, 2, 2)。
            text_len (int, 可选): 文本长度，默认为512。
            in_dim (int, 可选): 输入维度，默认为16。
            dim (int, 可选): 模型维度，默认为2048。
            ffn_dim (int, 可选): 前馈神经网络中间层维度，默认为8192。
            freq_dim (int, 可选): 频率维度，默认为256。
            text_dim (int, 可选): 文本模型维度，默认为4096。
            out_dim (int, 可选): 输出维度，默认为16。
            num_heads (int, 可选): 注意力头的数量，默认为16。
            num_layers (int, 可选): 变换器层的数量，默认为32。
            window_size (Tuple[int, int], 可选): 滑动窗口局部注意力的窗口大小，默认为 (-1, -1)。
            qk_norm (bool, 可选): 是否对查询和键进行归一化，默认为 True。
            cross_attn_norm (bool, 可选): 是否对交叉注意力进行归一化，默认为 True。
            eps (float, 可选): 层归一化的 epsilon，默认为1e-6。
        """
        super().__init__(model_type, patch_size, text_len, in_dim, dim, ffn_dim,
                         freq_dim, text_dim, out_dim, num_heads, num_layers,
                         window_size, qk_norm, cross_attn_norm, eps)
        # Vace 层列表
        # 如果未提供 vace_layers，则每隔两层设置一个 Vace 层
        self.vace_layers = [i for i in range(0, self.num_layers, 2)
                           ] if vace_layers is None else vace_layers
        
        # 设置 Vace 层的输入维度
        self.vace_in_dim = self.in_dim if vace_in_dim is None else vace_in_dim

        # 确保第一个块是 Vace 层
        assert 0 in self.vace_layers
        # Vace 层映射
        self.vace_layers_mapping = {
            i: n for n, i in enumerate(self.vace_layers)
        }  # 创建 Vace 层编号到索引的映射

        # 构建基础注意力块列表
        self.blocks = nn.ModuleList([
            BaseWanAttentionBlock(
                't2v_cross_attn',  # 交叉注意力类型
                self.dim,  # 模型维度
                self.ffn_dim,  # 前馈神经网络中间层维度
                self.num_heads,  # 注意力头数量
                self.window_size,  # 滑动窗口局部注意力窗口大小
                self.qk_norm,  # 是否对查询和键进行归一化
                self.cross_attn_norm,  # 是否对交叉注意力进行归一化
                self.eps,  # 层归一化 epsilon
                block_id=self.vace_layers_mapping[i]  # Vace 层编号映射
                if i in self.vace_layers else None)  # 如果当前块是 Vace 层，则设置 block_id；否则为 None
            for i in range(self.num_layers)
        ])

        # 构建 Vace 注意力块列表
        self.vace_blocks = nn.ModuleList([
            VaceWanAttentionBlock(
                't2v_cross_attn',  # 交叉注意力类型
                self.dim,  # 模型维度
                self.ffn_dim,  # 前馈神经网络中间层维度
                self.num_heads,  # 注意力头数量
                self.window_size,  # 滑动窗口局部注意力窗口大小
                self.qk_norm,  # 是否对查询和键进行归一化
                self.cross_attn_norm,  # 是否对交叉注意力进行归一化
                self.eps,  # 层归一化 epsilon
                block_id=i) for i in self.vace_layers  # 当前块的编号
        ])

        # Vace 块嵌入
        self.vace_patch_embedding = nn.Conv3d(
            self.vace_in_dim,  # 输入通道数
            self.dim,   # 输出通道数
            kernel_size=self.patch_size,  # 卷积核大小
            stride=self.patch_size)  # 卷积步幅

    def forward_vace(self, x, vace_context, seq_len, kwargs):
        """
        VaceWan 模型的前向传播方法，用于处理 Vace 上下文。

        参数:
            x (List[torch.Tensor]): 输入的视频张量列表，每个张量形状为 [C_in, F, H, W]。
            vace_context (List[torch.Tensor]): Vace 上下文列表，每个张量形状为 [B, L, C]。
            seq_len (int): 序列的最大长度，用于位置编码。
            kwargs (dict): 其他关键字参数。

        返回:
            List[torch.Tensor]: 处理后的 Vace 上下文列表。
        """
        # 嵌入层
        # 对每个上下文张量应用 3D 卷积嵌入
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        # 将嵌入后的张量展平并转置
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], # 对每个上下文张量进行填充，使其长度与序列长度一致
                      dim=1) for u in c
        ])

        # 参数处理
        new_kwargs = dict(x=x)  # 将输入视频张量添加到关键字参数中
        new_kwargs.update(kwargs)  # 更新关键字参数
        
        # 初始化提示列表
        hints = []
        for block in self.vace_blocks: # 遍历每个 Vace 注意力块
            c, c_skip = block(c, **new_kwargs) # 应用注意力块
            hints.append(c_skip)  # 将跳过连接的输出添加到提示列表中
        return hints

    def forward(
        self,
        x,
        t,
        vace_context,
        context,
        seq_len,
        vace_context_scale=1.0,
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
            clip_fea (Optional[torch.Tensor], 可选): CLIP 图像特征，用于图像到视频模式。
            y (Optional[List[torch.Tensor]], 可选): 条件视频输入，用于图像到视频模式，形状与 x 相同。

        返回:
            List[torch.Tensor]: 去噪后的视频张量列表，原始输入形状为 [C_out, F, H / 8, W / 8]。
        """
        # 获取设备信息
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # 嵌入层
        # 对每个输入视频张量应用 3D 卷积嵌入
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])  # 获取每个视频帧的网格大小
        # 将嵌入后的张量展平并转置
        x = [u.flatten(2).transpose(1, 2) for u in x]  
        # 计算每个视频帧的序列长度
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)  

        # 确保序列长度不超过最大序列长度
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # 时间嵌入
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float()) # 生成时间嵌入
            # 对时间嵌入进行投影
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            # 确保数据类型为 float32
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # 文本嵌入
        # 初始化文本长度
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    # 对每个文本嵌入进行填充，使其长度与最大文本长度一致
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context   # 对文本嵌入进行嵌入层处理
            ]))

        # CLIP 图像特征处理
        # if clip_fea is not None:
        #     context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        #     context = torch.concat([context_clip, context], dim=1)

        # 参数处理
        kwargs = dict(
            e=e0,  # 时间嵌入
            seq_lens=seq_lens,  # 序列长度
            grid_sizes=grid_sizes, # 网格大小
            freqs=self.freqs, # 频率
            context=context, # 文本嵌入
            context_lens=context_lens) # 文本长度

        hints = self.forward_vace(x, vace_context, seq_len, kwargs)
        kwargs['hints'] = hints  # 将提示添加到关键字参数中
        kwargs['context_scale'] = vace_context_scale  # 设置上下文缩放因子

        # 遍历每个注意力块
        for block in self.blocks:
            # 应用注意力块
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)  # 应用头部

        # 去块化
        # 对嵌入后的张量进行去块化
        x = self.unpatchify(x, grid_sizes)
        # 返回去噪后的视频张量列表
        return [u.float() for u in x]

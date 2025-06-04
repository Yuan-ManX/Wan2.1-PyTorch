# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import time
import traceback
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from .modules.vace_model import VaceWanModel
from .text2video import (
    FlowDPMSolverMultistepScheduler,
    FlowUniPCMultistepScheduler,
    T5EncoderModel,
    WanT2V,
    WanVAE,
    get_sampling_sigmas,
    retrieve_timesteps,
    shard_model,
)
from .utils.vace_processor import VaceVideoProcessor


class WanVace(WanT2V):

    """
    WanVace 类继承自 WanT2V，用于初始化和管理文本生成视频模型的各个组件。
    该类配置了模型参数、设备、分布式训练设置等，并加载了文本编码器、VAE 模型和视频生成模型。
    """
    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        """
        初始化 WanVace 模型组件。

        参数:
            config (EasyDict): 从 config.py 初始化的包含模型参数的对象。
            checkpoint_dir (str): 包含模型检查点的目录路径。
            device_id (int, 可选): 目标 GPU 设备的 ID，默认为0。
            rank (int, 可选): 分布式训练的进程排名，默认为0。
            t5_fsdp (bool, 可选): 是否启用 T5 模型的 FSDP 分片，默认为False。
            dit_fsdp (bool, 可选): 是否启用 DiT 模型的 FSDP 分片，默认为False。
            use_usp (bool, 可选): 是否启用 USP 分布策略，默认为False。
            t5_cpu (bool, 可选): 是否将 T5 模型放置在 CPU 上。仅在没有启用 t5_fsdp 时有效，默认为False。
        """
        # 设置设备为指定的 GPU 设备
        self.device = torch.device(f"cuda:{device_id}")

        # 初始化配置、排名和 T5 CPU 标志
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        # 从配置中获取训练时间步数和参数数据类型
        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        # 定义分片模型的函数，如果启用 t5_fsdp，则使用 shard_model 并传入 device_id
        shard_fn = partial(shard_model, device_id=device_id)

        # 初始化 T5 文本编码器模型
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,  # 文本长度
            dtype=config.t5_dtype,  # 数据类型
            device=torch.device('cpu'),  # 设备设置为 CPU
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),  # 检查点路径
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),  # 分词器路径
            shard_fn=shard_fn if t5_fsdp else None)  # 如果启用 t5_fsdp，则应用分片函数

        # 从配置中获取 VAE 步幅和补丁大小
        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size

        # 初始化 VAE 模型，加载预训练权重并放置在指定设备上
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        # 输出日志信息，指示正在从指定目录创建 VaceWanModel
        logging.info(f"Creating VaceWanModel from {checkpoint_dir}")

        # 从预训练检查点加载 VaceWanModel 模型，并设置为评估模式且不计算梯度
        self.model = VaceWanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        # 如果启用 USP 策略
        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
                usp_dit_forward_vace,
            )

            # 遍历模型中的每个块，并将 self_attn 的 forward 方法替换为 usp_attn_forward
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
                
            # 遍历模型中的每个 Vace 块，并将 self_attn 的 forward 方法替换为 usp_attn_forward
            for block in self.model.vace_blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
                
            # 将模型的 forward 方法替换为 usp_dit_forward
            self.model.forward = types.MethodType(usp_dit_forward, self.model)

            # 将模型的 forward_vace 方法替换为 usp_dit_forward_vace
            self.model.forward_vace = types.MethodType(usp_dit_forward_vace,
                                                       self.model)
            
            # 获取序列并行世界大小
            self.sp_size = get_sequence_parallel_world_size()
        else:
            # 如果未启用 USP 策略，序列并行大小设为1
            self.sp_size = 1

        # 如果分布式环境已初始化，则等待所有进程同步
        if dist.is_initialized():
            dist.barrier()

        # 如果启用 DiT FSDP 分片，则对模型进行分片
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            # 否则，将模型移动到指定设备
            self.model.to(self.device)

        # 从配置中获取负提示样本
        self.sample_neg_prompt = config.sample_neg_prompt

        # 初始化视频处理器，配置下采样参数、最小/最大面积、最小/最大 FPS 等
        self.vid_proc = VaceVideoProcessor(
            downsample=tuple(
                [x * y for x, y in zip(config.vae_stride, self.patch_size)]),  # 下采样尺寸
            min_area=720 * 1280,  # 最小面积
            max_area=720 * 1280,  # 最大面积
            min_fps=config.sample_fps,  # 最小 FPS
            max_fps=config.sample_fps,  # 最大 FPS
            zero_start=True,  # 是否从零开始
            seq_len=75600,  # 序列长度
            keep_last=True)  # 是否保留最后

    def vace_encode_frames(self, frames, ref_images, masks=None, vae=None):
        """
        对视频帧进行编码，生成潜在表示。如果提供了掩码，则分别对非活动区域和活动区域进行编码并合并。
        如果提供了参考图像，则将其潜在表示与帧的潜在表示合并。

        参数:
            frames (List[torch.Tensor]): 输入的视频帧列表，每个帧为一个张量。
            ref_images (Optional[List[torch.Tensor]]): 参考图像列表，可为None。
            masks (Optional[List[torch.Tensor]]): 掩码列表，用于区分非活动区域和活动区域，可为None。
            vae (Optional[object]): 可选的 VAE 模型实例，默认为None。如果提供，则使用该 VAE 进行编码。

        返回:
            List[torch.Tensor]: 编码后的潜在表示列表。
        """
        # 如果未提供 VAE，则使用初始化时加载的 VAE
        vae = self.vae if vae is None else vae

        # 如果未提供参考图像，则为每个帧分配一个 None
        if ref_images is None:
            ref_images = [None] * len(frames)
        else:
            # 确保帧和参考图像的数量一致
            assert len(frames) == len(ref_images)

        # 如果未提供掩码，则直接对所有帧进行编码
        if masks is None:
            latents = vae.encode(frames)
        else:
            # 将掩码二值化，大于0.5的像素设为1.0，其余设为0.0
            masks = [torch.where(m > 0.5, 1.0, 0.0) for m in masks]
            # 生成非活动区域图像（帧 * (1 - 掩码)）
            inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks)]
            # 生成活动区域图像（帧 * 掩码）
            reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks)]

            # 对非活动区域和活动区域图像进行编码
            inactive = vae.encode(inactive)
            reactive = vae.encode(reactive)

            # 将非活动区域和活动区域的潜在表示在通道维度上拼接
            latents = [
                torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)
            ]

        # 合并参考图像的潜在表示（如果有的话）
        cat_latents = []
        for latent, refs in zip(latents, ref_images):
            if refs is not None:
                if masks is None:
                    # 如果没有掩码，则直接对参考图像进行编码
                    ref_latent = vae.encode(refs)
                else:
                    # 如果有掩码，则对参考图像进行编码，并将其潜在表示与零张量拼接
                    ref_latent = vae.encode(refs)
                    ref_latent = [
                        torch.cat((u, torch.zeros_like(u)), dim=0)
                        for u in ref_latent
                    ]
                # 确保参考图像的潜在表示在通道维度上为1
                assert all([x.shape[1] == 1 for x in ref_latent])
                # 将参考图像的潜在表示与帧的潜在表示在通道维度上拼接
                latent = torch.cat([*ref_latent, latent], dim=1)
            cat_latents.append(latent)
        return cat_latents

    def vace_encode_masks(self, masks, ref_images=None, vae_stride=None):
        """
        对掩码进行编码和下采样处理。如果提供了参考图像，则在编码后的掩码中添加填充。

        参数:
            masks (List[torch.Tensor]): 输入的掩码列表，每个掩码为一个张量。
            ref_images (Optional[List[torch.Tensor]]): 参考图像列表，可为None。
            vae_stride (Optional[List[int]]): VAE 步幅参数列表，默认为None。如果为None，则使用初始化时设置的 vae_stride。

        返回:
            List[torch.Tensor]: 处理后的掩码列表。
        """
        # 如果未提供 vae_stride，则使用初始化时设置的 vae_stride
        vae_stride = self.vae_stride if vae_stride is None else vae_stride

        # 如果未提供参考图像，则为每个掩码分配一个 None
        if ref_images is None:
            ref_images = [None] * len(masks)
        else:
            # 确保掩码和参考图像的数量一致
            assert len(masks) == len(ref_images)

        result_masks = []
        for mask, refs in zip(masks, ref_images):
            # 获取掩码的形状
            c, depth, height, width = mask.shape
            # 计算新的深度尺寸
            new_depth = int((depth + 3) // vae_stride[0])
            # 计算新的高度和宽度尺寸，确保是 2 的倍数
            height = 2 * (int(height) // (vae_stride[1] * 2))
            width = 2 * (int(width) // (vae_stride[2] * 2))

            # 重塑掩码形状
            mask = mask[0, :, :, :]
            mask = mask.view(depth, height, vae_stride[1], width,
                             vae_stride[1])  # depth, height, 8, width, 8
            mask = mask.permute(2, 4, 0, 1, 3)  # 8, 8, depth, height, width
            mask = mask.reshape(vae_stride[1] * vae_stride[2], depth, height,
                                width)  # 8*8, depth, height, width

            # 对掩码进行插值下采样
            mask = F.interpolate(
                mask.unsqueeze(0),
                size=(new_depth, height, width),
                mode='nearest-exact').squeeze(0)

            # 如果提供了参考图像，则在编码后的掩码中添加填充
            if refs is not None:
                length = len(refs)
                mask_pad = torch.zeros_like(mask[:, :length, :, :])
                mask = torch.cat((mask_pad, mask), dim=1)
            result_masks.append(mask)
        return result_masks

    def vace_latent(self, z, m):
        """
        将潜在表示与掩码潜在表示在通道维度上拼接。

        参数:
            z (List[torch.Tensor]): 潜在表示列表。
            m (List[torch.Tensor]): 掩码潜在表示列表。

        返回:
            List[torch.Tensor]: 合并后的潜在表示列表。
        """
        return [torch.cat([zz, mm], dim=0) for zz, mm in zip(z, m)]

    def prepare_source(self, src_video, src_mask, src_ref_images, num_frames,
                       image_size, device):
        """
        准备视频源数据，包括调整视频尺寸、处理掩码以及加载参考图像。

        参数:
            src_video (List[torch.Tensor]): 输入的视频帧列表，每个视频为一个张量。
            src_mask (List[torch.Tensor]): 输入的掩码列表，每个掩码为一个张量。
            src_ref_images (List[List[Optional[str]]]): 参考图像路径的嵌套列表，每个视频对应一个参考图像列表。
            num_frames (int): 视频的总帧数。
            image_size (Tuple[int, int]): 目标图像尺寸 (高度, 宽度)。
            device (torch.device): 目标设备，如 GPU 或 CPU。

        返回:
            Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]]]: 处理后的视频帧列表、掩码列表和参考图像列表。
        """
        # 计算目标图像的面积
        area = image_size[0] * image_size[1]
        # 设置视频处理器的面积参数
        self.vid_proc.set_area(area)

        # 根据面积设置序列长度
        if area == 720 * 1280:
            self.vid_proc.set_seq_len(75600)
        elif area == 480 * 832:
            self.vid_proc.set_seq_len(32760)
        else:
            # 如果面积不支持，则抛出异常
            raise NotImplementedError(
                f'image_size {image_size} is not supported')

        # 调整图像尺寸顺序为 (宽度, 高度)
        image_size = (image_size[1], image_size[0])
        # 存储每个视频的实际图像尺寸
        image_sizes = []

        # 遍历每个视频及其对应的掩码
        for i, (sub_src_video,
                sub_src_mask) in enumerate(zip(src_video, src_mask)):
            if sub_src_mask is not None and sub_src_video is not None:
                # 使用视频处理器加载视频对（视频和掩码）
                src_video[i], src_mask[
                    i], _, _, _ = self.vid_proc.load_video_pair(
                        sub_src_video, sub_src_mask)
                # 将视频帧和掩码移动到目标设备
                src_video[i] = src_video[i].to(device)
                src_mask[i] = src_mask[i].to(device)
                # 对掩码进行归一化处理，确保值在 [0, 1] 之间
                src_mask[i] = torch.clamp(
                    (src_mask[i][:1, :, :, :] + 1) / 2, min=0, max=1)
                # 记录视频的实际图像尺寸
                image_sizes.append(src_video[i].shape[2:])
            elif sub_src_video is None:
                # 如果视频为 None，则用零张量填充视频帧
                src_video[i] = torch.zeros(
                    (3, num_frames, image_size[0], image_size[1]),
                    device=device)
                # 用全1张量填充掩码
                src_mask[i] = torch.ones_like(src_video[i], device=device)
                # 记录图像尺寸
                image_sizes.append(image_size)
            else:
                # 如果只有视频而没有掩码，则加载视频并用全1张量填充掩码
                src_video[i], _, _, _ = self.vid_proc.load_video(sub_src_video)
                src_video[i] = src_video[i].to(device)
                src_mask[i] = torch.ones_like(src_video[i], device=device)
                image_sizes.append(src_video[i].shape[2:])

        # 处理参考图像
        for i, ref_images in enumerate(src_ref_images):
            if ref_images is not None:
                # 获取当前视频的实际图像尺寸
                image_size = image_sizes[i]
                for j, ref_img in enumerate(ref_images):
                    if ref_img is not None:
                        # 打开参考图像并转换为 RGB 格式
                        ref_img = Image.open(ref_img).convert("RGB")
                        # 将图像转换为张量，并进行归一化处理
                        ref_img = TF.to_tensor(ref_img).sub_(0.5).div_(
                            0.5).unsqueeze(1)
                        # 检查参考图像的尺寸是否与目标尺寸一致
                        if ref_img.shape[-2:] != image_size:
                            canvas_height, canvas_width = image_size
                            ref_height, ref_width = ref_img.shape[-2:]
                            # 创建白色画布
                            white_canvas = torch.ones(
                                (3, 1, canvas_height, canvas_width),
                                device=device)  # 值范围为 [-1, 1]
                            # 计算缩放比例
                            scale = min(canvas_height / ref_height,
                                        canvas_width / ref_width)
                            new_height = int(ref_height * scale)
                            new_width = int(ref_width * scale)
                            # 对参考图像进行插值缩放
                            resized_image = F.interpolate(
                                ref_img.squeeze(1).unsqueeze(0),
                                size=(new_height, new_width),
                                mode='bilinear',
                                align_corners=False).squeeze(0).unsqueeze(1)
                            # 计算填充位置
                            top = (canvas_height - new_height) // 2
                            left = (canvas_width - new_width) // 2
                            # 将缩放后的图像放置在画布中央
                            white_canvas[:, :, top:top + new_height,
                                         left:left + new_width] = resized_image
                            ref_img = white_canvas
                        # 将处理后的参考图像移动到目标设备
                        src_ref_images[i][j] = ref_img.to(device)
        return src_video, src_mask, src_ref_images

    def decode_latent(self, zs, ref_images=None, vae=None):
        """
        解码潜在表示，生成视频帧。如果提供了参考图像，则在解码过程中考虑这些图像。

        参数:
            zs (List[torch.Tensor]): 输入的潜在表示列表，每个潜在表示为一个张量。
            ref_images (Optional[List[torch.Tensor]]): 参考图像列表，可为 None。
            vae (Optional[object]): 可选的 VAE 模型实例，默认为 None。如果提供，则使用该 VAE 进行解码。

        返回:
            List[torch.Tensor]: 解码后的视频帧列表。
        """
        # 如果未提供 VAE，则使用初始化时加载的 VAE
        vae = self.vae if vae is None else vae

        # 如果未提供参考图像，则为每个潜在表示分配一个 None
        if ref_images is None:
            ref_images = [None] * len(zs)
        else:
            # 确保潜在表示和参考图像的数量一致
            assert len(zs) == len(ref_images)

        # 对每个潜在表示进行处理，移除参考图像对应的部分
        trimed_zs = []
        for z, refs in zip(zs, ref_images):
            if refs is not None:
                z = z[:, len(refs):, :, :]
            trimed_zs.append(z)
        # 使用 VAE 对处理后的潜在表示进行解码，生成视频帧
        return vae.decode(trimed_zs)

    def generate(self,
                 input_prompt,
                 input_frames,
                 input_masks,
                 input_ref_images,
                 size=(1280, 720),
                 frame_num=81,
                 context_scale=1.0,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        """
        使用扩散过程从文本提示生成视频帧。

        参数:
            input_prompt (str): 用于生成内容的文本提示。
            size (Tuple[int, int], 可选): 控制视频分辨率，(宽度, 高度)，默认为 (1280, 720)。
            frame_num (int, 可选): 从视频中采样的帧数，默认为81。该数字应为4n+1。
            context_scale (float, 可选): 上下文缩放因子，影响生成过程中的上下文影响程度，默认为1.0。
            shift (float, 可选): 噪声调度移位参数，影响时间动态，默认为5.0。
            sample_solver (str, 可选): 用于采样视频的求解器，默认为 'unipc'。
            sampling_steps (int, 可选): 扩散采样的步数，默认为40。较高的值可以提高质量，但会减慢生成速度。
            guide_scale (float, 可选): 无分类器指导比例，控制提示的遵循程度与创造力之间的平衡，默认为5.0。
            n_prompt (str, 可选): 用于排除内容的负提示。如果未提供，则使用 `config.sample_neg_prompt`。
            seed (int, 可选): 用于噪声生成的随机种子。如果为-1，则使用随机种子，默认为-1。
            offload_model (bool, 可选): 如果为True，则在生成过程中将模型卸载到 CPU 以节省 VRAM，默认为True。

        返回:
            torch.Tensor: 生成的视频帧张量。维度为 (C, N, H, W)，其中:
                - C: 颜色通道数 (3 为 RGB)
                - N: 帧数 (81)
                - H: 帧高度 (来自 size)
                - W: 帧宽度 (来自 size)
        """
        # 预处理
        # F = frame_num
        # target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
        #                 size[1] // self.vae_stride[1],
        #                 size[0] // self.vae_stride[2])
        #
        # seq_len = math.ceil((target_shape[2] * target_shape[3]) /
        #                     (self.patch_size[1] * self.patch_size[2]) *
        #                     target_shape[1] / self.sp_size) * self.sp_size

        # 如果未提供负提示，则使用初始化时设置的负提示
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        
        # 设置随机种子。如果 seed 为 -1，则生成一个随机种子
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        # 如果未启用 T5 CPU，则将文本编码器模型移动到目标设备
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            # 对输入提示进行编码
            context = self.text_encoder([input_prompt], self.device)
            # 对负提示进行编码
            context_null = self.text_encoder([n_prompt], self.device)
            # 如果启用了模型卸载，则将文本编码器模型移动回 CPU
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            # 如果启用了 T5 CPU，则在 CPU 上对提示进行编码，然后将结果移动到目标设备
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        # 对视频帧进行 Vace 上下文编码
        z0 = self.vace_encode_frames(
            input_frames, input_ref_images, masks=input_masks)
        # 对掩码进行 Vace 掩码编码
        m0 = self.vace_encode_masks(input_masks, input_ref_images)
        # 将编码后的帧和掩码合并为潜在表示
        z = self.vace_latent(z0, m0)

        # 设置目标形状为编码后潜在表示的形状
        target_shape = list(z0[0].shape)
        # 调整目标形状的第一个维度（通道数）
        target_shape[0] = int(target_shape[0] / 2)
        # 生成与目标形状相同的随机噪声
        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]
        # 计算序列长度，用于模型输入
        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        # 定义一个上下文管理器，用于在模型前向传播时不进行梯度同步
        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # 进入评估模式，并设置自动混合精度和梯度不计算
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            # 初始化 UniPC 多步调度器
            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                # 设置调度器的步数
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                # 初始化 DPM++ 多步调度器
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                # 获取采样 sigmas
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                # 检索调度器的步数
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                # 如果不支持的求解器，则抛出异常
                raise NotImplementedError("Unsupported solver.")

            # 初始化潜在表示为随机噪声
            latents = noise

            # 定义上下文参数
            arg_c = {'context': context, 'seq_len': seq_len}
            # 定义负上下文参数
            arg_null = {'context': context_null, 'seq_len': seq_len}

            # 遍历每个时间步
            for _, t in enumerate(tqdm(timesteps)):
                # 将当前潜在表示作为模型输入
                latent_model_input = latents
                # 将当前时间步堆叠为张量
                timestep = [t]

                # 将模型移动到目标设备
                timestep = torch.stack(timestep)

                # 进行有条件和无条件预测
                self.model.to(self.device)
                noise_pred_cond = self.model(
                    latent_model_input,
                    t=timestep,
                    vace_context=z,
                    vace_context_scale=context_scale,
                    **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input,
                    t=timestep,
                    vace_context=z,
                    vace_context_scale=context_scale,
                    **arg_null)[0]
                
                # 计算最终的噪声预测
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                # 使用调度器进行一步采样
                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            # 最终的潜在表示
            x0 = latents
            # 如果启用了模型卸载，则将模型移动回 CPU 并清空 CUDA 缓存
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            # 如果是主进程，则解码潜在表示生成视频
            if self.rank == 0:
                videos = self.decode_latent(x0, input_ref_images)

        # 删除临时变量以释放内存
        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        # 如果分布式环境已初始化，则同步所有进程
        if dist.is_initialized():
            dist.barrier()

        # 返回生成的视频。如果不是主进程，则返回 None
        return videos[0] if self.rank == 0 else None


class WanVaceMP(WanVace):

    """
    WanVaceMP 类继承自 WanVace，用于初始化和管理多进程推理的文本生成视频模型。
    该类扩展了多进程支持，以实现并行推理和更高效的计算资源利用。
    """
    def __init__(self,
                 config,
                 checkpoint_dir,
                 use_usp=False,
                 ulysses_size=None,
                 ring_size=None):
        """
        初始化 WanVaceMP 模型组件，包括多进程环境配置和视频处理器设置。

        参数:
            config: 配置对象，包含模型参数，初始化自 config.py。
            checkpoint_dir (str): 模型检查点所在的目录路径。
            use_usp (bool, 可选): 是否启用 USP 分布策略，默认为 False。
            ulysses_size (Optional[int], 可选): Ulysses 尺寸参数，可为 None。
            ring_size (Optional[int], 可选): Ring 尺寸参数，可为 None。
        """
        # 初始化配置、模型检查点目录、USP 使用标志等参数
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.use_usp = use_usp

        # 设置多进程环境变量
        os.environ['MASTER_ADDR'] = 'localhost'  # 主节点地址
        os.environ['MASTER_PORT'] = '12345'  # 主节点端口
        os.environ['RANK'] = '0'   # 当前进程的排名
        os.environ['WORLD_SIZE'] = '1'  # 进程总数

        # 初始化队列和进程相关属性
        self.in_q_list = None  # 输入队列列表
        self.out_q = None  # 输出队列
        self.inference_pids = None  # 推理进程的 PID 列表
        self.ulysses_size = ulysses_size  # Ulysses 尺寸参数
        self.ring_size = ring_size  # Ring 尺寸参数

        # 调用动态加载方法，初始化多进程推理环境
        self.dynamic_load()

        # 设置设备为 CPU。如果有可用的 GPU，则设置为 GPU
        self.device = 'cpu' if torch.cuda.is_available() else 'cpu'

        # 初始化视频处理器，设置下采样参数、最小/最大面积、最小/最大 FPS 等
        self.vid_proc = VaceVideoProcessor(
            downsample=tuple(
                [x * y for x, y in zip(config.vae_stride, config.patch_size)]),  # 下采样尺寸
            min_area=480 * 832,  # 最小面积
            max_area=480 * 832,  # 最大面积
            min_fps=self.config.sample_fps,  # 最小 FPS
            max_fps=self.config.sample_fps,  # 最大 FPS
            zero_start=True,  # 是否从零开始
            seq_len=32760,  # 序列长度
            keep_last=True)  # 是否保留最后

    def dynamic_load(self):
        """
        动态加载推理进程，实现多进程推理。

        该方法初始化多进程环境，设置必要的队列和事件，并启动推理进程。
        """
        # 如果已经初始化推理进程，则直接返回
        if hasattr(self, 'inference_pids') and self.inference_pids is not None:
            return
        
        # 获取 GPU 推理的数量。如果未设置 LOCAL_WORLD_SIZE，则使用可用的 GPU 数量
        gpu_infer = os.environ.get(
            'LOCAL_WORLD_SIZE') or torch.cuda.device_count()
        
        # 获取当前进程的排名和进程总数
        pmi_rank = int(os.environ['RANK'])
        pmi_world_size = int(os.environ['WORLD_SIZE'])

        # 为每个 GPU 推理创建一个输入队列
        in_q_list = [
            torch.multiprocessing.Manager().Queue() for _ in range(gpu_infer)
        ]

        # 创建一个输出队列
        out_q = torch.multiprocessing.Manager().Queue()

        # 为每个 GPU 推理创建一个初始化事件
        initialized_events = [
            torch.multiprocessing.Manager().Event() for _ in range(gpu_infer)
        ]

        # 使用 mp.spawn 启动多进程推理
        context = mp.spawn(
            self.mp_worker,  # 推理工作函数 
            nprocs=gpu_infer,  # 启动的进程数
            args=(gpu_infer, pmi_rank, pmi_world_size, in_q_list, out_q,
                  initialized_events, self),
            join=False)  # 不阻塞主进程
        
        # 等待所有推理进程初始化完成
        all_initialized = False
        while not all_initialized:
            all_initialized = all(
                event.is_set() for event in initialized_events)
            if not all_initialized:
                time.sleep(0.1)  # 休眠 0.1 秒后重试

        # 输出日志信息，指示推理模型已初始化
        print('Inference model is initialized', flush=True)
        # 记录输入队列列表、输出队列和推理进程的 PID
        self.in_q_list = in_q_list
        self.out_q = out_q
        self.inference_pids = context.pids()
        self.initialized_events = initialized_events

    def transfer_data_to_cuda(self, data, device):
        """
        将数据转移到指定的 CUDA 设备。

        参数:
            data: 要转移的数据，可以是张量、列表或字典。
            device (torch.device): 目标 CUDA 设备。

        返回:
            Optional[torch.Tensor]: 转移后的数据。如果输入为 None，则返回 None。
        """
        if data is None:
            return None
        else:
            if isinstance(data, torch.Tensor):
                # 如果数据是张量，则直接转移到目标设备
                data = data.to(device)
            elif isinstance(data, list):
                # 如果数据是列表，则递归调用 transfer_data_to_cuda 方法
                data = [
                    self.transfer_data_to_cuda(subdata, device)
                    for subdata in data
                ]
            elif isinstance(data, dict):
                # 如果数据是字典，则递归调用 transfer_data_to_cuda 方法
                data = {
                    key: self.transfer_data_to_cuda(val, device)
                    for key, val in data.items()
                }
        return data

    def mp_worker(self, gpu, gpu_infer, pmi_rank, pmi_world_size, in_q_list,
                  out_q, initialized_events, work_env):
        """
        多进程推理工作函数，处理推理任务。

        参数:
            gpu (int): 当前进程的 GPU 编号。
            gpu_infer (int): GPU 推理的数量。
            pmi_rank (int): 当前进程的排名。
            pmi_world_size (int): 进程总数。
            in_q_list (List[mp.Queue]): 输入队列列表。
            out_q (mp.Queue): 输出队列。
            initialized_events (List[mp.Event]): 初始化事件列表。
            work_env (WanVaceMP): 当前 WanVaceMP 对象。
        """
        try:
            # 计算全局的世界大小和当前进程的排名
            world_size = pmi_world_size * gpu_infer
            rank = pmi_rank * gpu_infer + gpu
            print("world_size", world_size, "rank", rank, flush=True)

            # 设置当前进程的 GPU 设备
            torch.cuda.set_device(gpu)
            # 初始化分布式进程组
            dist.init_process_group(
                backend='nccl',  # 后端使用 NCCL
                init_method='env://',  # 使用环境变量进行初始化
                rank=rank,  # 当前进程排名
                world_size=world_size)  # 世界大小

            # 初始化分布式环境
            from xfuser.core.distributed import (
                init_distributed_environment,
                initialize_model_parallel,
            )
            init_distributed_environment(
                rank=dist.get_rank(), world_size=dist.get_world_size())

            # 初始化模型并行
            initialize_model_parallel(
                sequence_parallel_degree=dist.get_world_size(),
                ring_degree=self.ring_size or 1,
                ulysses_degree=self.ulysses_size or 1)

            # 从配置中获取训练时间步数和参数数据类型
            num_train_timesteps = self.config.num_train_timesteps
            param_dtype = self.config.param_dtype

            # 定义分片函数
            shard_fn = partial(shard_model, device_id=gpu)

            # 初始化 T5 文本编码器模型
            text_encoder = T5EncoderModel(
                text_len=self.config.text_len,
                dtype=self.config.t5_dtype,
                device=torch.device('cpu'),
                checkpoint_path=os.path.join(self.checkpoint_dir,
                                             self.config.t5_checkpoint),
                tokenizer_path=os.path.join(self.checkpoint_dir,
                                            self.config.t5_tokenizer),
                shard_fn=shard_fn if True else None)  # 如果启用分片，则应用分片函数
            # 将文本编码器模型移动到当前 GPU
            text_encoder.model.to(gpu)

            # 从配置中获取 VAE 步幅和补丁大小
            vae_stride = self.config.vae_stride
            patch_size = self.config.patch_size

            # 初始化 VAE 模型，加载预训练权重并放置在当前 GPU 上
            vae = WanVAE(
                vae_pth=os.path.join(self.checkpoint_dir,
                                     self.config.vae_checkpoint),
                device=gpu)
            
            # 输出日志信息，指示正在创建 VaceWanModel
            logging.info(f"Creating VaceWanModel from {self.checkpoint_dir}")

            # 从预训练检查点加载 VaceWanModel 模型，并设置为评估模式且不计算梯度
            model = VaceWanModel.from_pretrained(self.checkpoint_dir)
            model.eval().requires_grad_(False)

            # 如果启用 USP 策略
            if self.use_usp:
                from xfuser.core.distributed import get_sequence_parallel_world_size

                from .distributed.xdit_context_parallel import (
                    usp_attn_forward,
                    usp_dit_forward,
                    usp_dit_forward_vace,
                )
                # 遍历模型中的每个块，并将 self_attn 的 forward 方法替换为 usp_attn_forward
                for block in model.blocks:
                    block.self_attn.forward = types.MethodType(
                        usp_attn_forward, block.self_attn)
                # 遍历模型中的每个 Vace 块，并将 self_attn 的 forward 方法替换为 usp_attn_forward
                for block in model.vace_blocks:
                    block.self_attn.forward = types.MethodType(
                        usp_attn_forward, block.self_attn)
                # 将模型的 forward 方法替换为 usp_dit_forward
                model.forward = types.MethodType(usp_dit_forward, model)
                # 将模型的 forward_vace 方法替换为 usp_dit_forward_vace
                model.forward_vace = types.MethodType(usp_dit_forward_vace,
                                                      model)
                # 获取序列并行世界大小
                sp_size = get_sequence_parallel_world_size()
            else:
                # 如果未启用 USP 策略，序列并行大小设为1
                sp_size = 1

            # 等待所有进程同步
            dist.barrier()
            # 应用分片函数对模型进行分片
            model = shard_fn(model)
            # 从配置中获取负提示样本
            sample_neg_prompt = self.config.sample_neg_prompt

            # 清空 CUDA 缓存
            torch.cuda.empty_cache()
            # 设置初始化事件为已设置状态
            event = initialized_events[gpu]
            in_q = in_q_list[gpu]
            event.set()

            # 进入一个无限循环，处理输入队列中的任务
            while True:
                # 从输入队列中获取任务
                item = in_q.get()
                input_prompt, input_frames, input_masks, input_ref_images, size, frame_num, context_scale, \
                shift, sample_solver, sampling_steps, guide_scale, n_prompt, seed, offload_model = item
                # 将输入数据转移到当前 GPU
                input_frames = self.transfer_data_to_cuda(input_frames, gpu)
                input_masks = self.transfer_data_to_cuda(input_masks, gpu)
                input_ref_images = self.transfer_data_to_cuda(
                    input_ref_images, gpu)

                # 如果负提示为空，则使用负提示样本
                if n_prompt == "":
                    n_prompt = sample_neg_prompt
                
                # 设置随机种子
                seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
                seed_g = torch.Generator(device=gpu)
                seed_g.manual_seed(seed)

                # 对输入提示和负提示进行编码
                context = text_encoder([input_prompt], gpu)
                context_null = text_encoder([n_prompt], gpu)

                # 对视频帧进行 Vace 上下文编码
                z0 = self.vace_encode_frames(
                    input_frames, input_ref_images, masks=input_masks, vae=vae)
                m0 = self.vace_encode_masks(
                    input_masks, input_ref_images, vae_stride=vae_stride)
                z = self.vace_latent(z0, m0)
                
                # 设置目标形状
                target_shape = list(z0[0].shape)
                target_shape[0] = int(target_shape[0] / 2)
                # 生成随机噪声
                noise = [
                    torch.randn(
                        target_shape[0],
                        target_shape[1],
                        target_shape[2],
                        target_shape[3],
                        dtype=torch.float32,
                        device=gpu,
                        generator=seed_g)
                ]
                # 计算序列长度
                seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                                    (patch_size[1] * patch_size[2]) *
                                    target_shape[1] / sp_size) * sp_size

                # 定义一个上下文管理器，用于在模型前向传播时不进行梯度同步
                @contextmanager
                def noop_no_sync():
                    yield

                no_sync = getattr(model, 'no_sync', noop_no_sync)

                # 进入评估模式，并设置自动混合精度和梯度不计算
                with amp.autocast(
                        dtype=param_dtype), torch.no_grad(), no_sync():

                    if sample_solver == 'unipc':
                        # 初始化 UniPC 多步调度器
                        sample_scheduler = FlowUniPCMultistepScheduler(
                            num_train_timesteps=num_train_timesteps,
                            shift=1,
                            use_dynamic_shifting=False)
                        # 设置调度器的步数
                        sample_scheduler.set_timesteps(
                            sampling_steps, device=gpu, shift=shift)
                        timesteps = sample_scheduler.timesteps
                    
                    elif sample_solver == 'dpm++':
                        # 初始化 DPM++ 多步调度器
                        sample_scheduler = FlowDPMSolverMultistepScheduler(
                            num_train_timesteps=num_train_timesteps,
                            shift=1,
                            use_dynamic_shifting=False)
                        # 获取采样 sigmas
                        sampling_sigmas = get_sampling_sigmas(
                            sampling_steps, shift)
                        # 检索调度器的步数
                        timesteps, _ = retrieve_timesteps(
                            sample_scheduler,
                            device=gpu,
                            sigmas=sampling_sigmas)
                    else:
                        # 如果不支持的求解器，则抛出异常
                        raise NotImplementedError("Unsupported solver.")

                    # 初始化潜在表示为随机噪声
                    latents = noise

                    # 定义上下文参数
                    arg_c = {'context': context, 'seq_len': seq_len}
                    # 定义负上下文参数
                    arg_null = {'context': context_null, 'seq_len': seq_len}

                    # 遍历每个时间步
                    for _, t in enumerate(tqdm(timesteps)):
                        # 将当前潜在表示作为模型输入
                        latent_model_input = latents
                        # 将当前时间步堆叠为张量
                        timestep = [t]

                        timestep = torch.stack(timestep)

                        # 将模型移动到当前 GPU
                        model.to(gpu)
                        # 进行有条件和无条件预测
                        noise_pred_cond = model(
                            latent_model_input,
                            t=timestep,
                            vace_context=z,
                            vace_context_scale=context_scale,
                            **arg_c)[0]
                        noise_pred_uncond = model(
                            latent_model_input,
                            t=timestep,
                            vace_context=z,
                            vace_context_scale=context_scale,
                            **arg_null)[0]

                        # 计算最终的噪声预测
                        noise_pred = noise_pred_uncond + guide_scale * (
                            noise_pred_cond - noise_pred_uncond)

                        # 使用调度器进行一步采样
                        temp_x0 = sample_scheduler.step(
                            noise_pred.unsqueeze(0),
                            t,
                            latents[0].unsqueeze(0),
                            return_dict=False,
                            generator=seed_g)[0]
                        latents = [temp_x0.squeeze(0)]

                    # 清空 CUDA 缓存
                    torch.cuda.empty_cache()
                    # 最终的潜在表示
                    x0 = latents

                    # 如果是主进程，则解码潜在表示生成视频
                    if rank == 0:
                        videos = self.decode_latent(
                            x0, input_ref_images, vae=vae)

                # 删除临时变量以释放内存
                del noise, latents
                del sample_scheduler
                if offload_model:
                    gc.collect()
                    torch.cuda.synchronize()
                # 如果分布式环境已初始化，则同步所有进程
                if dist.is_initialized():
                    dist.barrier()

                # 将生成的视频放入输出队列
                if rank == 0:
                    out_q.put(videos[0].cpu())

        except Exception as e:
            # 捕获异常并输出堆栈信息
            trace_info = traceback.format_exc()
            print(trace_info, flush=True)
            print(e, flush=True)

    def generate(self,
                 input_prompt,
                 input_frames,
                 input_masks,
                 input_ref_images,
                 size=(1280, 720),
                 frame_num=81,
                 context_scale=1.0,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        """
        使用多进程推理生成视频帧。

        参数:
            input_prompt (str): 用于生成内容的文本提示。
            input_frames (List[torch.Tensor]): 输入的视频帧列表，每个帧为一个张量。
            input_masks (List[torch.Tensor]): 输入的掩码列表，每个掩码为一个张量。
            input_ref_images (List[List[Optional[str]]]): 参考图像路径的嵌套列表，每个视频对应一个参考图像列表。
            size (Tuple[int, int], 可选): 控制视频分辨率，(宽度, 高度)，默认为 (1280, 720)。
            frame_num (int, 可选): 从视频中采样的帧数，默认为81。该数字应为4n+1。
            context_scale (float, 可选): 上下文缩放因子，影响生成过程中的上下文影响程度，默认为1.0。
            shift (float, 可选): 噪声调度移位参数，影响时间动态，默认为5.0。
            sample_solver (str, 可选): 用于采样视频的求解器，默认为 'unipc'。
            sampling_steps (int, 可选): 扩散采样的步数，默认为40。较高的值可以提高质量，但会减慢生成速度。
            guide_scale (float, 可选): 无分类器指导比例，控制提示的遵循程度与创造力之间的平衡，默认为5.0。
            n_prompt (str, 可选): 用于排除内容的负提示。如果未提供，则使用 `config.sample_neg_prompt`。
            seed (int, 可选): 用于噪声生成的随机种子。如果为-1，则使用随机种子，默认为-1。
            offload_model (bool, 可选): 如果为True，则在生成过程中将模型卸载到 CPU 以节省 VRAM，默认为True。

        返回:
            Any: 生成的结果，具体类型取决于实现。
        """
        # 将输入参数打包成一个元组
        input_data = (input_prompt, input_frames, input_masks, input_ref_images,
                      size, frame_num, context_scale, shift, sample_solver,
                      sampling_steps, guide_scale, n_prompt, seed,
                      offload_model)
        # 将输入数据放入每个输入队列中
        for in_q in self.in_q_list:
            in_q.put(input_data)
        # 从输出队列中获取结果
        value_output = self.out_q.get()

        return value_output

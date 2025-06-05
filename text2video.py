# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanT2V:
    """
    WanT2V 类实现了基于扩散模型的文本生成视频（T2V，Text-to-Video）生成器。

    该类集成了 T5 编码器、WanVAE 以及 WanModel 等组件，用于将文本提示转换为视频内容。
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
        初始化 WanT2V 模型组件。

        参数:
            config (EasyDict): 包含模型参数的对象，初始化自 config.py。
            checkpoint_dir (str): 模型检查点所在的目录路径。
            device_id (int, 可选): 目标 GPU 设备的 ID，默认为0。
            rank (int, 可选): 分布式训练中的进程排名，默认为0。
            t5_fsdp (bool, 可选): 是否启用 T5 模型的 FSDP 分片，默认为 False。
            dit_fsdp (bool, 可选): 是否启用 DiT 模型的 FSDP 分片，默认为 False。
            use_usp (bool, 可选): 是否启用 USP 分布式策略，默认为 False。
            t5_cpu (bool, 可选): 是否将 T5 模型放置在 CPU 上。仅在没有启用 t5_fsdp 时有效，默认为 False。
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config  # 设置配置参数
        self.rank = rank  # 设置进程排名
        self.t5_cpu = t5_cpu

        # 设置训练时间步数量
        self.num_train_timesteps = config.num_train_timesteps
        # 设置参数数据类型
        self.param_dtype = config.param_dtype

        # 定义分片函数
        shard_fn = partial(shard_model, device_id=device_id)
        # 初始化 T5 编码器模型
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,  # 设置文本长度
            dtype=config.t5_dtype,  # 设置 T5 模型的数据类型
            device=torch.device('cpu'),  
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),  # 设置检查点路径
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),  # 设置分词器路径
            shard_fn=shard_fn if t5_fsdp else None)

        # 设置 VAE 步幅
        self.vae_stride = config.vae_stride
        # 设置图像块大小
        self.patch_size = config.patch_size
        # 初始化 WanVAE 模型
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),  # 设置 VAE 检查点路径
            device=self.device)

        # 输出日志信息，指示正在创建 WanModel
        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)  # 设置为评估模式，关闭梯度计算

        if use_usp:
             # 导入获取序列并行世界大小的函数
            from xfuser.core.distributed import get_sequence_parallel_world_size

            # 导入 USP 相关的分布式前向传播函数
            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
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
            input_prompt (str): 用于内容生成的文本提示。
            size (Tuple[int, int], 可选): 控制视频分辨率，(宽度,高度)，默认为 (1280,720)。
            frame_num (int, 可选): 从视频中采样的帧数。数字应为 4n+1，默认为81。
            shift (float, 可选): 噪声调度偏移参数。影响时间动态，默认为5.0。
            sample_solver (str, 可选): 用于采样视频的求解方法，默认为 'unipc'。
            sampling_steps (int, 可选): 扩散采样的步数。值越高，质量越好，但生成速度越慢，默认为40。
            guide_scale (float, 可选): 无分类器指导比例。控制提示遵循与创造力的平衡，默认为5.0。
            n_prompt (str, 可选): 用于排除内容的负提示。如果未给出，则使用 `config.sample_neg_prompt`，默认为 ""。
            seed (int, 可选): 用于噪声生成的随机种子。如果-1，则使用随机种子，默认为-1。
            offload_model (bool, 可选): 如果为 True，则在生成过程中将模型卸载到 CPU 以节省 VRAM，默认为 True。

        返回:
            torch.Tensor: 生成的视频帧张量。维度为 (C, N, H, W)，其中:
                - C: 颜色通道（3 为 RGB）
                - N: 帧数（81）
                - H: 帧高度（来自 size）
                - W: 帧宽度（来自 size）
        """
        # 预处理
        F = frame_num  # 获取帧数
        # 计算目标形状
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        # 计算序列长度
        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            # 如果未提供负提示，则使用负提示样本
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)  # 设置随机种子
        seed_g = torch.Generator(device=self.device)  # 初始化随机数生成器
        seed_g.manual_seed(seed)  # 设置随机种子

        if not self.t5_cpu:
            # 将文本编码器模型移动到设备
            self.text_encoder.model.to(self.device)
            # 对输入提示进行编码
            context = self.text_encoder([input_prompt], self.device)
            # 对负提示进行编码
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                # 如果启用卸载模型，则将文本编码器模型移动回 CPU
                self.text_encoder.model.cpu()
        else:
            # 对输入提示进行编码，使用 CPU
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            # 对负提示进行编码，使用 CPU
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            # 将编码后的提示移动到设备
            context = [t.to(self.device) for t in context] 
            # 将编码后的负提示移动到设备
            context_null = [t.to(self.device) for t in context_null]

        # 生成噪声张量
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

        # 定义一个上下文管理器，用于在训练过程中不进行梯度同步
        @contextmanager
        def noop_no_sync():
            yield

        # 获取模型的 no_sync 方法，如果不存在，则使用 noop_no_sync
        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # 评估模式
        # 使用自动混合精度，不计算梯度，不进行梯度同步
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # 采样视频
            latents = noise  # 将噪声作为初始潜在空间表示

            # 设置上下文参数
            arg_c = {'context': context, 'seq_len': seq_len}
            # 设置负上下文参数
            arg_null = {'context': context_null, 'seq_len': seq_len}

            for _, t in enumerate(tqdm(timesteps)):
                # 输入潜在空间表示
                latent_model_input = latents
                # 当前时间步
                timestep = [t]

                # 堆叠时间步
                timestep = torch.stack(timestep)

                # 将模型移动到设备
                self.model.to(self.device)
                # 计算有条件噪声预测
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                # 计算无条件噪声预测
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]

                # 计算最终噪声预测
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)
                
                # 进行一步采样
                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                # 更新潜在空间表示
                latents = [temp_x0.squeeze(0)]

            # 最终的潜在空间表示
            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                # 对潜在空间表示进行解码，得到视频
                videos = self.vae.decode(x0)

        # 删除噪声和潜在空间表示
        del noise, latents
        # 删除调度器
        del sample_scheduler
        if offload_model:
            # 收集垃圾
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            # 同步分布式环境
            dist.barrier()

        # 返回生成的视频，如果当前进程不是0，则返回 None
        return videos[0] if self.rank == 0 else None

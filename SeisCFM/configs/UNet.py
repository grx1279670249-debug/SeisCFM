import math
import torch
import torch.nn.functional as F
from typing import Optional
import torch.nn as nn
from fontTools.misc.cython import returns
from taming.modules.diffusionmodules.model import Encoder, Decoder

from torchcfm.models.unet.unet import UNetModel

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:

    if timesteps.dim() == 0:
        timesteps = timesteps[None]
    half = dim // 2

    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class SeismicUNet(torch.nn.Module):
    def __init__(
            self,
            img_size=(129, 188),
            in_channels: int = 3,
            model_channels: int = 64,
            out_channels: int = 3,
            num_res_blocks: int = 2,
            channel_mult=(1, 2, 4, 8),
            attention_resolutions=(4, 8),  # 对应 DownSample 深度 4×、8× 时加注意力
            dropout: float = 0.0,
            n_fault_types: int = 5,
            **kwargs,
    ):
        super().__init__()
        self.H, self.W = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fault_types = n_fault_types

        # 1. 主干 UNet —— 设 use_scale_shift_norm=True 以启用 FiLM
        self.unet = UNetModel(
            image_size=max(self.H, self.W),  # <- 原来是 self.W
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=None,
            use_scale_shift_norm=True,
            **kwargs,
        )

        # 2. 条件嵌入 —— 连续元数据
        self.time_embed_dim = model_channels * 4
        meta_embed_dim = model_channels * 3
        self.meta_mlp = torch.nn.Sequential(
            torch.nn.Linear(3, meta_embed_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(meta_embed_dim, meta_embed_dim),
        )

        # 3. 条件嵌入 —— 断层类型
        self.fault_emb = torch.nn.Embedding(n_fault_types, model_channels)

        self.alpha_time = torch.nn.Parameter(torch.tensor(1.0))
        self.alpha_other = torch.nn.Parameter(torch.tensor(1.0))

    def forward(
            self,
            t: torch.Tensor,
            x: torch.Tensor,
            meta_cont: torch.Tensor,
            fault_type: torch.Tensor,
    ):

        fault_type = fault_type.squeeze(-1)

        emb_time = self.unet.time_embed(timestep_embedding(t, self.unet.model_channels))
        emb_meta = self.meta_mlp(meta_cont)
        emb_fault = self.fault_emb(fault_type.long()).to(emb_time.dtype)
        emd_meta_fault = torch.cat([emb_meta, emb_fault], dim=-1)

        cond_emb = self.alpha_time * emb_time + self.alpha_other * emd_meta_fault  # (N, time_embed_dim)

        hs = []
        h = x.type(self.unet.dtype)
        for module in self.unet.input_blocks:
            h = module(h, cond_emb)
            hs.append(h)
        h = self.unet.middle_block(h, cond_emb)
        for module in self.unet.output_blocks:
            skip = hs.pop()
            if skip.shape[-2:] != h.shape[-2:]:
                skip = F.interpolate(skip, size=h.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = module(h, cond_emb)
        h = h.type(x.dtype)
        out = self.unet.out(h)
        target_hw = x.shape[-2:]  # (H_in, W_in)
        if out.shape[-2:] != target_hw:
            out = F.interpolate(out, size=target_hw, mode="bilinear", align_corners=False)

        return out
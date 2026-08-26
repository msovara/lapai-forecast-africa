"""
lapai_inference/model.py

InceptionNeXt-style student CNN for ~1 degree global grids:
- Multi-branch depthwise convs: 3x3, 1x11 (zonal), 11x1 (meridional)
- Geocyclic padding on longitude
- Learnable geographic node features (8 channels)
- Pressure-level aggregation 13 -> 3 via 1x1 conv (Perceiver-style)

Tensor layout: NCHW with H=latitude, W=longitude (e.g. 181 x 360).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


def geocyclic_pad(x: torch.Tensor, pad_w: int) -> torch.Tensor:
    """Circular pad along width (longitude): [..., H, W]."""
    if pad_w <= 0:
        return x
    left = x[..., -pad_w:]
    right = x[..., :pad_w]
    return torch.cat([left, x, right], dim=-1)


class InceptionNeXtBlock(nn.Module):
    """Three parallel depthwise branches + 1x1 fusion (no spatial pooling)."""

    def __init__(
        self,
        channels: int,
        zonal_kernel: int = 11,
        merid_kernel: int = 11,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        assert channels % 4 == 0, "channels should be divisible by 4 for stable grouping"
        self.channels = channels
        self.zonal_kernel = zonal_kernel
        self.merid_kernel = merid_kernel
        pad_m = merid_kernel // 2

        self.dw_local = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.dw_zonal = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, zonal_kernel),
            padding=0,
            groups=channels,
            bias=False,
        )
        self.dw_merid = nn.Conv2d(
            channels,
            channels,
            kernel_size=(merid_kernel, 1),
            padding=(pad_m, 0),
            groups=channels,
            bias=False,
        )
        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels, eps=norm_eps)
        self.pw = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Circular longitude pad so Conv2d k=(1,K), pad=0 preserves width W when pad_w = (K-1)//2
        pad_w = (self.zonal_kernel - 1) // 2
        xz = geocyclic_pad(x, pad_w)
        zonal = self.dw_zonal(xz)

        local_b = self.dw_local(x)
        merid_b = self.dw_merid(x)

        y = torch.cat([local_b, zonal, merid_b], dim=1)
        y = self.pw(y)
        y = self.norm(y)
        y = self.act(y)
        return y + x


@dataclass
class LapAIStudentConfig:
    lat: int = 181
    lon: int = 360
    in_channels_raw: int = 65
    """Channels after optional explicit surface stack (before pressure aggregator)."""
    pressure_levels_in: int = 13
    pressure_levels_out: int = 3
    multilevel_groups: int = 5
    """Number of variables replicated across vertical (t,u,v,q,z style); each has pressure_levels_in planes."""
    geo_features: int = 8
    base_channels: int = 256
    stages: Tuple[int, int, int] = (2, 3, 3)
    zonal_kernel: int = 11
    merid_kernel: int = 11
    out_channels: int = 3
    """Intentional Cout=3 head order: tp, msl, 2t (partial-state MVP; no 3→65 decoder)."""


class LapAIStudentCNN(nn.Module):
    """
    CNN student with staged InceptionNeXt blocks.
    Returns dict with 'pred', optional 'feat_stage2', 'feat_stage3' for FitNets distillation.
    """

    def __init__(self, cfg: Optional[LapAIStudentConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or LapAIStudentConfig()
        c = self.cfg

        multilevel_ch = c.pressure_levels_in * c.multilevel_groups
        surf_ch = max(0, c.in_channels_raw - multilevel_ch)
        self.multilevel_ch = multilevel_ch
        self.surf_ch = surf_ch

        self.pressure_mix = nn.Conv2d(
            multilevel_ch, c.pressure_levels_out * c.multilevel_groups, kernel_size=1, bias=True
        )
        backbone_in = c.pressure_levels_out * c.multilevel_groups + surf_ch + c.geo_features

        self.geo_embed = nn.Parameter(torch.zeros(1, c.geo_features, c.lat, c.lon))

        self.stem = nn.Sequential(
            nn.Conv2d(backbone_in, c.base_channels, kernel_size=1, bias=False),
            nn.GELU(),
        )

        blocks: List[nn.Module] = []
        for n in c.stages:
            for _ in range(n):
                blocks.append(
                    InceptionNeXtBlock(
                        c.base_channels,
                        zonal_kernel=c.zonal_kernel,
                        merid_kernel=c.merid_kernel,
                    )
                )
        self.backbone = nn.ModuleList(blocks)
        self.stage_splits = list(c.stages)

        self.head = nn.Conv2d(c.base_channels, c.out_channels, kernel_size=1)

        nn.init.trunc_normal_(self.geo_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, C_in, Lat, Lon) mixed ERA5 / IC tensor matching cfg.in_channels_raw ordering.
        """
        cfg = self.cfg
        if self.multilevel_ch > 0:
            mv = x[:, : self.multilevel_ch]
            rest_start = self.multilevel_ch
            mv = self.pressure_mix(mv)
            tail = x[:, rest_start:, :, :]
            xb = torch.cat([mv, tail], dim=1)
        else:
            xb = x

        g = self.geo_embed.expand(xb.shape[0], -1, -1, -1)
        xb = torch.cat([xb, g], dim=1)
        h = self.stem(xb)

        feat_s2: Optional[torch.Tensor] = None
        feat_s3: Optional[torch.Tensor] = None
        idx = 0
        for si, n_blk in enumerate(self.stage_splits):
            for _ in range(n_blk):
                h = self.backbone[idx](h)
                idx += 1
            if si == 1:
                feat_s2 = h
            if si == 2:
                feat_s3 = h

        pred = self.head(h)
        return {"pred": pred, "feat_stage2": feat_s2, "feat_stage3": feat_s3}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def receptive_field_gradient_map(
    model: LapAIStudentCNN,
    x: torch.Tensor,
    out_lat_idx: int,
    out_lon_idx: int,
    out_chan: int = 0,
) -> torch.Tensor:
    """Empirical sensitivity map via |d y / d x| at one output grid point."""
    model.eval()
    xi = x.detach().clone().requires_grad_(True)
    out = model(xi)["pred"]
    scalar = out[0, out_chan, out_lat_idx, out_lon_idx]
    scalar.backward()
    assert xi.grad is not None
    return xi.grad.abs().mean(dim=1)


__all__ = [
    "LapAIStudentConfig",
    "LapAIStudentCNN",
    "InceptionNeXtBlock",
    "geocyclic_pad",
    "receptive_field_gradient_map",
]

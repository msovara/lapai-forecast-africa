"""Three-component distillation losses (A: lat-weighted MAE, B: feature MSE, C: spectral)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def latitude_broadcast_weights(lat_weights_1d: torch.Tensor, tensor_hw: torch.Tensor) -> torch.Tensor:
    """Expand cos(lat) weights (H,) to (1,1,H,1) matching NCHW."""
    w = lat_weights_1d.view(1, 1, -1, 1).to(device=tensor_hw.device, dtype=tensor_hw.dtype)
    return w


def channel_norm_view(stats_1d: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Broadcast (C,) mean/std to NCHW."""
    return stats_1d.view(1, -1, 1, 1).to(device=like.device, dtype=like.dtype)


def normalize_channels(
    x: torch.Tensor,
    mean: Optional[torch.Tensor],
    std: Optional[torch.Tensor],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-channel z-score using cache stats (C,). No-op if mean/std missing."""
    if mean is None or std is None:
        return x
    m = channel_norm_view(mean, x)
    s = channel_norm_view(std, x).clamp_min(eps)
    return (x - m) / s


def apply_soft_physical_constraints(
    pred: torch.Tensor, *, tp_mode: str = "relu"
) -> torch.Tensor:
    """
    Soft physicality on Cout=3 (tp, msl, 2t): non-negative tp only.
    No hard msl clamp in the forward path (zeros grads when pred << 1e5).

    tp_mode:
      - ``relu``: hard non-negativity (legacy; zeros grads for negative logits)
      - ``softplus``: scaled softplus ``softplus(1000·x)/1000`` ≈ relu on metre-scale
        tp while keeping a small gradient near zero (escapes all-dry MAE collapse)
      - ``none``: leave tp unconstrained (caller may clamp at eval)
    """
    if pred.shape[1] < 1:
        return pred
    raw = pred[:, 0:1]
    if tp_mode == "softplus":
        # Head is in physical metres (~1e-4); scale so softplus acts near zero.
        tp = F.softplus(raw * 1000.0) / 1000.0
    elif tp_mode == "none":
        tp = raw
    else:
        tp = F.relu(raw)
    rest = pred[:, 1:]
    return torch.cat([tp, rest], dim=1)


def loss_area_weighted_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    lat_weights: torch.Tensor,
    channel_weights: Optional[torch.Tensor] = None,
    spatial_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cosine-lat MAE; optional per-channel and/or (H,W)/(1,1,H,W) spatial weights."""
    w = latitude_broadcast_weights(lat_weights, pred)
    if spatial_weights is not None:
        sw = spatial_weights
        if sw.ndim == 2:
            sw = sw.view(1, 1, sw.shape[0], sw.shape[1])
        w = w * sw.to(device=w.device, dtype=w.dtype)
    err = w * (pred - target).abs()
    if channel_weights is not None:
        cw = channel_weights.view(1, -1, 1, 1).to(device=err.device, dtype=err.dtype)
        err = err * cw
    return err.mean()


def loss_precip_log1p_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    lat_weights: torch.Tensor,
    *,
    tp_index: int = 0,
    mm_scale: float = 1000.0,
    wet_threshold_m: float = 1.0e-4,
    wet_boost: float = 4.0,
    spatial_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Precipitation-aware term on channel ``tp_index`` (metres):

    - Compare ``log1p(tp * mm_scale)`` so light rain is not drowned by MAE in m.
    - Up-weight wet target pixels (``tp >= wet_threshold_m``) by ``wet_boost``.
    """
    p = pred[:, tp_index : tp_index + 1].clamp_min(0.0)
    t = target[:, tp_index : tp_index + 1].clamp_min(0.0)
    w = latitude_broadcast_weights(lat_weights, p)
    if spatial_weights is not None:
        sw = spatial_weights
        if sw.ndim == 2:
            sw = sw.view(1, 1, sw.shape[0], sw.shape[1])
        w = w * sw.to(device=w.device, dtype=w.dtype)
    wet = (t >= wet_threshold_m).to(dtype=p.dtype)
    w = w * (1.0 + wet_boost * wet)
    err = (torch.log1p(p * mm_scale) - torch.log1p(t * mm_scale)).abs()
    return (w * err).mean()


class FeatureDistillationHead(nn.Module):
    """Project student grid features to teacher dimension for FitNets-style loss."""

    def __init__(self, in_ch: int, teacher_ch: int, proj_dim: int = 128) -> None:
        super().__init__()
        self.proj_s = nn.Conv2d(in_ch, proj_dim, kernel_size=1)
        self.proj_t = nn.Conv2d(teacher_ch, proj_dim, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.proj_s.weight, nonlinearity="linear")
        nn.init.kaiming_normal_(self.proj_t.weight, nonlinearity="linear")

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        ps = self.proj_s(student_feat)
        pt = self.proj_t(teacher_feat)
        # Spatial instance-norm before MSE so raw teacher feature scale cannot dominate L_A.
        eps = 1e-6
        ps = (ps - ps.mean(dim=(-2, -1), keepdim=True)) / (ps.std(dim=(-2, -1), keepdim=True) + eps)
        pt = (pt - pt.mean(dim=(-2, -1), keepdim=True)) / (pt.std(dim=(-2, -1), keepdim=True) + eps)
        return F.mse_loss(ps, pt)


def loss_fft_power_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    k_max: Optional[int] = None,
    lat_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Spectral surrogate: MSE of log radial power from rFFT along longitude (fast CPU surrogate).
    Swap for torch-harmonics SHT when available for spherical fidelity.
    """
    spec_p = torch.fft.rfft(pred.float(), dim=-1)
    spec_t = torch.fft.rfft(target.float(), dim=-1)
    pow_p = (spec_p.real ** 2 + spec_p.imag ** 2).clamp(min=1e-12).log()
    pow_t = (spec_t.real ** 2 + spec_t.imag ** 2).clamp(min=1e-12).log()
    err = (pow_p - pow_t) ** 2
    if k_max is not None:
        err = err[..., :k_max]
    if lat_weights is not None:
        # Keep (1,1,H,1) so weights broadcast over N,C and rFFT freq dim.
        w = latitude_broadcast_weights(lat_weights, pred)
        return (err * w).mean()
    return err.mean()


def combined_distillation_loss(
    pred: torch.Tensor,
    era5: torch.Tensor,
    teacher_pred: Optional[torch.Tensor],
    feat_s2: Optional[torch.Tensor],
    feat_s3: Optional[torch.Tensor],
    feat_t10: Optional[torch.Tensor],
    feat_t14: Optional[torch.Tensor],
    lat_weights: torch.Tensor,
    heads: Tuple[FeatureDistillationHead, FeatureDistillationHead],
    *,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.3,
    use_feature_loss: bool = True,
    use_spectral_loss: bool = True,
    spectral_k_max: Optional[int] = 40,
    target_mean: Optional[torch.Tensor] = None,
    target_std: Optional[torch.Tensor] = None,
    channel_weights: Optional[torch.Tensor] = None,
    spatial_weights: Optional[torch.Tensor] = None,
    africa_mix: float = 0.0,
    africa_spatial_weights: Optional[torch.Tensor] = None,
    precip_log1p_weight: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    pred_n = normalize_channels(pred, target_mean, target_std)
    era5_n = normalize_channels(era5, target_mean, target_std)
    la_g = loss_area_weighted_mae(
        pred_n, era5_n, lat_weights, channel_weights=channel_weights, spatial_weights=spatial_weights
    )
    if africa_mix > 0.0 and africa_spatial_weights is not None:
        la_a = loss_area_weighted_mae(
            pred_n,
            era5_n,
            lat_weights,
            channel_weights=channel_weights,
            spatial_weights=africa_spatial_weights,
        )
        la = (1.0 - africa_mix) * la_g + africa_mix * la_a
    else:
        la = la_g
    total = alpha * la
    parts = {"L_A": la.detach()}

    if precip_log1p_weight > 0.0:
        # Physical units (metres), not channel-normalized — log1p needs real scale.
        sw = africa_spatial_weights if (africa_mix > 0.0 and africa_spatial_weights is not None) else spatial_weights
        # Domain-mix precip term the same way as L_A when Africa mix is on.
        if africa_mix > 0.0 and africa_spatial_weights is not None:
            lp_g = loss_precip_log1p_mae(pred, era5, lat_weights, spatial_weights=None)
            lp_a = loss_precip_log1p_mae(
                pred, era5, lat_weights, spatial_weights=africa_spatial_weights
            )
            lp = (1.0 - africa_mix) * lp_g + africa_mix * lp_a
        else:
            lp = loss_precip_log1p_mae(pred, era5, lat_weights, spatial_weights=sw)
        total = total + precip_log1p_weight * lp
        parts["L_tp"] = lp.detach()
    else:
        parts["L_tp"] = torch.tensor(0.0, device=pred.device)

    if use_feature_loss and feat_s2 is not None and feat_s3 is not None and feat_t10 is not None and feat_t14 is not None:
        lb = heads[0](feat_s2, feat_t10) + heads[1](feat_s3, feat_t14)
        total = total + beta * lb
        parts["L_B"] = lb.detach()
    else:
        parts["L_B"] = torch.tensor(0.0, device=pred.device)

    if use_spectral_loss and teacher_pred is not None:
        teacher_n = normalize_channels(teacher_pred, target_mean, target_std)
        lc = loss_fft_power_mse(pred_n, teacher_n, k_max=spectral_k_max, lat_weights=lat_weights)
        total = total + gamma * lc
        parts["L_C"] = lc.detach()
    else:
        parts["L_C"] = torch.tensor(0.0, device=pred.device)

    parts["total"] = total.detach()
    return total, parts

"""Africa bounding masks and tensor helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

# Aligned with graphcast-africa / config/domains.yaml (lon east to 70E).
AFRICA_LAT: Tuple[float, float] = (-40.0, 40.0)
AFRICA_LON: Tuple[float, float] = (-20.0, 70.0)


def lat_lon_to_indices(
    lat: np.ndarray,
    lon: np.ndarray,
    lat_bounds: Tuple[float, float],
    lon_bounds: Tuple[float, float],
) -> Tuple[slice, slice]:
    lat_mask = (lat >= lat_bounds[0]) & (lat <= lat_bounds[1])
    lon_mask = (lon >= lon_bounds[0]) & (lon <= lon_bounds[1])
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise ValueError("Empty Africa slice; check lat/lon coordinates")
    return slice(int(lat_idx.min()), int(lat_idx.max()) + 1), slice(int(lon_idx.min()), int(lon_idx.max()) + 1)


def africa_crop(tensor_nchw: torch.Tensor, lat_slice: slice, lon_slice: slice) -> torch.Tensor:
    return tensor_nchw[..., lat_slice, lon_slice]


def area_rmse(pred: torch.Tensor, target: torch.Tensor, lat_weights: torch.Tensor) -> torch.Tensor:
    w = lat_weights.view(1, 1, -1, 1).to(pred)
    num = (w * (pred - target) ** 2).sum()
    den = w.expand_as(pred).sum()
    return torch.sqrt(num / den)


def anomaly_correlation_coefficient(pred_anom: torch.Tensor, ref_anom: torch.Tensor, lat_weights: torch.Tensor) -> torch.Tensor:
    w = lat_weights.view(1, 1, -1, 1).to(pred_anom)
    num = (w * pred_anom * ref_anom).sum()
    den = torch.sqrt((w * pred_anom**2).sum() * (w * ref_anom**2).sum()).clamp(min=1e-12)
    return num / den

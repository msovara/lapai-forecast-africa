"""Africa bounding masks and tensor helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

# Original Track A / Phase 0 Africa box (config/domains.yaml, lon east to 55E).
AFRICA_LAT: Tuple[float, float] = (-40.0, 40.0)
AFRICA_LON: Tuple[float, float] = (-20.0, 55.0)


def lon_to_180(lon: np.ndarray) -> np.ndarray:
    """Map longitudes to [-180, 180) for box tests (tolerant of 0..360 grids)."""
    return ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0


def africa_hw_mask(
    lat: np.ndarray,
    lon: np.ndarray,
    lat_bounds: Tuple[float, float] = AFRICA_LAT,
    lon_bounds: Tuple[float, float] = AFRICA_LON,
) -> np.ndarray:
    """(H,W) float32 mask for Africa box; handles lon in 0..360 or −180..180."""
    lat = np.asarray(lat, dtype=np.float64).reshape(-1)
    lon180 = lon_to_180(np.asarray(lon, dtype=np.float64).reshape(-1))
    # Build (H,W) via broadcasting — do NOT use in-place &= on (H,1); that
    # refuses to expand to (H,W) (ValueError: non-broadcastable output operand).
    lat_ok = (lat[:, None] >= lat_bounds[0]) & (lat[:, None] <= lat_bounds[1])
    lon_ok = (lon180[None, :] >= lon_bounds[0]) & (lon180[None, :] <= lon_bounds[1])
    return (lat_ok & lon_ok).astype(np.float32)


def lat_lon_to_indices(
    lat: np.ndarray,
    lon: np.ndarray,
    lat_bounds: Tuple[float, float],
    lon_bounds: Tuple[float, float],
) -> Tuple[slice, slice]:
    """Axis-aligned crop slices.

    For lon boxes that cross the antimeridian / date line in a 0..360 mesh
    (e.g. Africa −20..55), prefer ``africa_hw_mask`` — a single ``slice`` cannot
    represent the wrap. This helper keeps the contiguous lon≥0 part for legacy callers.
    """
    lat_mask = (lat >= lat_bounds[0]) & (lat <= lat_bounds[1])
    lon180 = lon_to_180(lon)
    lon_mask = (lon180 >= lon_bounds[0]) & (lon180 <= lon_bounds[1])
    # If wrap leaves a hole, fall back to contiguous lon in [0, lon_max] only.
    lon_idx = np.where(lon_mask)[0]
    lat_idx = np.where(lat_mask)[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise ValueError("Empty Africa slice; check lat/lon coordinates")
    # Contiguous slice only if indices are contiguous; else use full lon span of hits
    # (may include non-Africa longitudes when wrap exists — use africa_hw_mask instead).
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

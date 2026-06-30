"""Offline unstructured->unstructured regrid bridge (N320 ICs -> O96 model grid).

Used when forecasting a coarsened Track A student whose data grid differs from the
N320 grid produced by the CDS initial-condition pipeline. Avoids earthkit-regrid
matrix downloads (Lengau compute nodes have no outbound internet): builds a
k-nearest-neighbour inverse-distance interpolation on the unit sphere from the two
grids' lat/lon coordinates alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _latlon_to_xyz(latlon_deg: np.ndarray) -> np.ndarray:
    lat = np.radians(latlon_deg[:, 0])
    lon = np.radians(latlon_deg[:, 1])
    cos_lat = np.cos(lat)
    return np.stack([cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)], axis=1)


@dataclass
class GridBridge:
    """Precomputed k-NN IDW weights mapping source grid values to a target grid."""

    idx: np.ndarray  # (n_tgt, k) source indices
    weights: np.ndarray  # (n_tgt, k) normalized weights
    n_src: int
    n_tgt: int

    def apply(self, field: np.ndarray) -> np.ndarray:
        """Regrid a field whose last axis indexes source grid points.

        Supports (n_src,) and (..., n_src) arrays (e.g. (time, n_src)).
        """
        arr = np.asarray(field)
        if arr.shape[-1] != self.n_src:
            raise ValueError(f"last axis {arr.shape[-1]} != n_src {self.n_src}")
        gathered = arr[..., self.idx]  # (..., n_tgt, k)
        out = np.einsum("...tk,tk->...t", gathered, self.weights)
        return out.astype(arr.dtype, copy=False)


def build_grid_bridge(
    src_latlon_deg: np.ndarray,
    tgt_latlon_deg: np.ndarray,
    *,
    k: int = 4,
    power: float = 2.0,
    eps: float = 1e-12,
) -> GridBridge:
    """Build a k-NN inverse-distance bridge from source to target lat/lon grids."""
    from scipy.spatial import cKDTree

    src = np.asarray(src_latlon_deg, dtype=np.float64).reshape(-1, 2)
    tgt = np.asarray(tgt_latlon_deg, dtype=np.float64).reshape(-1, 2)
    k = min(k, src.shape[0])

    tree = cKDTree(_latlon_to_xyz(src))
    dist, idx = tree.query(_latlon_to_xyz(tgt), k=k)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    w = 1.0 / np.power(np.maximum(dist, eps), power)
    exact = dist < eps
    if exact.any():
        w[exact.any(axis=1)] = 0.0
        rows = np.where(exact.any(axis=1))[0]
        for r in rows:
            w[r] = 0.0
            w[r, np.argmin(dist[r])] = 1.0
    w = w / w.sum(axis=1, keepdims=True)

    return GridBridge(idx=idx, weights=w, n_src=src.shape[0], n_tgt=tgt.shape[0])

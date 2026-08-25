"""
Training tensor contract and cache helpers (Zarr + HDF5).

Zarr/HDF5 group ``lapai_cache`` keys:
  state_in — student input (T, Cin, H, W)
  era5_target — MAE target (T, Cout, H, W)
  teacher_pred — teacher outputs (T, Cout, H, W)
  teacher_feat_L10, teacher_feat_L14 — cached processor tensors (T, D*, H, W)
  init_id, lead_hours — optional metadata along T

PyTorch layout: NCHW (H latitude south-north, W longitude).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

SCHEMA_VERSION = "lapai_cache/v1"

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore

try:
    import zarr
except ImportError:
    zarr = None  # type: ignore


@dataclass
class CachePartitionSpec:
    time_chunk: int = 32
    lat_lon_chunks: tuple = (181, 360)


def validate_attrs(attrs: Mapping[str, Any]) -> None:
    ver = attrs.get("lapai_schema_version")
    if ver != SCHEMA_VERSION:
        raise ValueError(f"Expected lapai_schema_version={SCHEMA_VERSION!r}, got {ver!r}")


def lat_lon_mesh(lat_count: int = 181, lon_count: int = 360) -> tuple[np.ndarray, np.ndarray]:
    lat = np.linspace(-90.0, 90.0, lat_count, dtype=np.float64)
    lon = np.linspace(0.0, 360.0 - 360.0 / lon_count, lon_count, dtype=np.float64)
    return lat, lon


def cosine_latitude_weights(lat: np.ndarray) -> np.ndarray:
    lat_rad = np.deg2rad(lat).reshape(-1, 1)
    return np.cos(lat_rad).astype(np.float32)


def write_shard_hdf5(path: Path, arrays: dict, attrs: Optional[dict] = None) -> None:
    if h5py is None:
        raise ImportError("h5py required for HDF5 cache writes")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.create_group("lapai_cache")
        for k, v in arrays.items():
            g.create_dataset(k, data=v, compression="gzip", compression_opts=4)
        if attrs:
            for ak, av in attrs.items():
                g.attrs[ak] = av


def read_shard_hdf5(path: Path) -> dict:
    if h5py is None:
        raise ImportError("h5py required for HDF5 cache reads")
    out: Dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        g = f["lapai_cache"]
        for k in g.keys():
            out[str(k)] = np.asarray(g[k])
        out["_attrs"] = dict(g.attrs)
    return out


def open_zarr_group(path: Path, mode: str = "a"):
    if zarr is None:
        raise ImportError("zarr package required")
    path.parent.mkdir(parents=True, exist_ok=True)
    return zarr.open_group(str(path), mode=mode)


def init_zarr_store(store_path: Path, attrs: Mapping[str, Any], overwrite: bool = False) -> Any:
    if overwrite and store_path.exists():
        import shutil

        shutil.rmtree(store_path)
    root = open_zarr_group(store_path, mode="a")
    merged = dict(attrs)
    merged["lapai_schema_version"] = SCHEMA_VERSION
    root.attrs.update(merged)
    return root


def zarr_append(root_group: Any, name: str, arr: np.ndarray, start_idx: int) -> None:
    if zarr is None:
        raise ImportError("zarr package required")
    if name not in root_group:
        chunks = (min(32, arr.shape[0]),) + tuple(min(x, 64) for x in arr.shape[1:])
        compress = zarr.Blosc(cname="zstd", clevel=5)
        root_group.create_dataset(
            name,
            shape=(0,) + arr.shape[1:],
            chunks=chunks,
            dtype=arr.dtype,
            compressor=compress,
        )
    ds = root_group[name]
    new_len = start_idx + arr.shape[0]
    ds.resize((new_len,) + ds.shape[1:])
    ds[start_idx:new_len] = arr

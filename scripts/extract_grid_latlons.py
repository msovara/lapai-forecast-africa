#!/usr/bin/env python3
"""Extract a model's data-grid lat/lon (degrees) from an anemoi inference checkpoint.

Anemoi stores ``model.node_attributes.latlons_data`` as columns
``[sin(lat), sin(lon), cos(lat), cos(lon)]``. This decodes them to a portable
(N, 2) ``[lat_deg, lon_deg]`` array so the offline N320->O96 IC bridge does not
need to reload a multi-GB checkpoint at forecast time.

Example:
  python scripts/extract_grid_latlons.py \
    --ckpt models/teacher_n320_gt6/inference.ckpt \
    --out data/processed/lapai/n320_latlons.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def decode_latlons(latlons_data: np.ndarray) -> np.ndarray:
    """Columns [sin_lat, sin_lon, cos_lat, cos_lon] -> (N, 2) [lat_deg, lon_deg]."""
    arr = np.asarray(latlons_data, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"expected (N, 4) latlons_data, got {arr.shape}")
    sin_lat, sin_lon, cos_lat, cos_lon = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    lat = np.degrees(np.arctan2(sin_lat, cos_lat))
    lon = np.degrees(np.arctan2(sin_lon, cos_lon)) % 360.0
    return np.stack([lat, lon], axis=1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = obj.state_dict() if hasattr(obj, "state_dict") else obj
    key = next((k for k in sd if k.endswith("node_attributes.latlons_data")), None)
    if key is None:
        raise SystemExit("node_attributes.latlons_data not found in checkpoint")
    latlons = decode_latlons(sd[key].cpu().numpy())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, latlons.astype(np.float32))
    print(f"{args.out}: {latlons.shape} lat[{latlons[:,0].min():.3f},{latlons[:,0].max():.3f}] "
          f"lon[{latlons[:,1].min():.3f},{latlons[:,1].max():.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

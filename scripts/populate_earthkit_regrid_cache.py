#!/usr/bin/env python3
"""Download earthkit-regrid matrix DB once (needs internet). Sync cache to Lengau GPU jobs.

Triggers 0.25 deg -> N320 interpolation so regrid matrices land in the repo cache
used by run_phase0_forecast.py (default: data/cache/earthkit/regrid).

Run on laptop or Lengau login node, then GPU jobs work offline.

Example:
  python scripts/populate_earthkit_regrid_cache.py
  python scripts/populate_earthkit_regrid_cache.py --cache-dir data/cache/earthkit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.run_n320_gt6_opendata_forecast import (  # noqa: E402
    _configure_earthkit_caches,
    _patch_earthkit_regrid_windows_urls,
)
from utils.cds_ic import _regrid_to_n320  # noqa: E402


def _regrid_n320_to_latlon025(values: np.ndarray) -> np.ndarray:
    import earthkit.regrid as ekr

    return np.asarray(
        ekr.interpolate(values, {"grid": "N320"}, {"grid": (0.25, 0.25)}),
        dtype=np.float64,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Populate earthkit-regrid cache for offline Lengau")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=_REPO / "data" / "cache" / "earthkit",
        help="Root passed to _configure_earthkit_caches (contains regrid/ subdir)",
    )
    args = p.parse_args()

    _patch_earthkit_regrid_windows_urls()
    _configure_earthkit_caches(args.cache_dir.resolve())
    latlon = np.zeros((721, 1440), dtype=np.float32)
    n320 = _regrid_to_n320(latlon)
    print(f"  0.25 -> N320: shape={n320.shape}")
    back = _regrid_n320_to_latlon025(n320.astype(np.float64))
    print(f"  N320 -> 0.25: shape={back.shape}")
    print(f"OK: regrid cache ready under {args.cache_dir / 'regrid'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

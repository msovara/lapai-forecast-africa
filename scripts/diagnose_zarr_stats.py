#!/usr/bin/env python3
"""Diagnose NaN-loss risk from an anemoi Zarr's normalization statistics.

Reads the precomputed stat arrays (mean/stdev/minimum/maximum) and variable names
DIRECTLY via zarr/numpy -- no anemoi import (avoids offline registry/network hangs).

Flags two failure modes that produce intermittent NaN training loss:
  1. Degenerate stdev (== 0 or non-finite): standardization divides by ~0 -> inf/NaN.
  2. fp16 overflow risk: |(extreme - mean) / stdev| approaching fp16 max (65504).

Usage:
  python scripts/diagnose_zarr_stats.py [/path/to/dataset.zarr]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import zarr

DEFAULT = "/home/msovara/repos/lapai-forecast/data/processed/lapai/era5_n96_smoke.zarr"
FP16_MAX = 65504.0


def _variable_names(z, nvars: int) -> list[str]:
    attrs = dict(z.attrs)
    for key in ("variables", "variable", "name_to_index"):
        val = attrs.get(key)
        if isinstance(val, list) and len(val) == nvars:
            return [str(v) for v in val]
        if isinstance(val, dict) and len(val) == nvars:
            return [k for k, _ in sorted(val.items(), key=lambda kv: kv[1])]
    return [f"var_{i}" for i in range(nvars)]


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    z = zarr.open(path, mode="r")

    mean = np.asarray(z["mean"][:], dtype=np.float64)
    std = np.asarray(z["stdev"][:], dtype=np.float64)
    mn = np.asarray(z["minimum"][:], dtype=np.float64)
    mx = np.asarray(z["maximum"][:], dtype=np.float64)
    nvars = mean.shape[0]
    variables = _variable_names(z, nvars)

    print(f"dataset: {path}")
    print(f"variables: {nvars}")
    try:
        dates = z["dates"][:]
        print(f"dates: {dates[0]} .. {dates[-1]}  (n={len(dates)})")
    except Exception:
        pass
    try:
        print(f"data shape: {z['data'].shape}  dtype: {z['data'].dtype}")
    except Exception:
        pass
    print()

    bad_std, nonfinite, rows = [], [], []
    for i, name in enumerate(variables):
        m, s, lo, hi = mean[i], std[i], mn[i], mx[i]
        if not np.isfinite([m, s, lo, hi]).all():
            nonfinite.append(name)
        if not np.isfinite(s) or s <= 0:
            bad_std.append((name, s))
            norm = np.inf
        else:
            norm = max(abs((hi - m) / s), abs((lo - m) / s))
        rows.append((name, m, s, lo, hi, norm))

    print("=== degenerate stdev (stdev <= 0 or non-finite) ===")
    print("  " + ("\n  ".join(f"{n:14s} stdev={s}" for n, s in bad_std) if bad_std else "none"))
    print()

    print("=== non-finite stats (mean/std/min/max) ===")
    print("  " + (", ".join(nonfinite) if nonfinite else "none"))
    print()

    print("=== top 25 fp16-overflow risk (|normalized extreme|, fp16 max=65504) ===")
    rows.sort(key=lambda r: (np.inf if not np.isfinite(r[5]) else r[5]), reverse=True)
    print(f"  {'variable':14s} {'mean':>12s} {'stdev':>12s} {'min':>12s} {'max':>12s} {'norm':>11s}")
    for name, m, s, lo, hi, norm in rows[:25]:
        flag = "  <-- RISK" if (not np.isfinite(norm)) or norm > 1000 else ""
        nrm = "inf" if not np.isfinite(norm) else f"{norm:11.1f}"
        print(f"  {name:14s} {m:12.4g} {s:12.4g} {lo:12.4g} {hi:12.4g} {nrm:>11}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

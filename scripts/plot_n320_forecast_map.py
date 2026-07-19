#!/usr/bin/env python3
"""Plot a field from a LapAI N320 forecast NetCDF over the Africa eval domain."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.n320_forecast_io import (  # noqa: E402
    EVAL_ALIASES,
    default_plot_path,
    plot_temperature_celsius,
    plot_unstructured_map,
)


def main() -> int:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    p = argparse.ArgumentParser(description="Plot N320 forecast NetCDF field over Africa")
    p.add_argument("netcdf", type=Path, help="Forecast NetCDF from run_n320_gt6_opendata_forecast.py")
    p.add_argument("--var", default="2t", help="Variable name in NetCDF (default: 2t)")
    p.add_argument("--time-index", type=int, default=-1, help="Time index to plot (default: last step)")
    p.add_argument("--domain", default="africa", help="Domain from configs/eval.yaml")
    p.add_argument("--output", type=Path, default=None, help="PNG path (default: alongside NetCDF)")
    p.add_argument(
        "--no-mask-ocean",
        action="store_true",
        help="Show temperature over ocean (default: land-only with ocean masked)",
    )
    args = p.parse_args()

    from netCDF4 import Dataset

    nc_path = args.netcdf.resolve()
    if not nc_path.is_file():
        raise SystemExit(f"Missing NetCDF: {nc_path}")

    with Dataset(nc_path) as nc:
        if args.var not in nc.variables:
            available = [k for k in nc.variables if k not in ("time", "latitude", "longitude")]
            raise SystemExit(f"Variable `{args.var}` not in file. Available: {sorted(available)}")
        lat = np.asarray(nc.variables["latitude"][:])
        lon = np.asarray(nc.variables["longitude"][:])
        n_time = nc.dimensions["time"].size
        idx = args.time_index if args.time_index >= 0 else n_time + args.time_index
        if idx < 0 or idx >= n_time:
            raise SystemExit(f"time-index {args.time_index} out of range (n_time={n_time})")
        values = np.asarray(nc.variables[args.var][idx])
        lsm = np.asarray(nc.variables["lsm"][:]) if "lsm" in nc.variables else None
        ref = getattr(nc, "reference_time", "forecast")
        step_h = int(nc.variables["time"][idx]) // 3600

    out = args.output or default_plot_path(nc_path, args.var, args.domain)
    eval_name = EVAL_ALIASES.get(args.var, args.var)
    title = f"n320_gt6 {eval_name} (+{step_h}h from {ref})"

    mask_ocean = not args.no_mask_ocean

    if args.var == "2t":
        plot_temperature_celsius(
            lat,
            lon,
            values,
            title=title,
            output=out,
            domain=args.domain,
            lsm=lsm,
            mask_ocean=mask_ocean,
        )
    else:
        plot_unstructured_map(
            lat,
            lon,
            values,
            title=title,
            output=out,
            domain=args.domain,
            lsm=lsm,
            mask_ocean=mask_ocean,
        )

    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

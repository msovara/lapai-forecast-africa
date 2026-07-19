#!/usr/bin/env python3
"""Time basic reads from the JRA-3Q Anemoi zarr store (pre-training diagnostic)."""
import sys
import time
from pathlib import Path

import xarray as xr


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <zarr_path>", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"ERROR: dataset not found: {path}", file=sys.stderr)
        return 1

    print(f"Opening: {path}")
    t0 = time.perf_counter()
    try:
        ds = xr.open_zarr(path, consolidated=True)
    except ValueError as exc:
        if ".zmetadata" not in str(exc):
            raise
        print("NOTE: no consolidated .zmetadata; opening with consolidated=False")
        ds = xr.open_zarr(path, consolidated=False)
    t_open = time.perf_counter() - t0
    print(f"open_zarr: {t_open:.2f}s dims={dict(ds.dims)} vars={len(ds.data_vars)}")

    if "time" not in ds.dims:
        print("WARNING: no time dimension")
        return 0

    t1 = time.perf_counter()
    _ = ds.isel(time=0).load()
    t_first = time.perf_counter() - t1
    print(f"isel(time=0).load(): {t_first:.2f}s")

    if ds.sizes["time"] > 1:
        t2 = time.perf_counter()
        _ = ds.isel(time=1).load()
        t_second = time.perf_counter() - t2
        print(f"isel(time=1).load(): {t_second:.2f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

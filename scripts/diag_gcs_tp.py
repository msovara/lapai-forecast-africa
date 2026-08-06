#!/usr/bin/env python3
"""Diagnose GCS total_precipitation Zarr: why open_era5_at_valid_time returns all-NaN."""
from __future__ import annotations

import numpy as np
import xarray as xr

from evaluation.gcs_era5_truth import (
    GCS_OPTS,
    era5_gcs_path,
    open_era5_at_valid_time,
    pick_era5_var,
)


def main() -> None:
    path = era5_gcs_path("tp", 2023)
    print("path:", path)
    ds = xr.open_zarr(path, storage_options=GCS_OPTS, consolidated=True)
    print("sizes:", {k: int(v) for k, v in ds.sizes.items()})
    print("data_vars:", list(ds.data_vars))
    print("coords:", list(ds.coords))
    var = pick_era5_var(ds)
    da = ds[var]
    print("picked:", var, "dims:", da.dims)
    if "step" in da.dims:
        print("step values:", da["step"].values)
    if "time" in da.dims:
        print("time[0]:", da.time.values[0], "time[-1]:", da.time.values[-1])

    # Finite fraction for a few raw slices (no Africa subset)
    times = da.time.values.astype("datetime64[ns]")
    targets = [
        np.datetime64("2023-01-01T00:00:00"),
        np.datetime64("2023-01-02T00:00:00"),
        np.datetime64("2023-01-02T06:00:00"),
    ]
    for target in targets:
        ti = int(np.argmin(np.abs(times - target)))
        print(f"\n--- nearest time to {target}: idx={ti} actual={times[ti]} ---")
        sl = da.isel(time=ti)
        if "step" in sl.dims:
            for si, step in enumerate(sl["step"].values):
                v = np.asarray(sl.isel(step=si).values)
                fin = float(np.isfinite(v).mean())
                print(f"  step={step}: finite={fin:.4f}", end="")
                if fin > 0:
                    print(f" min={float(np.nanmin(v)):.6g} max={float(np.nanmax(v)):.6g}")
                else:
                    print(" (all nan)")
        else:
            v = np.asarray(sl.values)
            fin = float(np.isfinite(v).mean())
            print(f"  finite={fin:.4f}")

    print("\n=== via open_era5_at_valid_time (Africa subset) ===")
    for target in targets:
        try:
            truth = open_era5_at_valid_time("tp", target)
            v = np.asarray(truth.values)
            fin = float(np.isfinite(v).mean())
            print(f"{target}: shape={truth.shape} finite={fin:.4f}")
        except Exception as exc:
            print(f"{target}: ERROR {exc}")

    print("\n=== t2m control at 2023-01-02T00 ===")
    t2 = open_era5_at_valid_time("t2m", np.datetime64("2023-01-02T00:00:00"))
    print("t2m finite", float(np.isfinite(t2.values).mean()))


if __name__ == "__main__":
    main()

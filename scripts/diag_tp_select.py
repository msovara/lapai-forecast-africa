#!/usr/bin/env python3
"""Probe which (time, step) _select_tp picks vs finite data."""
from __future__ import annotations

import numpy as np
import zarr
import gcsfs

from evaluation.gcs_era5_truth import (
    GCS_OPTS,
    _select_tp_at_valid_time,
    era5_gcs_path,
    open_era5_at_valid_time,
    pick_era5_var,
)
import xarray as xr


def main() -> None:
    path = era5_gcs_path("tp", 2023)
    ds = xr.open_zarr(path, storage_options=GCS_OPTS, consolidated=True)
    var = pick_era5_var(ds)
    da = ds[var]
    for drop in ("number", "surface", "expver"):
        if drop in da.dims and da.sizes[drop] == 1:
            da = da.isel({drop: 0})
    print("time[0]", da.time.values[0], "step", da.step.values)
    if "valid_time" in ds:
        vt = ds["valid_time"]
        print("valid_time dims", vt.dims, "sample[0,:3]", vt.values[0, :3])

    target = np.datetime64("2023-01-02T00:00:00")
    sel = _select_tp_at_valid_time(da, target)
    print("selected shape", sel.shape, "coords", {k: sel.coords[k].values for k in sel.coords if k in ("time", "step", "valid_time")})
    # load tiny corner to check finite without Africa
    tiny = np.asarray(sel.isel(latitude=slice(200, 210), longitude=slice(0, 10)).values)
    print("tiny finite", float(np.isfinite(tiny).mean()), "min", np.nanmin(tiny) if np.isfinite(tiny).any() else None)

    # manual: find via valid_time
    if "valid_time" in ds.coords or "valid_time" in ds:
        vt = np.asarray(ds["valid_time"].values)
        # may already be datetime64
        if np.issubdtype(vt.dtype, np.datetime64):
            vt_dt = vt.astype("datetime64[ns]")
        else:
            vt_dt = (vt.astype("float64") * 1e9).astype("datetime64[ns]")
        err = np.abs(vt_dt.astype("datetime64[ns]").astype("int64") - np.datetime64(target, "ns").astype("int64"))
        # among near matches (<1h), prefer finite
        near = np.argwhere(err <= int(3.6e12))  # 1h in ns
        print("near matches within 1h:", len(near))
        for ti, si in near[:20]:
            print(f"  ti={ti} si={si} vt={vt_dt[ti,si]} err_h={err[ti,si]/3.6e12:.3f}")

    print("\nopen_era5_at_valid_time finite frac...")
    truth = open_era5_at_valid_time("tp", target)
    v = np.asarray(truth.values)
    print("shape", v.shape, "finite", float(np.isfinite(v).mean()))


if __name__ == "__main__":
    main()

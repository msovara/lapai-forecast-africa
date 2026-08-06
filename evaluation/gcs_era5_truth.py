"""Load Mvula team ERA5 Zarr slices from gs://code4earth/era5 for Phase 0 scoring."""

from __future__ import annotations

import numpy as np
import xarray as xr

GCS_OPTS = {"token": "google_default"}

# scorecard name -> GCS folder stem (without year suffix)
ERA5_GCS_STEMS = {
    "t2m": "2m_temperature",
    "tp": "total_precipitation",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
}

AFRICA_LAT = slice(-40, 40)
AFRICA_LON = slice(-20, 70)


def era5_gcs_path(var: str, year: int, bucket: str = "gs://code4earth/era5") -> str:
    stem = ERA5_GCS_STEMS[var]
    return f"{bucket}/{stem}_{year}.zarr"


def pick_era5_var(ds: xr.Dataset) -> str:
    for name in ("t2m", "tp", "u10", "v10", "u", "v"):
        if name in ds.data_vars:
            return name
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars))
    raise KeyError(f"No known ERA5 variable in {list(ds.data_vars)}")


def valid_time_index(ds: xr.Dataset, when: np.datetime64) -> int:
    if "valid_time" in ds:
        times = np.asarray(ds["valid_time"].values, dtype="datetime64[ns]")
    else:
        times = np.asarray(ds["time"].values, dtype="datetime64[ns]")
    target = np.datetime64(when, "ns")
    return int(np.argmin(np.abs(times - target)))


def normalize_lon(da: xr.DataArray) -> xr.DataArray:
    if float(da.longitude.max()) > 300:
        lon = ((da.longitude + 180) % 360) - 180
        da = da.assign_coords(longitude=lon).sortby("longitude")
    return da


def subset_africa(da: xr.DataArray) -> xr.DataArray:
    if da.latitude[0] > da.latitude[-1]:
        lat_slice = slice(40, -40)
    else:
        lat_slice = AFRICA_LAT
    return da.sel(latitude=lat_slice, longitude=AFRICA_LON)


def _step_hours(step_vals: np.ndarray) -> np.ndarray:
    """Convert step coordinate values to hour offsets (handles timedelta64)."""
    if np.issubdtype(step_vals.dtype, np.timedelta64):
        return step_vals.astype("timedelta64[h]").astype(np.int64).astype(np.float64)
    vals = np.asarray(step_vals, dtype=np.float64)
    # Nanosecond timedeltas sometimes arrive as plain floats/ints.
    if np.nanmax(np.abs(vals)) > 1e11:
        return vals / 3.6e12
    return vals


def _select_tp_at_valid_time(da: xr.DataArray, when: np.datetime64) -> xr.DataArray:
    """Pick tp slice whose valid time best matches ``when``.

    Prefers an explicit ``valid_time`` (time, step) coordinate when present.
    Otherwise reconstructs valid = reference time + step. Step may be hours,
    6-hour indices, or timedelta64 — all are normalised to hours.
    """
    if "step" not in da.dims:
        return da
    target = np.datetime64(when, "ns")
    best_t = best_s = 0
    best_err_ns = np.iinfo(np.int64).max

    # Fast path: 2-D valid_time from ERA5 short-range precip Zarrs.
    vt = da.coords.get("valid_time")
    if vt is not None and set(getattr(vt, "dims", ())) >= {"time", "step"}:
        vt_ns = np.asarray(vt.values, dtype="datetime64[ns]").astype("int64")
        tgt = int(target.astype("int64"))
        err = np.abs(vt_ns - tgt)
        best_t, best_s = [int(i) for i in np.unravel_index(int(np.argmin(err)), err.shape)]
        return da.isel(time=best_t, step=best_s)

    time_vals = np.asarray(da["time"].values, dtype="datetime64[ns]")
    hours = _step_hours(np.asarray(da["step"].values))
    hour_offsets = [hours]
    if float(np.nanmax(hours)) <= 12:
        hour_offsets.append(hours * 6)
    for offsets in hour_offsets:
        for ti, ref in enumerate(time_vals):
            for si, h in enumerate(offsets):
                valid = ref + np.timedelta64(int(h), "h")
                err_ns = abs(int((valid - target) / np.timedelta64(1, "ns")))
                if err_ns < best_err_ns:
                    best_err_ns = err_ns
                    best_t, best_s = ti, si
    return da.isel(time=best_t, step=best_s)


def open_era5_slice(path: str, when: np.datetime64) -> xr.DataArray:
    ds = xr.open_zarr(path, storage_options=GCS_OPTS, consolidated=True)
    var = pick_era5_var(ds)
    da = ds[var]
    for drop in ("number", "surface", "expver"):
        if drop in da.dims and da.sizes[drop] == 1:
            da = da.isel({drop: 0})
    if var == "tp" and "step" in da.dims:
        da = _select_tp_at_valid_time(da, when)
    else:
        time_dim = "time" if "time" in da.dims else da.dims[0]
        da = da.isel({time_dim: valid_time_index(ds, when)})
        if "step" in da.dims and da.sizes["step"] == 1:
            da = da.isel(step=0)
    return da.squeeze(drop=True)


def open_era5_at_valid_time(var: str, when: np.datetime64, *, bucket: str = "gs://code4earth/era5") -> xr.DataArray:
    year = int(str(when)[:4])
    path = era5_gcs_path(var, year, bucket=bucket)
    da = open_era5_slice(path, when)
    da = normalize_lon(da)
    return subset_africa(da)


def align_truth_to_pred(truth: xr.DataArray, pred: xr.DataArray) -> xr.DataArray:
    truth = truth.load()
    pred = pred.load()
    if "latitude" in truth.dims and truth.latitude.size > 1:
        if float(truth.latitude[0]) > float(truth.latitude[-1]):
            truth = truth.sortby("latitude")
    if "longitude" in truth.dims and truth.longitude.size > 1:
        truth = normalize_lon(truth)
        if float(truth.longitude[0]) > float(truth.longitude[-1]):
            truth = truth.sortby("longitude")
    return truth.reindex(
        latitude=pred.latitude,
        longitude=pred.longitude,
        method="nearest",
    )

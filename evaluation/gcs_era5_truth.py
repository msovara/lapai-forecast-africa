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
AFRICA_LON = slice(-20, 55)


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


def open_era5_slice(path: str, when: np.datetime64) -> xr.DataArray:
    ds = xr.open_zarr(path, storage_options=GCS_OPTS)
    var = pick_era5_var(ds)
    da = ds[var]
    for drop in ("number", "surface", "expver"):
        if drop in da.dims and da.sizes[drop] == 1:
            da = da.isel({drop: 0})
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
    return truth.reindex(
        latitude=pred.latitude,
        longitude=pred.longitude,
        method="nearest",
    )

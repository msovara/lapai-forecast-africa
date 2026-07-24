"""Read GraphCast Africa forecast Zarrs from GCS.

Bucket layout (AfriClimate):
  gs://africlimate-ai-fcst-training/graphcast-africa/{year}_{stem}.zarr

Each store is Africa-cropped at 0.25° (lat -40..40, lon -20..70) with dims
``(time, prediction_timedelta, lat, lon)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

GCS_BUCKET_PREFIX = "gs://africlimate-ai-fcst-training/graphcast-africa"

# LapAI scorecard short name -> GraphCast / WeatherBench-style stem
GRAPHCAST_VAR_STEMS: dict[str, str] = {
    "t2m": "2m_temperature",
    "tp": "total_precipitation_6hr",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
}


def graphcast_zarr_uri(var: str, year: int, *, bucket_prefix: str = GCS_BUCKET_PREFIX) -> str:
    stem = GRAPHCAST_VAR_STEMS[var]
    return f"{bucket_prefix}/{year}_{stem}.zarr"


def open_graphcast_forecast(
    var: str,
    year: int,
    *,
    bucket_prefix: str = GCS_BUCKET_PREFIX,
    storage_options: dict[str, Any] | None = None,
) -> xr.DataArray:
    """Open one variable/year forecast Zarr as a DataArray (consolidated metadata)."""
    uri = graphcast_zarr_uri(var, year, bucket_prefix=bucket_prefix)
    opts = {"token": "google_default", **(storage_options or {})}
    ds = xr.open_zarr(uri, consolidated=True, storage_options=opts)
    # Prefer the named data var; fall back to first non-coord array.
    if var in ds.data_vars:
        da = ds[var]
    elif GRAPHCAST_VAR_STEMS[var] in ds.data_vars:
        da = ds[GRAPHCAST_VAR_STEMS[var]]
    else:
        data_vars = [k for k in ds.data_vars if k not in ("lat", "lon", "time", "prediction_timedelta")]
        if not data_vars:
            raise KeyError(f"No data variable in {uri}; keys={list(ds.data_vars)}")
        da = ds[data_vars[0]]
    # Normalise coord names to LapAI scorecard convention where possible.
    rename = {}
    if "lat" in da.coords and "latitude" not in da.coords:
        rename["lat"] = "latitude"
    if "lon" in da.coords and "longitude" not in da.coords:
        rename["lon"] = "longitude"
    if rename:
        da = da.rename(rename)
    return da


def _lead_hours_from_timedelta(td_coord: Any) -> np.ndarray:
    vals = np.asarray(td_coord)
    # nanosecond timedelta64 or int hours
    if np.issubdtype(vals.dtype, np.timedelta64):
        return (vals / np.timedelta64(1, "h")).astype(np.float64)
    return vals.astype(np.float64)


def select_init_lead(
    da: xr.DataArray,
    *,
    init: np.datetime64 | str,
    lead_hours: float,
) -> xr.DataArray:
    """Pick one init + lead from a GraphCast Africa forecast DataArray.

    Skips all-NaN inits when the exact init matches a hole (caller may choose
    a neighbouring day). Raises if the selected field is entirely NaN.
    """
    init64 = np.datetime64(init, "ns")
    if "time" not in da.dims:
        raise ValueError("expected 'time' dim on GraphCast forecast")
    times = np.asarray(da["time"].values, dtype="datetime64[ns]")
    ti = int(np.argmin(np.abs(times - init64)))
    if abs(int((times[ti] - init64) / np.timedelta64(1, "h"))) > 3:
        raise ValueError(f"no init near {init64}; nearest {times[ti]}")

    lead_dim = "prediction_timedelta" if "prediction_timedelta" in da.dims else None
    if lead_dim is None:
        raise ValueError("expected 'prediction_timedelta' dim")
    leads = _lead_hours_from_timedelta(da[lead_dim].values)
    li = int(np.argmin(np.abs(leads - float(lead_hours))))
    if abs(leads[li] - float(lead_hours)) > 1.0:
        raise ValueError(f"no lead near +{lead_hours}h; nearest {leads[li]}")

    out = da.isel(time=ti, **{lead_dim: li})
    if bool(np.isnan(np.asarray(out.values, dtype=np.float64)).all()):
        raise ValueError(
            f"GraphCast field all-NaN at init={times[ti]} lead={leads[li]}h "
            "(known hole days exist in the GCS archive)"
        )
    return out

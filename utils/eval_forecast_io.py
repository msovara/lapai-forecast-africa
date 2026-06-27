"""Write Mvua-protocol eval NetCDF (regular lat/lon, canonical variable names)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np

from utils.n320_forecast_io import (
    EVAL_ALIASES,
    _crop_global_latlon025,
    _regrid_n320_to_latlon025,
    load_eval_domain,
)

# Model short names -> scorecard names written to NetCDF.
MODEL_TO_EVAL = {
    "2t": "t2m",
    "tp": "tp",
    "10u": "u10",
    "10v": "v10",
}

PHASE0_VARS = ("t2m", "tp", "u10", "v10")


def _as_numpy(fields: dict[str, Any]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, value in fields.items():
        arr = np.asarray(value)
        if hasattr(value, "detach"):
            arr = value.detach().cpu().numpy()
        out[name] = np.asarray(arr, dtype=np.float32)
    return out


def regrid_state_fields_to_africa(
    fields: dict[str, Any],
    *,
    variables: tuple[str, ...] = PHASE0_VARS,
    cache_root: Path | None = None,
    domain: str = "africa",
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Regrid N320 vector fields to a regular 0.25 deg Africa subset."""
    domain_box = load_eval_domain(domain)
    lat_bounds = domain_box["lat"]
    lon_bounds = domain_box["lon"]
    raw = _as_numpy(fields)

    # Map eval name -> model field name present in state.
    model_names: dict[str, str] = {}
    for eval_name in variables:
        for model_name, alias in EVAL_ALIASES.items():
            if alias == eval_name and model_name in raw:
                model_names[eval_name] = model_name
                break
        if eval_name not in model_names and eval_name in raw:
            model_names[eval_name] = eval_name

    out_fields: dict[str, np.ndarray] = {}
    lat_1d = lon_1d = None
    for eval_name, model_name in model_names.items():
        global_field = _regrid_n320_to_latlon025(
            raw[model_name],
            cache_root=cache_root,
        )
        lat_1d, lon_1d, _, _, cropped = _crop_global_latlon025(
            global_field,
            lat_bounds=lat_bounds,
            lon_bounds=lon_bounds,
        )
        out_fields[eval_name] = np.asarray(cropped, dtype=np.float32)

    if lat_1d is None or lon_1d is None:
        raise ValueError(f"No requested variables found in state fields: {variables}")

    return out_fields, np.asarray(lat_1d), np.asarray(lon_1d)


def write_eval_regridded_netcdf(
    states: list[dict],
    path: Path,
    *,
    reference_date: dt.datetime,
    variables: tuple[str, ...] = PHASE0_VARS,
    cache_root: Path | None = None,
    domain: str = "africa",
) -> Path:
    """Write (time, latitude, longitude) NetCDF with t2m/tp/u10/v10 for run_scorecard."""
    import xarray as xr

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not states:
        raise ValueError("No forecast states to write")

    times: list[np.datetime64] = []
    stacks: dict[str, list[np.ndarray]] = {v: [] for v in variables}
    lat_1d = lon_1d = None

    for state in states:
        valid = state["date"]
        if isinstance(valid, dt.datetime):
            times.append(np.datetime64(valid))
        else:
            times.append(np.datetime64(str(valid)))

        cropped, lat_1d, lon_1d = regrid_state_fields_to_africa(
            state["fields"],
            variables=variables,
            cache_root=cache_root,
            domain=domain,
        )
        for name, grid in cropped.items():
            stacks[name].append(grid)

    data_vars = {}
    for name in variables:
        if not stacks[name]:
            continue
        data_vars[name] = (("time", "latitude", "longitude"), np.stack(stacks[name]))

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": np.array(times),
            "latitude": lat_1d,
            "longitude": lon_1d,
        },
        attrs={
            "title": "LapAI Phase 0 baseline forecast (eval grid)",
            "reference_time": reference_date.isoformat(sep=" "),
            "grid": "0.25 deg lat/lon Africa subset",
            "conventions": "CF-1.8",
        },
    )
    ds.to_netcdf(path)
    return path

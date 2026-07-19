"""Write Mvua-protocol eval NetCDF (regular lat/lon, canonical variable names)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np

from utils.n320_forecast_io import EVAL_ALIASES, load_eval_domain
from utils.regrid import crop_global_latlon025, regrid_n320_to_latlon025

# Model short names -> scorecard names written to NetCDF.
MODEL_TO_EVAL = {
    "2t": "t2m",
    "tp": "tp",
    "10u": "u10",
    "10v": "v10",
}

PHASE0_VARS = ("t2m", "tp", "u10", "v10")

# N320 octahedral reduced-Gaussian grid point count (teacher native grid). A model
# output with a different point count (e.g. a coarsened O96 Track A student) cannot
# use the earthkit N320->lat/lon matrix and is regridded with the offline grid bridge.
_N320_SIZE = 542080
_LATLON025_SHAPE = (721, 1440)


def _latlon025_target() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Global 0.25 deg grid (ECMWF 90..-90 lat, 0..359.75 lon) as 1-D coords + (N,2) latlons."""
    nlat, nlon = _LATLON025_SHAPE
    lat_1d = np.linspace(90.0, -90.0, nlat)
    lon_1d = np.linspace(0.0, 360.0 - 360.0 / nlon, nlon)
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
    tgt = np.stack([lat2d.ravel(), lon2d.ravel()], axis=1)
    return lat_1d, lon_1d, tgt


def build_output_grid_bridge(latitudes: np.ndarray, longitudes: np.ndarray):
    """Offline k-NN IDW bridge: unstructured model output grid -> global 0.25 deg field.

    Mirror of the IC grid bridge for the output side. Used when a coarsened student
    forecasts on a grid other than the teacher's N320 (no earthkit matrix, no internet).
    """
    from utils.grid_bridge import build_grid_bridge

    _, _, tgt = _latlon025_target()
    src = np.stack([np.asarray(latitudes), np.asarray(longitudes)], axis=1)
    return build_grid_bridge(src, tgt)


def _as_numpy(fields: dict[str, Any]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, value in fields.items():
        if hasattr(value, "detach"):
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value)
        out[name] = np.asarray(arr, dtype=np.float32).copy()
    return out


def snapshot_forecast_state(state: dict[str, Any]) -> dict[str, Any]:
    """Copy fields so runner in-place updates do not alias across saved steps."""
    return {**state, "fields": _as_numpy(state["fields"])}


def regrid_state_fields_to_africa(
    fields: dict[str, Any],
    *,
    variables: tuple[str, ...] = PHASE0_VARS,
    cache_root: Path | None = None,
    domain: str = "africa",
    bridge: Any | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Regrid model output fields to a regular 0.25 deg Africa subset.

    With ``bridge=None`` the teacher's N320 output is regridded via the earthkit
    matrix. Passing an offline ``GridBridge`` (from ``build_output_grid_bridge``)
    regrids an arbitrary unstructured output grid (e.g. a coarsened O96 student)
    onto the global 0.25 deg grid without earthkit/internet."""
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
        if bridge is not None:
            global_field = bridge.apply(
                np.asarray(raw[model_name], dtype=np.float64)
            ).reshape(_LATLON025_SHAPE)
        else:
            global_field = regrid_n320_to_latlon025(
                raw[model_name],
                cache_root=cache_root,
            )
        lat_1d, lon_1d, _, _, cropped = crop_global_latlon025(
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

    # If the model output is not on the teacher N320 grid (e.g. a coarsened O96
    # student), build the offline output grid bridge once from the latlons the
    # runner attached to each state, and reuse it for every field and timestep.
    bridge = None
    first = states[0]
    sample = next(iter(_as_numpy(first["fields"]).values()))
    n_out = int(np.asarray(sample).shape[-1])
    if n_out != _N320_SIZE and "latitudes" in first and "longitudes" in first:
        print(f"output grid bridge: {n_out} -> {_LATLON025_SHAPE[0]}x{_LATLON025_SHAPE[1]} (k-NN IDW, offline)")
        bridge = build_output_grid_bridge(first["latitudes"], first["longitudes"])

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
            bridge=bridge,
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

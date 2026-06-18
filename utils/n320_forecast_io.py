"""Save and plot N320 unstructured forecast fields from anemoi SimpleRunner states."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_CONFIG = _REPO_ROOT / "configs" / "eval.yaml"

# Model short names -> eval.yaml canonical names (for labels)
EVAL_ALIASES = {
    "2t": "t2m",
    "10u": "u10",
    "10v": "v10",
    "tp": "tp",
    "cp": "tp",
}


def load_eval_domain(name: str = "africa") -> dict[str, tuple[float, float]]:
    with EVAL_CONFIG.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    region = cfg["domain"]["regions"][name]
    return {"lat": tuple(region["lat"]), "lon": tuple(region["lon"])}


def _fix_longitudes(lons: np.ndarray) -> np.ndarray:
    return np.where(lons > 180, lons - 360, lons)


def _as_numpy(fields: dict[str, Any]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, value in fields.items():
        arr = np.asarray(value)
        if hasattr(value, "detach"):
            arr = value.detach().cpu().numpy()
        out[name] = np.asarray(arr, dtype=np.float32)
    return out


def attach_grid_coords(state: dict, latitudes: np.ndarray, longitudes: np.ndarray) -> dict:
    state = dict(state)
    state.setdefault("latitudes", np.asarray(latitudes, dtype=np.float32))
    state.setdefault("longitudes", np.asarray(longitudes, dtype=np.float32))
    return state


def write_forecast_netcdf(
    states: list[dict],
    path: Path,
    *,
    reference_date: dt.datetime,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> Path:
    """Write forecast steps to NetCDF (time, values) with lat/lon on values dim."""
    from netCDF4 import Dataset

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    latitudes = np.asarray(latitudes, dtype=np.float32)
    longitudes = np.asarray(longitudes, dtype=np.float32)
    n_values = latitudes.size

    with Dataset(path, "w", format="NETCDF4") as nc:
        nc.createDimension("values", n_values)
        time_dim = nc.createDimension("time", None)

        time_var = nc.createVariable("time", "i4", ("time",))
        time_var.units = f"seconds since {reference_date.isoformat(sep=' ')}"
        time_var.long_name = "time"
        time_var.calendar = "gregorian"

        lat_var = nc.createVariable("latitude", "f4", ("values",))
        lat_var.units = "degrees_north"
        lat_var.long_name = "latitude"
        lat_var[:] = latitudes

        lon_var = nc.createVariable("longitude", "f4", ("values",))
        lon_var.units = "degrees_east"
        lon_var.long_name = "longitude"
        lon_var[:] = longitudes

        var_cache: dict[str, Any] = {}
        for i, state in enumerate(states):
            valid = state["date"]
            if isinstance(valid, dt.datetime):
                step_seconds = int((valid - reference_date).total_seconds())
            else:
                step_seconds = int(state.get("step").total_seconds()) if state.get("step") else i * 21600
            time_var[i] = step_seconds

            fields = _as_numpy(state["fields"])
            for name, values in fields.items():
                if values.ndim != 1 or values.size != n_values:
                    continue
                if name not in var_cache:
                    var_cache[name] = nc.createVariable(name, "f4", ("time", "values"), fill_value=np.nan)
                var_cache[name][i] = values

        nc.title = "LapAI n320_gt6 open-data forecast"
        nc.grid = "N320 unstructured"
        nc.reference_time = reference_date.isoformat(sep=" ")

    return path


def plot_unstructured_map(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    output: Path,
    domain: str = "africa",
    cmap: str = "RdYlBu_r",
    units: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    """Scatter map for one unstructured field over an eval.yaml domain."""
    import matplotlib.pyplot as plt

    bounds = load_eval_domain(domain)
    lat = np.asarray(latitudes, dtype=np.float64)
    lon = _fix_longitudes(np.asarray(longitudes, dtype=np.float64))
    val = np.asarray(values, dtype=np.float64)

    mask = (
        (lat >= bounds["lat"][0])
        & (lat <= bounds["lat"][1])
        & (lon >= bounds["lon"][0])
        & (lon <= bounds["lon"][1])
        & np.isfinite(val)
    )
    if not mask.any():
        raise ValueError(f"No grid points in domain '{domain}' for plotting")

    lat_m = lat[mask]
    lon_m = lon[mask]
    val_m = val[mask]

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    sc = ax.scatter(
        lon_m,
        lat_m,
        c=val_m,
        s=0.35,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
        rasterized=True,
    )
    ax.set_xlim(bounds["lon"])
    ax.set_ylim(bounds["lat"])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    if units:
        cbar.set_label(units)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_temperature_celsius(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    values_k: np.ndarray,
    *,
    title: str,
    output: Path,
    domain: str = "africa",
) -> Path:
    """Plot 2m temperature (Kelvin in model) as degrees Celsius."""
    values_c = np.asarray(values_k, dtype=np.float64) - 273.15
    return plot_unstructured_map(
        latitudes,
        longitudes,
        values_c,
        title=title,
        output=output,
        domain=domain,
        cmap="RdYlBu_r",
        units="degC",
        vmin=-10,
        vmax=45,
    )


def default_plot_path(netcdf_path: Path, var: str, domain: str = "africa") -> Path:
    stem = netcdf_path.with_suffix("")
    return stem.parent / f"{stem.name}_{var}_{domain}.png"

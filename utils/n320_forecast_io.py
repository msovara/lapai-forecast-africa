"""Save and plot N320 unstructured forecast fields from anemoi SimpleRunner states."""

from __future__ import annotations

import datetime as dt
import os
import sys
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

OCEAN_COLOR = "#ffffff"


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
    static_fields: dict[str, Any] | None = None,
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

        if static_fields:
            for name, values in _as_numpy(static_fields).items():
                arr = np.asarray(values)
                if arr.ndim != 1 or arr.size != n_values:
                    continue
                var = nc.createVariable(name, "f4", ("values",))
                var.long_name = name
                var[:] = arr

        nc.title = "LapAI n320_gt6 open-data forecast"
        nc.grid = "N320 unstructured"
        nc.reference_time = reference_date.isoformat(sep=" ")

    return path


def _ensure_earthkit_regrid(*, cache_root: Path | None = None) -> None:
    """Configure earthkit caches and Windows URL fix before N320->lat/lon regrid."""
    cache_root = cache_root or (_REPO_ROOT / "data" / "cache" / "earthkit")
    data_dir = cache_root / "data"
    regrid_dir = cache_root / "regrid"
    data_dir.mkdir(parents=True, exist_ok=True)
    regrid_dir.mkdir(parents=True, exist_ok=True)

    import earthkit.data as ekd
    from earthkit.regrid.utils import caching as regrid_caching

    ekd.config.set(
        {
            "cache-policy": "user",
            "user-cache-directory": str(data_dir),
            "maximum-cache-disk-usage": 99,
        }
    )
    regrid_caching.SETTINGS.update(
        {
            "cache-policy": "user",
            "user-cache-directory": str(regrid_dir),
            "maximum-cache-disk-usage": 99,
        }
    )

    if os.name != "nt":
        return

    import logging

    import earthkit.regrid.db as regrid_db
    from earthkit.regrid.utils.download import download_and_cache

    log = logging.getLogger("earthkit.regrid.db")

    if getattr(regrid_db.UrlAccessor, "_lapai_matrix_path_patched", False):
        return

    def matrix_path_fixed(self, name: str) -> str:
        base = self._url.rstrip("/")
        rel = name.replace("\\", "/").lstrip("/")
        url = f"{base}/{rel}"
        try:
            return download_and_cache(
                url,
                owner="url",
                verify=True,
                force=None,
                chunk_size=1024 * 1024,
                http_headers=None,
                update_if_out_of_date=False,
                maximum_retries=5,
                retry_after=10,
            )
        except Exception:
            log.error("Could not download matrix file=%s", url)
            raise

    regrid_db.UrlAccessor.matrix_path = matrix_path_fixed  # type: ignore[method-assign]
    regrid_db.UrlAccessor._lapai_matrix_path_patched = True


def _regrid_n320_to_latlon025(values: np.ndarray, *, cache_root: Path | None = None) -> np.ndarray:
    """MIR regrid N320 vector -> global 0.25 deg lat/lon field (721 x 1440)."""
    import earthkit.regrid as ekr

    _ensure_earthkit_regrid(cache_root=cache_root)
    values = np.asarray(values, dtype=np.float64)
    out = ekr.interpolate(values, {"grid": "N320"}, {"grid": (0.25, 0.25)})
    return np.asarray(out, dtype=np.float64)


def _crop_global_latlon025(
    field: np.ndarray,
    *,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Crop a 721x1440 0.25 deg field (ECMWF 0-360 layout) to lat/lon bounds."""
    field = np.asarray(field)
    lats = np.linspace(90.0, -90.0, field.shape[0])
    lons = np.linspace(0.0, 360.0 - 360.0 / field.shape[1], field.shape[1])

    lat_sl = (lats >= lat_bounds[0]) & (lats <= lat_bounds[1])
    lon_min, lon_max = lon_bounds
    if lon_min < 0.0:
        lon_sl = (lons >= (360.0 + lon_min)) | (lons <= lon_max)
    else:
        lon_sl = (lons >= lon_min) & (lons <= lon_max)

    grid = field[np.ix_(lat_sl, lon_sl)]
    lat_1d = lats[lat_sl]
    lon_1d = lons[lon_sl]
    if lon_min < 0.0:
        lon_1d = np.where(lon_1d > 180.0, lon_1d - 360.0, lon_1d)
        order = np.argsort(lon_1d)
        lon_1d = lon_1d[order]
        grid = grid[:, order]

    lat_1d, lon_1d, lat_edges, lon_edges, grid = _normalize_plot_grid(lat_1d, lon_1d, grid)
    return lat_1d, lon_1d, lat_edges, lon_edges, grid


def _normalize_plot_grid(
    lat_1d: np.ndarray,
    lon_1d: np.ndarray,
    grid: np.ndarray,
    *,
    resolution: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return south-to-north latitudes and cell edges for matplotlib."""
    lat_1d = np.asarray(lat_1d, dtype=np.float64)
    lon_1d = np.asarray(lon_1d, dtype=np.float64)
    grid = np.asarray(grid) if not np.ma.isMaskedArray(grid) else grid.copy()

    if lat_1d.size > 1 and lat_1d[0] > lat_1d[-1]:
        lat_1d = lat_1d[::-1]
        grid = grid[::-1, :]

    half = resolution / 2.0
    lat_edges = np.linspace(lat_1d[0] - half, lat_1d[-1] + half, lat_1d.size + 1)
    lon_edges = np.linspace(lon_1d[0] - half, lon_1d[-1] + half, lon_1d.size + 1)
    return lat_1d, lon_1d, lat_edges, lon_edges, grid


def _render_field(
    ax,
    grid: np.ndarray,
    lat_1d: np.ndarray,
    lon_1d: np.ndarray,
    *,
    cmap,
    vmin: float,
    vmax: float,
    transform=None,
) -> Any:
    """Gouraud-shaded pcolormesh (smooth, no cell-edge or contour artifacts)."""
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
    kwargs = {
        "cmap": cmap,
        "vmin": vmin,
        "vmax": vmax,
        "shading": "gouraud",
        "rasterized": True,
        "zorder": 1,
    }
    if transform is not None:
        kwargs["transform"] = transform
    mesh = ax.pcolormesh(lon2d, lat2d, grid, **kwargs)
    mesh.set_edgecolor("face")
    mesh.set_linewidth(0)
    return mesh


def _upsample_grid(
    grid: np.ndarray,
    lat_1d: np.ndarray,
    lon_1d: np.ndarray,
    *,
    factor: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upsample a lat/lon grid for smoother display (MIR 0.25 deg -> finer)."""
    from scipy.ndimage import zoom

    if factor <= 1:
        return lat_1d, lon_1d, grid

    if np.ma.isMaskedArray(grid):
        fill = float(np.ma.mean(grid))
        data = grid.filled(fill)
        mask = grid.mask
    else:
        data = np.asarray(grid, dtype=np.float64)
        mask = ~np.isfinite(data)
        fill = float(np.nanmean(data))

    up = zoom(np.where(mask, fill, data), factor, order=1)
    lat_up = np.linspace(lat_1d[0], lat_1d[-1], up.shape[0])
    lon_up = np.linspace(lon_1d[0], lon_1d[-1], up.shape[1])
    if mask.any():
        mask_up = zoom(mask.astype(np.float32), factor, order=0) >= 0.5
        up = np.ma.array(up, mask=mask_up)
    return lat_up, lon_up, up


def _interpolate_unstructured_to_latlon(
    lat: np.ndarray,
    lon: np.ndarray,
    values: np.ndarray,
    *,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
    resolution: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate N320 (or other unstructured) points onto a regular lat/lon grid."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    n_lon = int(round((lon_bounds[1] - lon_bounds[0]) / resolution)) + 1
    n_lat = int(round((lat_bounds[1] - lat_bounds[0]) / resolution)) + 1
    lon_1d = np.linspace(lon_bounds[0], lon_bounds[1], n_lon)
    lat_1d = np.linspace(lat_bounds[0], lat_bounds[1], n_lat)
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)

    valid = np.isfinite(values)
    if not valid.any():
        raise ValueError("No finite values to interpolate")
    pts = np.column_stack([lon[valid], lat[valid]])
    linear = LinearNDInterpolator(pts, values[valid])
    nearest = NearestNDInterpolator(pts, values[valid])
    grid = linear(lon2d, lat2d)
    grid = np.where(np.isfinite(grid), grid, nearest(lon2d, lat2d))

    half = resolution / 2.0
    lat_edges = np.linspace(lat_1d[0] - half, lat_1d[-1] + half, lat_1d.size + 1)
    lon_edges = np.linspace(lon_1d[0] - half, lon_1d[-1] + half, lon_1d.size + 1)
    return lat_1d, lon_1d, lat_edges, lon_edges, grid


def _smooth_display_grid(
    grid: np.ndarray,
    *,
    sigma: float = 1.5,
    lat_1d: np.ndarray | None = None,
) -> np.ndarray:
    """Land-only smoothing for display (reduces MIR tile moiré)."""
    from scipy.ndimage import gaussian_filter

    if sigma <= 0.0:
        return grid

    if np.ma.isMaskedArray(grid):
        data = grid.astype(float).filled(np.nan)
        land = ~grid.mask
        orig_mask = grid.mask
    else:
        data = np.asarray(grid, dtype=np.float64)
        land = np.isfinite(data)
        orig_mask = ~land

    if not land.any():
        return grid

    def _weighted_smooth(values: np.ndarray, s: float) -> np.ndarray:
        weights = land.astype(np.float64)
        num = gaussian_filter(np.where(land, values, 0.0), sigma=s)
        den = gaussian_filter(weights, sigma=s)
        out = np.full_like(values, np.nan, dtype=np.float64)
        valid = (den > 1e-8) & land
        out[valid] = num[valid] / den[valid]
        return out

    smoothed = _weighted_smooth(data, sigma)
    if lat_1d is not None and lat_1d.size == grid.shape[0]:
        heavy = _weighted_smooth(data, sigma * 2.2)
        north_w = np.clip((lat_1d - 10.0) / 25.0, 0.0, 1.0)[:, None]
        smoothed = np.where(land, (1.0 - north_w) * smoothed + north_w * heavy, np.nan)

    if np.ma.isMaskedArray(grid):
        return np.ma.array(smoothed, mask=orig_mask)
    return smoothed


def _style_map_axes(
    ax,
    bounds: dict[str, tuple[float, float]],
    *,
    use_cartopy: bool,
) -> None:
    """Coastlines, country borders, and lat/lon tick labels (no grid lines)."""
    import matplotlib.pyplot as plt

    lon_min, lon_max = bounds["lon"]
    lat_min, lat_max = bounds["lat"]
    lon_ticks = np.arange(int(np.floor(lon_min / 10) * 10), int(np.ceil(lon_max / 10) * 10) + 1, 10)
    lat_ticks = np.arange(int(np.floor(lat_min / 10) * 10), int(np.ceil(lat_max / 10) * 10) + 1, 10)

    if use_cartopy:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

        ax.set_facecolor(OCEAN_COLOR)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="#222222", zorder=3)
        ax.add_feature(
            cfeature.BORDERS,
            linewidth=0.45,
            edgecolor="#666666",
            linestyle="-",
            alpha=0.85,
            zorder=3,
        )
        ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
        ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.tick_params(axis="both", labelsize=9, pad=2)
        ax.set_xlabel("Longitude", labelpad=6, fontsize=10)
        ax.set_ylabel("Latitude", labelpad=6, fontsize=10)
        return

    ax.set_facecolor(OCEAN_COLOR)
    ax.set_xticks(lon_ticks)
    ax.set_yticks(lat_ticks)
    ax.set_xlabel("Longitude", labelpad=4)
    ax.set_ylabel("Latitude", labelpad=4)


def _lsm_on_latlon_grid(
    lat: np.ndarray,
    lon: np.ndarray,
    lsm: np.ndarray,
    lat_1d: np.ndarray,
    lon_1d: np.ndarray,
) -> np.ndarray:
    """Regrid a 1D N320 land-sea mask onto an existing lat/lon plot grid."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    valid = np.isfinite(lsm)
    pts = np.column_stack([lon[valid], lat[valid]])
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
    linear = LinearNDInterpolator(pts, lsm[valid])
    nearest = NearestNDInterpolator(pts, lsm[valid])
    out = linear(lon2d, lat2d)
    return np.where(np.isfinite(out), out, nearest(lon2d, lat2d))


def _land_mask_on_latlon_grid(lat_1d: np.ndarray, lon_1d: np.ndarray) -> np.ndarray:
    """True where grid cell centre is over land (Natural Earth 50m)."""
    try:
        import cartopy.io.shapereader as shpreader
        import shapely.ops
        import shapely.vectorized
    except ImportError as exc:
        raise ImportError(
            "Ocean masking needs model `lsm` in the NetCDF or cartopy+shapely installed"
        ) from exc

    land_shp = shpreader.natural_earth(resolution="50m", category="physical", name="land")
    land = shapely.ops.unary_union(list(shpreader.Reader(land_shp).geometries()))
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
    return shapely.vectorized.contains(land, lon2d, lat2d)


def _apply_land_mask(
    grid: np.ndarray,
    lat_1d: np.ndarray,
    lon_1d: np.ndarray,
    *,
    lsm: np.ndarray | None = None,
    lsm_threshold: float = 0.5,
) -> np.ndarray:
    """Mask ocean cells; prefer model lsm on the same grid if supplied."""
    if lsm is not None and lsm.shape == grid.shape:
        land = np.asarray(lsm) >= lsm_threshold
    else:
        land = _land_mask_on_latlon_grid(lat_1d, lon_1d)
    masked = np.ma.array(grid, mask=~land)
    return masked


def _bin_to_latlon(
    lat: np.ndarray,
    lon: np.ndarray,
    values: np.ndarray,
    *,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
    resolution: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fallback: scipy griddata onto a regular lat/lon grid."""
    from scipy.interpolate import griddata

    n_lon = int(round((lon_bounds[1] - lon_bounds[0]) / resolution)) + 1
    n_lat = int(round((lat_bounds[1] - lat_bounds[0]) / resolution)) + 1
    lon_1d = np.linspace(lon_bounds[0], lon_bounds[1], n_lon)
    lat_1d = np.linspace(lat_bounds[0], lat_bounds[1], n_lat)
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
    grid = griddata((lon, lat), values, (lon2d, lat2d), method="linear")
    if np.isnan(grid).any():
        nearest = griddata((lon, lat), values, (lon2d, lat2d), method="nearest")
        grid = np.where(np.isnan(grid), nearest, grid)
    half = resolution / 2.0
    lat_edges = np.linspace(lat_1d[0] - half, lat_1d[-1] + half, lat_1d.size + 1)
    lon_edges = np.linspace(lon_1d[0] - half, lon_1d[-1] + half, lon_1d.size + 1)
    return lat_1d, lon_1d, lat_edges, lon_edges, grid


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
    resolution: float = 0.25,
    cache_root: Path | None = None,
    lsm: np.ndarray | None = None,
    mask_ocean: bool = True,
    smooth_sigma: float = 2.5,
    upsample_factor: int = 6,
) -> Path:
    """Map one N320 field: MIR regrid for plotting, then smooth upsample."""
    import matplotlib.pyplot as plt

    bounds = load_eval_domain(domain)
    lat = np.asarray(latitudes, dtype=np.float64)
    lon = _fix_longitudes(np.asarray(longitudes, dtype=np.float64))
    val = np.asarray(values, dtype=np.float64)

    lat_1d: np.ndarray
    lon_1d: np.ndarray
    lat_edges: np.ndarray
    lon_edges: np.ndarray
    grid: np.ndarray
    lsm_grid = None

    use_mir = val.ndim == 1 and val.size == lat.size and val.size >= 500_000
    if use_mir:
        try:
            global_field = _regrid_n320_to_latlon025(val, cache_root=cache_root)
            lat_1d, lon_1d, lat_edges, lon_edges, grid = _crop_global_latlon025(
                global_field,
                lat_bounds=bounds["lat"],
                lon_bounds=bounds["lon"],
            )
            if lsm is not None and np.asarray(lsm).size == val.size:
                lsm_global = _regrid_n320_to_latlon025(np.asarray(lsm, dtype=np.float64), cache_root=cache_root)
                _, _, _, _, lsm_grid = _crop_global_latlon025(
                    lsm_global,
                    lat_bounds=bounds["lat"],
                    lon_bounds=bounds["lon"],
                )
        except Exception as exc:
            print(f"earthkit regrid failed ({exc}); falling back to scipy griddata", file=sys.stderr)
            use_mir = False

    if not use_mir:
        domain_mask = (
            (lat >= bounds["lat"][0] - 5.0)
            & (lat <= bounds["lat"][1] + 5.0)
            & (lon >= bounds["lon"][0] - 5.0)
            & (lon <= bounds["lon"][1] + 5.0)
            & np.isfinite(val)
        )
        if not domain_mask.any():
            raise ValueError(f"No grid points in domain '{domain}' for plotting")
        lat_1d, lon_1d, lat_edges, lon_edges, grid = _bin_to_latlon(
            lat[domain_mask],
            lon[domain_mask],
            val[domain_mask],
            lat_bounds=bounds["lat"],
            lon_bounds=bounds["lon"],
            resolution=resolution,
        )
        if lsm is not None and np.asarray(lsm).size == lat.size:
            lsm_arr = np.asarray(lsm, dtype=np.float64)
            _, _, _, _, lsm_grid = _bin_to_latlon(
                lat[domain_mask],
                lon[domain_mask],
                lsm_arr[domain_mask],
                lat_bounds=bounds["lat"],
                lon_bounds=bounds["lon"],
                resolution=resolution,
            )

    val_m = grid[np.isfinite(grid)] if not np.ma.isMaskedArray(grid) else grid.compressed()

    if mask_ocean:
        try:
            grid = _apply_land_mask(grid, lat_1d, lon_1d, lsm=lsm_grid)
        except ImportError as exc:
            print(f"Warning: {exc}; plotting without ocean mask", file=sys.stderr)
        val_m = grid.compressed() if np.ma.isMaskedArray(grid) else grid[np.isfinite(grid)]

    grid = _smooth_display_grid(grid, sigma=smooth_sigma, lat_1d=lat_1d)
    lat_1d, lon_1d, grid = _upsample_grid(grid, lat_1d, lon_1d, factor=upsample_factor)
    grid = _smooth_display_grid(grid, sigma=1.2, lat_1d=lat_1d)

    if vmin is None or vmax is None:
        p2, p98 = np.percentile(val_m, [2, 98])
        vmin = p2 if vmin is None else vmin
        vmax = p98 if vmax is None else vmax
    if vmin >= vmax:
        vmin, vmax = float(np.min(val_m)), float(np.max(val_m))

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(OCEAN_COLOR)

    try:
        import cartopy.crs as ccrs

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent(
            [bounds["lon"][0], bounds["lon"][1], bounds["lat"][0], bounds["lat"][1]],
            crs=ccrs.PlateCarree(),
        )
        mesh = _render_field(
            ax,
            grid,
            lat_1d,
            lon_1d,
            cmap=cmap_obj,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
        )
        _style_map_axes(ax, bounds, use_cartopy=True)
        fig.subplots_adjust(left=0.08, bottom=0.10, right=0.88, top=0.94)
    except ImportError:
        fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
        mesh = _render_field(
            ax,
            grid,
            lat_1d,
            lon_1d,
            cmap=cmap_obj,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlim(bounds["lon"])
        ax.set_ylim(bounds["lat"])
        ax.set_aspect("equal", adjustable="box")
        _style_map_axes(ax, bounds, use_cartopy=False)

    ax.set_title(title)
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.85, pad=0.02)
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
    lsm: np.ndarray | None = None,
    mask_ocean: bool = True,
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
        lsm=lsm,
        mask_ocean=mask_ocean,
    )


def default_plot_path(netcdf_path: Path, var: str, domain: str = "africa") -> Path:
    stem = netcdf_path.with_suffix("")
    return stem.parent / f"{stem.name}_{var}_{domain}.png"

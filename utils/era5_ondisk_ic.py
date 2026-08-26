#!/usr/bin/env python3
"""Build 65-ch student IC states from on-disk / public ARCO ERA5 (no CDS).

Primary source: ``gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3``
(anonymous GCS). Same public ARCO store already used for Track A rescoring when
the private ``code4earth`` bucket is unavailable.

Optional local npy cache under ``data/cache/student_ic_arco/`` so subsequent
runs (or the credit env without gcsfs) can reuse built states offline.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

LOG = logging.getLogger(__name__)

LEVELS_13 = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
STATE_VARS = ["t", "u", "v", "q", "z"]

ARCO_DEFAULT = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ARCO_VAR_MAP = {
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
    "z": "geopotential",
}

DEFAULT_STATE_CACHE = Path(
    os.environ.get(
        "LAPAI_STUDENT_IC_CACHE",
        "data/cache/student_ic_arco",
    )
)


def _state_cache_path(cache_dir: Path, date: str, time: str) -> Path:
    return Path(cache_dir) / f"state_{date}_{time}_65x181x360.npy"


def _meta_cache_path(cache_dir: Path, date: str, time: str) -> Path:
    return Path(cache_dir) / f"state_{date}_{time}_65x181x360.meta.json"


def _regrid_025_to_1deg(field_hw: np.ndarray, lat_src: np.ndarray, lon_src: np.ndarray,
                        lat_t: np.ndarray, lon_t: np.ndarray) -> np.ndarray:
    """Nearest-neighbour 0.25° → 1° (lon 0–360)."""
    import xarray as xr

    lon_src = np.mod(np.asarray(lon_src, dtype=np.float64), 360.0)
    # Ensure lon ascending for xarray reindex
    order = np.argsort(lon_src)
    lon_src = lon_src[order]
    field = np.asarray(field_hw, dtype=np.float32)[:, order]
    da = xr.DataArray(
        field,
        dims=("latitude", "longitude"),
        coords={"latitude": np.asarray(lat_src, dtype=np.float64), "longitude": lon_src},
    )
    if float(da.latitude[0]) > float(da.latitude[-1]):
        da = da.sortby("latitude")
    out = da.reindex(latitude=lat_t, longitude=lon_t, method="nearest")
    return np.asarray(out.values, dtype=np.float32)


def load_cached_student_state(
    date: str,
    time: str,
    *,
    cache_dir: str | Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Return (state, meta) if a local npy IC exists, else None."""
    import json

    cache_dir = Path(cache_dir or DEFAULT_STATE_CACHE)
    path = _state_cache_path(cache_dir, date, time)
    if not path.is_file():
        return None
    state = np.load(str(path)).astype(np.float32)
    meta: dict[str, Any] = {
        "init_date": date,
        "init_time": time,
        "ic_source": "arco_era5_cache",
        "cache_path": str(path),
        "state_shape": list(state.shape),
    }
    meta_path = _meta_cache_path(cache_dir, date, time)
    if meta_path.is_file():
        meta.update(json.loads(meta_path.read_text(encoding="utf-8")))
    return state, meta


def save_student_state_cache(
    state: np.ndarray,
    date: str,
    time: str,
    meta: dict[str, Any],
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    import json

    cache_dir = Path(cache_dir or DEFAULT_STATE_CACHE)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _state_cache_path(cache_dir, date, time)
    np.save(str(path), np.asarray(state, dtype=np.float32))
    meta_out = dict(meta)
    meta_out["cache_path"] = str(path)
    _meta_cache_path(cache_dir, date, time).write_text(
        json.dumps(meta_out, indent=2), encoding="utf-8"
    )
    return path


def build_student_state_from_arco(
    date: str,
    time: str = "0000",
    *,
    arco_path: str = ARCO_DEFAULT,
    state_cache_dir: str | Path | None = None,
    use_cache: bool = True,
    ds: Any | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build ``(65,181,360)`` student state at ``date``+``time`` from ARCO ERA5.

    Only the analysis frame at the IC time is used (matches held-out CDS builder,
    which takes ``fields[key][1]`` and ignores the t−6h frame for the 65-ch stack).
    """
    from lapai_inference.cache_schema import lat_lon_mesh

    if use_cache:
        hit = load_cached_student_state(date, time, cache_dir=state_cache_dir)
        if hit is not None:
            LOG.info("ARCO IC cache hit %s %s", date, time)
            return hit

    import xarray as xr

    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "dummy")
    opts = {"token": os.environ.get("LAPAI_GCS_TOKEN", "anon")}
    close_ds = False
    if ds is None:
        LOG.info("Opening ARCO ERA5 %s", arco_path)
        ds = xr.open_zarr(arco_path, storage_options=opts, consolidated=True)
        close_ds = True

    try:
        hh = int(time[:2])
        mm = int(time[2:]) if len(time) >= 4 else 0
        when = np.datetime64(
            datetime.strptime(f"{date}{hh:02d}{mm:02d}", "%Y%m%d%H%M").isoformat()
        )
        lat_t, lon_t = lat_lon_mesh(181, 360)
        channels: list[np.ndarray] = []
        got_time = when
        for var in STATE_VARS:
            name = ARCO_VAR_MAP[var]
            if name not in ds:
                raise KeyError(f"ARCO missing {name}")
            da = ds[name]
            # Select all 13 levels at once for one time.
            cube = da.sel(time=when, method="nearest").sel(level=LEVELS_13)
            cube = cube.load()
            if "time" in cube.coords:
                got_time = np.datetime64(cube.time.values, "ns")
            # cube dims: level, latitude, longitude
            lat_src = np.asarray(cube.latitude.values, dtype=np.float64)
            lon_src = np.asarray(cube.longitude.values, dtype=np.float64)
            for li, _lev in enumerate(LEVELS_13):
                sl = cube.isel(level=li)
                field = np.asarray(sl.values, dtype=np.float32)
                if field.ndim != 2:
                    field = np.squeeze(field)
                channels.append(_regrid_025_to_1deg(field, lat_src, lon_src, lat_t, lon_t))
        state = np.stack(channels, axis=0)
        meta: dict[str, Any] = {
            "init_date": date,
            "init_time": time,
            "ic_source": "arco_era5",
            "arco_path": arco_path,
            "arco_time_selected": str(got_time),
            "state_shape": list(state.shape),
            "levels": list(LEVELS_13),
            "variables": list(STATE_VARS),
        }
        if use_cache:
            save_student_state_cache(state, date, time, meta, cache_dir=state_cache_dir)
        LOG.info("ARCO IC ready %s %s shape=%s", date, time, state.shape)
        return state, meta
    finally:
        if close_ds:
            ds.close()


def prefetch_arco_student_states(
    specs: list[tuple[str, str]],
    *,
    arco_path: str = ARCO_DEFAULT,
    state_cache_dir: str | Path | None = None,
) -> dict[tuple[str, str], tuple[np.ndarray, dict[str, Any]]]:
    """Build many IC states with a single ARCO open (cache hits skip network)."""
    import xarray as xr

    out: dict[tuple[str, str], tuple[np.ndarray, dict[str, Any]]] = {}
    missing: list[tuple[str, str]] = []
    for date, time in specs:
        hit = load_cached_student_state(date, time, cache_dir=state_cache_dir)
        if hit is not None:
            out[(date, time)] = hit
        else:
            missing.append((date, time))
    if not missing:
        return out

    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "dummy")
    opts = {"token": os.environ.get("LAPAI_GCS_TOKEN", "anon")}
    LOG.info("Opening ARCO once for %d missing ICs", len(missing))
    ds = xr.open_zarr(arco_path, storage_options=opts, consolidated=True)
    try:
        for date, time in missing:
            state, meta = build_student_state_from_arco(
                date,
                time,
                arco_path=arco_path,
                state_cache_dir=state_cache_dir,
                use_cache=True,
                ds=ds,
            )
            out[(date, time)] = (state, meta)
    finally:
        ds.close()
    return out


def build_student_state_from_ondisk(
    date: str,
    time: str = "0000",
    *,
    arco_path: str = ARCO_DEFAULT,
    state_cache_dir: str | Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Offline IC path: local npy cache, else public ARCO ERA5 (no CDS)."""
    hit = load_cached_student_state(date, time, cache_dir=state_cache_dir)
    if hit is not None:
        return hit
    return build_student_state_from_arco(
        date,
        time,
        arco_path=arco_path,
        state_cache_dir=state_cache_dir,
        use_cache=True,
    )

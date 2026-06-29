"""ERA5 initial conditions via CDS API with content-addressed GRIB cache.

Cache keys match the aifs-africa CDS layout so existing
``~/.cache/aifs-africa/era5/*.grib2`` files are reused on Lengau.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from utils.aifs_fields import PARAM_PL_CDS, PARAM_SFC, PRESSURE_LEVELS

LOG = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/lapai-forecast/era5")
AIFS_AFRICA_CACHE_DIR = os.path.expanduser("~/.cache/aifs-africa/era5")


def cache_path(request: dict, cache_dir: str | Path) -> Path:
    key = hashlib.md5(json.dumps(request, sort_keys=True).encode()).hexdigest()
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{key}.grib2"


def _regrid_to_n320(values: np.ndarray) -> np.ndarray:
    import earthkit.regrid as ekr

    values = np.asarray(values, dtype=np.float32)
    if values.shape != (721, 1440):
        raise ValueError(f"expected (721, 1440) before N320 regrid, got {values.shape}")
    return np.asarray(
        ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": "N320"}),
        dtype=np.float32,
    )


def ensure_cds_grib_cached(
    dataset: str,
    request: dict,
    *,
    cache_dir: str | Path,
    allow_download: bool = True,
) -> Path:
    """Download CDS GRIB to cache if missing; return path (no GRIB read / regrid)."""
    path = cache_path({"dataset": dataset, **request}, cache_dir)
    if path.is_file():
        LOG.info("CDS cache hit -> %s", path)
        return path
    if not allow_download:
        raise FileNotFoundError(
            f"CDS cache miss (offline mode): {path}\n"
            "Populate on a machine with internet:\n"
            "  python scripts/populate_phase0_ic_cache.py"
        )
    import cdsapi

    LOG.info("CDS download -> %s", path)
    tmp = path.with_suffix(".grib2.tmp")
    client = cdsapi.Client()
    client.retrieve(dataset, request, str(tmp))
    tmp.replace(path)
    return path


def ensure_era5_ic_grib_cache(
    date: str,
    time: str,
    *,
    cache_dir: str | Path | None = None,
    allow_download: bool = True,
) -> list[Path]:
    """Ensure all GRIB files for init (date, time) and t-6h exist in cache."""
    cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
    dt = datetime.datetime.strptime(f"{date}{time}", "%Y%m%d%H%M")
    lag = dt - datetime.timedelta(hours=6)
    datetimes = [
        (lag.strftime("%Y%m%d"), lag.strftime("%H%M")),
        (dt.strftime("%Y%m%d"), dt.strftime("%H%M")),
    ]
    paths: list[Path] = []
    for d, t in datetimes:
        paths.append(
            ensure_cds_grib_cached(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "param": PARAM_SFC,
                    "date": d,
                    "time": t,
                    "grid": [0.25, 0.25],
                    "format": "grib",
                },
                cache_dir=cache_dir,
                allow_download=allow_download,
            )
        )
        paths.append(
            ensure_cds_grib_cached(
                "reanalysis-era5-pressure-levels",
                {
                    "product_type": "reanalysis",
                    "param": PARAM_PL_CDS,
                    "pressure_level": PRESSURE_LEVELS,
                    "date": d,
                    "time": t,
                    "grid": [0.25, 0.25],
                    "format": "grib",
                },
                cache_dir=cache_dir,
                allow_download=allow_download,
            )
        )
    LOG.info("CDS GRIB cache ready for %s %s (%d files)", date, time, len(paths))
    return paths


def cached_cds(
    dataset: str,
    request: dict,
    *,
    cache_dir: str | Path,
    allow_download: bool = True,
):
    import earthkit.data as ekd

    path = ensure_cds_grib_cached(
        dataset, request, cache_dir=cache_dir, allow_download=allow_download
    )
    return ekd.from_source("file", str(path))


def _retrieve_sfc(date: str, time: str, cache_dir: str | Path, allow_download: bool):
    return cached_cds(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "param": PARAM_SFC,
            "date": date,
            "time": time,
            "grid": [0.25, 0.25],
            "format": "grib",
        },
        cache_dir=cache_dir,
        allow_download=allow_download,
    )


def _retrieve_pl(date: str, time: str, cache_dir: str | Path, allow_download: bool):
    return cached_cds(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "param": PARAM_PL_CDS,
            "pressure_level": PRESSURE_LEVELS,
            "date": date,
            "time": time,
            "grid": [0.25, 0.25],
            "format": "grib",
        },
        cache_dir=cache_dir,
        allow_download=allow_download,
    )


def _fields_to_dict(data, *, levelist: bool = False) -> dict[str, list[np.ndarray]]:
    result: dict[str, list[np.ndarray]] = defaultdict(list)
    for field in data:
        name = (
            f"{field.metadata('param')}_{field.metadata('levelist')}"
            if levelist
            else field.metadata("param")
        )
        result[name].append(_regrid_to_n320(field.to_numpy(dtype=np.float32)))
    return result


def retrieve_era5_fields(
    date: str,
    time: str,
    *,
    cache_dir: str | Path | None = None,
    allow_download: bool = True,
) -> dict[str, np.ndarray]:
    """Return AIFS input fields for init (date, time), stacked as (2, n320)."""
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    dt = datetime.datetime.strptime(f"{date}{time}", "%Y%m%d%H%M")
    lag = dt - datetime.timedelta(hours=6)
    datetimes = [
        (lag.strftime("%Y%m%d"), lag.strftime("%H%M")),
        (dt.strftime("%Y%m%d"), dt.strftime("%H%M")),
    ]

    fields: dict[str, list[np.ndarray]] = {}
    for d, t in datetimes:
        for name, arrays in _fields_to_dict(
            _retrieve_sfc(d, t, cache_dir, allow_download), levelist=False
        ).items():
            fields.setdefault(name, []).extend(arrays)
        for name, arrays in _fields_to_dict(
            _retrieve_pl(d, t, cache_dir, allow_download), levelist=True
        ).items():
            fields.setdefault(name, []).extend(arrays)

    out: dict[str, np.ndarray] = {}
    for name, parts in fields.items():
        out[name] = np.stack(parts)
    LOG.info("CDS IC ready: %d fields for %s %s (cache=%s)", len(out), date, time, cache_dir)
    return out


def build_cds_input_state(
    init_dt: datetime.datetime,
    *,
    cache_dir: str | Path | None = None,
    allow_download: bool = True,
) -> dict:
    date = init_dt.strftime("%Y%m%d")
    time = init_dt.strftime("%H%M")
    return {
        "date": init_dt,
        "fields": retrieve_era5_fields(date, time, cache_dir=cache_dir, allow_download=allow_download),
    }


def list_required_cache_keys(inits: list[str], init_time: str = "0000") -> list[Path]:
    """Return expected cache file paths for documentation / verification."""
    cache_dir = Path(DEFAULT_CACHE_DIR)
    paths: list[Path] = []
    for init in inits:
        dt = datetime.datetime.strptime(f"{init}{init_time}", "%Y%m%d%H%M")
        lag = dt - datetime.timedelta(hours=6)
        for d, t in (
            (lag.strftime("%Y%m%d"), lag.strftime("%H%M")),
            (dt.strftime("%Y%m%d"), dt.strftime("%H%M")),
        ):
            paths.append(
                cache_path(
                    {
                        "dataset": "reanalysis-era5-single-levels",
                        "product_type": "reanalysis",
                        "param": PARAM_SFC,
                        "date": d,
                        "time": t,
                        "grid": [0.25, 0.25],
                        "format": "grib",
                    },
                    cache_dir,
                )
            )
            paths.append(
                cache_path(
                    {
                        "dataset": "reanalysis-era5-pressure-levels",
                        "product_type": "reanalysis",
                        "param": PARAM_PL_CDS,
                        "pressure_level": PRESSURE_LEVELS,
                        "date": d,
                        "time": t,
                        "grid": [0.25, 0.25],
                        "format": "grib",
                    },
                    cache_dir,
                )
            )
    return paths

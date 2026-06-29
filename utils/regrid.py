"""earthkit-regrid helpers — shared by CDS IC load and eval NetCDF output.

Public entry point for reviewers comparing to aifs-africa:

- ``regrid_to_n320`` — ERA5 0.25° lat/lon (721×1440) → N320 vector (CDS initial conditions)
- ``regrid_n320_to_latlon025`` — N320 vector → global 0.25° field (eval / scorecard NetCDF)

Implementations live in ``utils/cds_ic.py`` (input) and ``utils/n320_forecast_io.py`` (output).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.cds_ic import _regrid_to_n320
from utils.n320_forecast_io import _crop_global_latlon025, _regrid_n320_to_latlon025

# Public aliases (aifs-africa naming parity).
regrid_to_n320 = _regrid_to_n320


def regrid_n320_to_latlon025(
    values: np.ndarray,
    *,
    cache_root: Path | None = None,
) -> np.ndarray:
    """N320 unstructured vector → global 0.25° lat/lon (721×1440)."""
    return _regrid_n320_to_latlon025(values, cache_root=cache_root)


__all__ = [
    "regrid_to_n320",
    "regrid_n320_to_latlon025",
    "crop_global_latlon025",
]

crop_global_latlon025 = _crop_global_latlon025

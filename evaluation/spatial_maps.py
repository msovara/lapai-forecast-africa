"""Load forecast / ERA5 fields for spatial map panels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from evaluation.gcs_era5_truth import align_truth_to_pred, open_era5_at_valid_time
from evaluation.phase0_scorecard import _resolve_fc_var, forecast_step_and_valid_time

PHASE0_FORECAST_DIR = Path("data/processed/phase0/forecasts")
TRACKA_FORECAST_DIR = Path("data/processed/trackA/forecasts")

VAR_LABELS = {
    "t2m": ("2 m temperature", "K"),
    "u10": ("10 m u-wind", "m s⁻¹"),
    "v10": ("10 m v-wind", "m s⁻¹"),
    "tp": ("Total precipitation", "m"),
}

DEFAULT_INITS = ["20230101", "20230108", "20230115", "20230122", "20230129"]
DEFAULT_LEADS = [24, 48, 72, 120, 168, 240]


def forecast_path(base_dir: Path, init: str) -> Path:
    return base_dir / f"{init}_00Z.nc"


def load_forecast_slice(
    path: Path,
    variable: str,
    lead_hours: int,
) -> tuple[xr.DataArray, np.datetime64]:
    if not path.is_file():
        raise FileNotFoundError(path)
    fc = xr.open_dataset(path)
    try:
        step, valid_time = forecast_step_and_valid_time(fc, lead_hours)
        fc_name = _resolve_fc_var(fc, variable)
        if fc_name is None:
            raise KeyError(f"{variable!r} not in {path.name}")
        da = fc[fc_name].isel(time=step).load()
        da.name = variable
        return da, valid_time
    finally:
        fc.close()


def load_era5_slice(
    variable: str,
    valid_time: np.datetime64,
    *,
    pred_template: xr.DataArray,
    gcs_bucket: str = "gs://code4earth/era5",
) -> xr.DataArray:
    truth = open_era5_at_valid_time(variable, valid_time, bucket=gcs_bucket)
    aligned = align_truth_to_pred(truth, pred_template)
    aligned.name = variable
    return aligned


def field_stats(a: xr.DataArray | np.ndarray, b: xr.DataArray | np.ndarray) -> dict[str, float]:
    """RMSE and bias (a minus b) over finite points.

    Accepts xarray DataArrays or plain ndarrays (Streamlit often passes the latter).
    """

    def _as_float64(x: xr.DataArray | np.ndarray) -> np.ndarray:
        # Prefer .values only for xarray; never call .values on ndarray.
        if isinstance(x, xr.DataArray):
            data = x.values
        elif hasattr(x, "values") and not isinstance(x, np.ndarray):
            data = x.values  # pandas / other array wrappers
        else:
            data = x
        return np.asarray(data, dtype=np.float64)

    p = _as_float64(a)
    t = _as_float64(b)
    mask = np.isfinite(p) & np.isfinite(t)
    if not mask.any():
        return {"rmse": float("nan"), "bias": float("nan")}
    diff = p - t
    rmse = float(np.sqrt(np.mean(diff[mask] ** 2)))
    bias = float(np.mean(diff[mask]))
    return {"rmse": rmse, "bias": bias}

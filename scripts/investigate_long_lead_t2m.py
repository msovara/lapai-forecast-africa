#!/usr/bin/env python3
"""Diagnose Phase 0 long-lead t2m scorecard degradation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from evaluation.gcs_era5_truth import align_truth_to_pred, open_era5_at_valid_time
from evaluation.phase0_scorecard import forecast_step_and_valid_time
from lapai_inference.cache_schema import cosine_latitude_weights

FC = _REPO / "data/processed/phase0/forecasts/20230101_00Z.nc"
LEADS = [24, 48, 72, 120, 168, 240]


def weighted_rmse_acc(pred: np.ndarray, truth: np.ndarray, lat: np.ndarray) -> tuple[float, float]:
    lat1d = np.asarray(lat, dtype=np.float64).ravel()
    w1d = np.asarray(cosine_latitude_weights(lat1d), dtype=np.float64).ravel()
    w2d = w1d[:, None] * np.ones((1, pred.shape[1]), dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(truth)
    err2 = (pred - truth) ** 2
    rmse = float(np.sqrt(np.sum(w2d[mask] * err2[mask]) / np.sum(w2d[mask])))
    clim = np.nanmean(truth, axis=0, keepdims=True)
    pa, ta = pred - clim, truth - clim
    num = np.sum(w2d[mask] * pa[mask] * ta[mask])
    den = np.sqrt(np.sum(w2d[mask] * pa[mask] ** 2) * np.sum(w2d[mask] * ta[mask] ** 2))
    acc = float(num / den) if den > 0 else float("nan")
    return rmse, acc


def main() -> None:
    fc = xr.open_dataset(FC)
    ref = np.datetime64(fc.attrs.get("reference_time", "2023-01-01T00:00:00"))
    print("reference_time attr:", fc.attrs.get("reference_time"))
    print("n_time:", fc.sizes["time"], "first/last time:", fc.time.values[0], fc.time.values[-1])
    print()
    print(f"{'lead':>6} {'step':>4} {'valid_fc':>26} {'valid_exp':>26} {'dt_h':>6} {'RMSE':>8} {'ACC':>8} {'bias':>8}")
    for lead in LEADS:
        step, valid_fc = forecast_step_and_valid_time(fc, lead)
        valid_exp = ref + np.timedelta64(lead, "h")
        dt_h = (valid_fc - valid_exp) / np.timedelta64(1, "h")
        pred = fc["t2m"].isel(time=step).load()
        truth = open_era5_at_valid_time("t2m", valid_fc).load()
        aligned = align_truth_to_pred(truth, pred)
        p = pred.values.squeeze()
        t = aligned.values.squeeze()
        lat = pred.latitude.values
        rmse, acc = weighted_rmse_acc(p, t, lat)
        bias = float(np.nanmean(p - t))
        print(
            f"+{lead:>3}h {step:>4} {str(valid_fc):>26} {str(valid_exp):>26} {float(dt_h):>6.0f} "
            f"{rmse:>8.2f} {acc:>8.3f} {bias:>8.2f}"
        )
        # Also score using expected valid time (if mismatch)
        if abs(float(dt_h)) > 0.5:
            truth2 = open_era5_at_valid_time("t2m", valid_exp).load()
            a2 = align_truth_to_pred(truth2, pred)
            r2, c2 = weighted_rmse_acc(p, a2.values.squeeze(), lat)
            print(f"       -> if truth at valid_exp: RMSE={r2:.2f} ACC={c2:.3f}")

    print("\nstep-to-step t2m delta (mean abs change between consecutive steps):")
    arr = fc["t2m"].values
    for i in range(1, min(40, arr.shape[0])):
        d = float(np.mean(np.abs(arr[i] - arr[i - 1])))
        if i in (3, 19, 39):
            print(f"  step {i-1}->{i}: mean |delta| = {d:.3f} K")

    fc.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Estimate whether constant t2m debias at +24h would pass A1 (one GCS open)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from evaluation.gcs_era5_truth import (
    GCS_OPTS,
    align_truth_to_pred,
    era5_gcs_path,
    normalize_lon,
    open_era5_slice,
    pick_era5_var,
    subset_africa,
)
from evaluation.phase0_scorecard import _score_point, forecast_step_and_valid_time
from lapai_inference.cache_schema import cosine_latitude_weights

ROOT = Path(__file__).resolve().parents[1]
INITS = ["20230101", "20230108", "20230115", "20230122", "20230129"]


def main() -> None:
    base = json.loads((ROOT / "reports/PHASE0_BASELINE_SCORECARD.json").read_text())
    path = era5_gcs_path("t2m", 2023)
    print("opening", path, flush=True)
    ds = xr.open_zarr(path, storage_options=GCS_OPTS, consolidated=True)
    var = pick_era5_var(ds)
    print("opened", var, dict(ds.sizes), flush=True)

    print(
        f"{'init':8} {'raw':>8} {'debias':>8} {'base':>8} {'need':>8} "
        f"{'bias':>8} {'raw_ok':>6} {'deb_ok':>6}",
        flush=True,
    )
    for init in INITS:
        fc = xr.open_dataset(ROOT / f"data/processed/trackA/forecasts/{init}_00Z.nc")
        step, vt = forecast_step_and_valid_time(fc, 24)
        pred = fc["t2m"].isel(time=step).assign_coords(name="t2m")
        # mirror open_era5_at_valid_time without re-opening
        year = int(str(vt)[:4])
        assert year == 2023
        da = ds[var]
        for drop in ("number", "surface", "expver"):
            if drop in da.dims and da.sizes[drop] == 1:
                da = da.isel({drop: 0})
        time_dim = "time" if "time" in da.dims else da.dims[0]
        times = np.asarray(ds["valid_time"].values if "valid_time" in ds else ds["time"].values)
        # use same helper path as production
        from evaluation.gcs_era5_truth import valid_time_index

        truth = da.isel({time_dim: valid_time_index(ds, vt)}).squeeze(drop=True)
        if "step" in truth.dims:
            truth = truth.isel(step=0)
        truth = subset_africa(normalize_lon(truth))

        ta = align_truth_to_pred(truth, pred)
        p = np.asarray(pred.values, dtype=np.float64).squeeze()
        t = np.asarray(ta.values, dtype=np.float64).squeeze()
        m = np.isfinite(p) & np.isfinite(t)
        lat = np.asarray(pred.latitude.values, dtype=np.float64).squeeze()
        w = np.asarray(cosine_latitude_weights(lat), dtype=np.float64).squeeze()
        w2 = w[:, None] * np.ones(p.shape[1], dtype=np.float64)
        bias = float(np.sum(w2[m] * (p[m] - t[m])) / np.sum(w2[m]))
        raw = _score_point(pred, truth)["rmse"]
        pred_db = pred.copy(deep=True)
        pred_db.values = np.asarray(pred.values, dtype=np.float64) - bias
        deb = _score_point(pred_db.assign_coords(name="t2m"), truth)["rmse"]
        brow = next(r for r in base["results"] if r["init_date"] == init and r["lead_hours"] == 24)
        brmse = float(brow["variables"]["t2m"]["rmse"])
        need = brmse * 1.05
        print(
            f"{init:8} {raw:8.3f} {deb:8.3f} {brmse:8.3f} {need:8.3f} "
            f"{bias:+8.3f} {str(raw <= need):>6} {str(deb <= need):>6}",
            flush=True,
        )
        fc.close()
    ds.close()


if __name__ == "__main__":
    main()

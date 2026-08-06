#!/usr/bin/env python3
"""Re-score only ``tp`` rows in TRACKA_COARSEN_SCORECARD.json (cached GCS open)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from evaluation.gcs_era5_truth import (
    GCS_OPTS,
    align_truth_to_pred,
    era5_gcs_path,
    normalize_lon,
    pick_era5_var,
    subset_africa,
    _select_tp_at_valid_time,
)
from evaluation.phase0_scorecard import _resolve_fc_var, _score_point, forecast_step_and_valid_time

ROOT = Path(__file__).resolve().parents[1]
SCORECARD = ROOT / "reports" / "TRACKA_COARSEN_SCORECARD.json"


def _resolve_forecast(row: dict) -> Path:
    path = Path(row["forecast"])
    if path.is_file():
        return path
    alt = ROOT / "data/processed/trackA/forecasts" / path.name
    return alt if alt.is_file() else path


def main() -> None:
    card = json.loads(SCORECARD.read_text(encoding="utf-8"))
    # One open for the whole rescore (2023 Jan inits).
    path = era5_gcs_path("tp", 2023)
    print("opening", path, flush=True)
    ds = xr.open_zarr(path, storage_options=GCS_OPTS, consolidated=True)
    var = pick_era5_var(ds)
    da0 = ds[var]
    for drop in ("number", "surface", "expver"):
        if drop in da0.dims and da0.sizes[drop] == 1:
            da0 = da0.isel({drop: 0})
    print("opened", dict(da0.sizes), flush=True)

    n_ok = n_nan = n_err = 0
    for row in card.get("results", []):
        if row.get("skipped"):
            continue
        lead = int(row["lead_hours"])
        fc_path = _resolve_forecast(row)
        fc = xr.open_dataset(fc_path)
        try:
            step, valid_time = forecast_step_and_valid_time(fc, lead)
            fc_name = _resolve_fc_var(fc, "tp")
            if fc_name is None:
                row["variables"]["tp"] = {"error": "missing in forecast file"}
                n_err += 1
                continue
            pred = fc[fc_name].isel(time=step).assign_coords(name="tp")
            truth = _select_tp_at_valid_time(da0, valid_time)
            truth = subset_africa(normalize_lon(truth)).squeeze(drop=True)
            scores = _score_point(pred, truth)
            out = {
                k: (None if (isinstance(v, float) and not np.isfinite(v)) else float(v))
                for k, v in scores.items()
            }
            row["variables"]["tp"] = out
            if out.get("rmse") is None:
                n_nan += 1
            else:
                n_ok += 1
            print(
                f"{row.get('init_date')} +{lead}h tp rmse={out.get('rmse')} "
                f"acc={out.get('acc')} pod={out.get('pod_1mm')}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            row["variables"]["tp"] = {"error": str(exc)}
            n_err += 1
            print(f"{row.get('init_date')} +{lead}h ERROR {exc}", flush=True)
        finally:
            fc.close()

    ds.close()
    card["tp_rescored_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    card["tp_rescore_note"] = "tp only; fixed step/valid_time selection in gcs_era5_truth"
    SCORECARD.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {SCORECARD}", flush=True)
    print(f"ok={n_ok} still_nan={n_nan} err={n_err}", flush=True)


if __name__ == "__main__":
    main()

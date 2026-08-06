#!/usr/bin/env python3
"""Fast Track A gate rescore: cache one GCS open per variable; gate leads only."""
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
    valid_time_index,
    _select_tp_at_valid_time,
)
from evaluation.phase0_scorecard import (
    _resolve_fc_var,
    _score_point,
    forecast_step_and_valid_time,
    write_phase0_scorecard,
)
from evaluation.trackA_gate import run_coarsen_gate, write_gate_report

ROOT = Path(__file__).resolve().parents[1]
FORECAST_DIR = ROOT / "data/processed/trackA/forecasts"
SCORECARD = ROOT / "reports/TRACKA_COARSEN_SCORECARD.json"
GATE_OUT = ROOT / "reports/TRACKA_A1_GATE.json"
VARS = ["t2m", "tp", "u10", "v10"]
LEADS = [24, 48]


class TruthCache:
    def __init__(self) -> None:
        self._ds: dict[str, xr.Dataset] = {}
        self._da: dict[str, xr.DataArray] = {}

    def open_var(self, var: str, year: int = 2023) -> xr.DataArray:
        key = f"{var}_{year}"
        if key in self._da:
            return self._da[key]
        path = era5_gcs_path(var, year)
        print(f"opening {path}", flush=True)
        ds = xr.open_zarr(path, storage_options=GCS_OPTS, consolidated=True)
        name = pick_era5_var(ds)
        da = ds[name]
        for drop in ("number", "surface", "expver"):
            if drop in da.dims and da.sizes[drop] == 1:
                da = da.isel({drop: 0})
        self._ds[key] = ds
        self._da[key] = da
        print(f"  opened {name} {dict(da.sizes)}", flush=True)
        return da

    def at(self, var: str, when: np.datetime64) -> xr.DataArray:
        year = int(str(when)[:4])
        da = self.open_var(var, year)
        if var == "tp" and "step" in da.dims:
            sl = _select_tp_at_valid_time(da, when)
        else:
            ds = self._ds[f"{var}_{year}"]
            time_dim = "time" if "time" in da.dims else da.dims[0]
            sl = da.isel({time_dim: valid_time_index(ds, when)})
            if "step" in sl.dims and sl.sizes.get("step", 1) == 1:
                sl = sl.isel(step=0)
        return subset_africa(normalize_lon(sl.squeeze(drop=True)))

    def close(self) -> None:
        for ds in self._ds.values():
            ds.close()


def main() -> None:
    paths = sorted(FORECAST_DIR.glob("*_00Z.nc"))
    if not paths:
        raise SystemExit(f"no forecasts in {FORECAST_DIR}")
    print("forecasts:", [p.name for p in paths], flush=True)
    cache = TruthCache()
    results = []
    try:
        for path in paths:
            init = path.name.split("_")[0]
            fc = xr.open_dataset(path)
            try:
                for lead in LEADS:
                    step, vt = forecast_step_and_valid_time(fc, lead)
                    row = {
                        "forecast": str(path),
                        "init_date": init,
                        "lead_hours": lead,
                        "valid_time": str(vt),
                        "variables": {},
                    }
                    for var in VARS:
                        fc_name = _resolve_fc_var(fc, var)
                        if fc_name is None:
                            row["variables"][var] = {"error": "missing in forecast"}
                            continue
                        pred = fc[fc_name].isel(time=step).assign_coords(name=var)
                        try:
                            truth = cache.at(var, vt)
                            scores = _score_point(pred, truth)
                            row["variables"][var] = {
                                k: (None if isinstance(v, float) and not np.isfinite(v) else float(v))
                                for k, v in scores.items()
                            }
                        except Exception as exc:  # noqa: BLE001
                            row["variables"][var] = {"error": str(exc)}
                        rmse = row["variables"][var].get("rmse")
                        print(f"{init} +{lead}h {var} rmse={rmse}", flush=True)
                    results.append(row)
            finally:
                fc.close()
    finally:
        cache.close()

    card = {
        "phase": "trackA_coarsen_extend_15k",
        "status": "candidate",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checkpoint": "models/trackA_coarsen_full_extend_runs/.../inference-last.ckpt",
        "domain": "africa",
        "variables": VARS,
        "leads_hours": LEADS,
        "truth_source": "gcs",
        "inits_scored": [p.name.split("_")[0] for p in paths],
        "forecasts": [str(p) for p in paths],
        "results": results,
    }
    write_phase0_scorecard(card, SCORECARD)
    print(f"wrote {SCORECARD}", flush=True)

    report = run_coarsen_gate(SCORECARD, config=__import__("evaluation.run_scorecard", fromlist=["_read_eval_config"])._read_eval_config(ROOT / "configs/trackA_coarsen_full.yaml"))
    write_gate_report(report, GATE_OUT)
    n_ok = sum(1 for c in report["checks"] if c.get("passed"))
    n = len(report["checks"])
    print(f"\nGATE passed={report['passed']} ({n_ok}/{n})", flush=True)
    for c in report["checks"]:
        if c.get("variable") == "t2m" and c.get("lead_hours") == 24:
            print(
                f"  t2m+24 {c['init_date']}: deg={c.get('degradation_pct'):.1f}% "
                f"pass={c.get('passed')} base={c.get('baseline_rmse'):.3f} cand={c.get('candidate_rmse'):.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()

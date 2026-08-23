#!/usr/bin/env python3
"""Score A1b GT forecasts and gate A2 prune vs A1b (prune-only damage)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from evaluation.gcs_era5_truth import (
    GCS_OPTS,
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
from evaluation.run_scorecard import _read_eval_config

ROOT = Path(__file__).resolve().parents[1]
A1B_DIR = ROOT / "data/processed/trackA_gt_coarsen/forecasts"
A1B_CARD = ROOT / "reports/TRACKA_GT_COARSEN_SCORECARD.json"
PRUNE_CARD = ROOT / "reports/TRACKA_PRUNE_SCORECARD.json"
GATE_OUT = ROOT / "reports/TRACKA_A2_GATE_VS_A1B.json"
CFG = _read_eval_config(ROOT / "configs/trackA_prune.yaml")
VARS = list((CFG.get("gate") or {}).get("variables") or ["t2m", "u10", "v10", "tp"])
LEADS = list((CFG.get("gate") or {}).get("leads_hours") or [24, 48])


class TruthCache:
    def __init__(self) -> None:
        self._ds: dict[str, xr.Dataset] = {}
        self._da: dict[str, xr.DataArray] = {}

    def open_var(self, var: str, year: int = 2023) -> xr.DataArray:
        key = f"{var}_{year}"
        if key in self._da:
            return self._da[key]
        path = era5_gcs_path(var, year)
        print(f"opening {path} opts={GCS_OPTS}", flush=True)
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


def score_dir(forecast_dir: Path, scorecard_out: Path, *, phase: str, checkpoint: str) -> Path:
    paths = sorted(forecast_dir.glob("*_00Z.nc"))
    if not paths:
        raise SystemExit(f"no forecasts in {forecast_dir}")
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
                        print(f"{init} +{lead}h {var} rmse={row['variables'][var].get('rmse')}", flush=True)
                    results.append(row)
            finally:
                fc.close()
    finally:
        cache.close()

    card = {
        "phase": phase,
        "status": "candidate",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checkpoint": checkpoint,
        "domain": "africa",
        "variables": VARS,
        "leads_hours": LEADS,
        "truth_source": "gcs",
        "inits_scored": [p.name.split("_")[0] for p in paths],
        "forecasts": [str(p) for p in paths],
        "results": results,
    }
    write_phase0_scorecard(card, scorecard_out)
    print(f"wrote {scorecard_out}", flush=True)
    return scorecard_out


def main() -> int:
    if not any(A1B_DIR.glob("*_00Z.nc")):
        raise SystemExit(f"missing A1b forecasts under {A1B_DIR}")
    need_score = True
    if A1B_CARD.is_file():
        newest_fc = max(p.stat().st_mtime for p in A1B_DIR.glob("*_00Z.nc"))
        need_score = newest_fc > A1B_CARD.stat().st_mtime
    if need_score:
        score_dir(
            A1B_DIR,
            A1B_CARD,
            phase="trackA_a1b_gt_coarsen",
            checkpoint="models/teacher_gt_coarsened.ckpt",
        )
    else:
        print(f"reuse {A1B_CARD}", flush=True)

    if not PRUNE_CARD.is_file():
        raise SystemExit(f"missing prune scorecard {PRUNE_CARD}")

    gate_cfg = dict(CFG)
    gate = dict(gate_cfg.get("gate") or {})
    gate["baseline_scorecard"] = str(A1B_CARD.relative_to(ROOT)).replace("\\", "/")
    gate_cfg["gate"] = gate

    report = run_coarsen_gate(PRUNE_CARD, baseline_scorecard=A1B_CARD, config=gate_cfg)
    report["step"] = "prune_vs_a1b"
    write_gate_report(report, GATE_OUT)
    n_ok = sum(1 for c in report["checks"] if c.get("passed"))
    print(f"GATE passed={report['passed']} ({n_ok}/{len(report['checks'])}) -> {GATE_OUT}", flush=True)
    for c in report["checks"]:
        if "error" in c:
            print(
                f"  FAIL {c.get('init_date')} +{c.get('lead_hours')}h {c.get('variable')}: {c.get('error')}",
                flush=True,
            )
            continue
        print(
            f"  {c['init_date']} +{c['lead_hours']}h {c['variable']}: "
            f"deg={c.get('degradation_pct'):.2f}% pass={c.get('passed')} "
            f"base={c.get('baseline_rmse'):.4g} cand={c.get('candidate_rmse'):.4g}",
            flush=True,
        )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

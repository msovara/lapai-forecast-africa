#!/usr/bin/env python3
"""Laptop GCS rescore of A1 vs A2 FULL on the eval.yaml Africa box (lon -20:55).

Does not rerun 240h forecasts. Does not touch smoke TRACKA_A2_GATE.json or
trackA_prune/forecasts (without _full). Backs up ARCO mixed FULL reports
before overwrite. Both sides vs gs://code4earth/era5 (ADC).
"""
from __future__ import annotations

import shutil
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
    _eval_africa_box,
    _resolve_fc_var,
    _score_point,
    forecast_step_and_valid_time,
    write_phase0_scorecard,
)
from evaluation.run_scorecard import _read_eval_config
from evaluation.trackA_gate import compare_scorecards, write_gate_report

ROOT = Path(__file__).resolve().parents[1]
A1_DIR = ROOT / "data/processed/trackA_a1_windows/forecasts"
A2_DIR = ROOT / "data/processed/trackA_prune_full/forecasts"
A1_FALLBACK = ROOT / "data/processed/trackA/forecasts"
A1_OUT = ROOT / "reports/TRACKA_COARSEN_SCORECARD_GCS_RESCORE.json"
A2_OUT = ROOT / "reports/TRACKA_PRUNE_FULL_SCORECARD.json"
A2_BACKUP = ROOT / "reports/TRACKA_PRUNE_FULL_SCORECARD_arco.json"
GATE_OUT = ROOT / "reports/TRACKA_A2_GATE_FULL.json"
GATE_BACKUP = ROOT / "reports/TRACKA_A2_GATE_FULL_arco.json"
SMOKE_GATE = ROOT / "reports/TRACKA_A2_GATE.json"
SMOKE_FC = ROOT / "data/processed/trackA_prune/forecasts"
CFG = _read_eval_config(ROOT / "configs/trackA_prune_full.yaml")
VARS = list((CFG.get("gate") or {}).get("variables") or ["t2m", "u10", "v10", "tp"])
LEADS = list((CFG.get("gate") or {}).get("leads_hours") or [24, 48])
EXPECT = ["20230101", "20230108", "20230115", "20230122", "20230129"]
TRUTH_BUCKET = "gs://code4earth/era5"
DOMAIN_BOX = {"lat": [-40.0, 40.0], "lon": [-20.0, 55.0]}


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


def _backup_if_needed(src: Path, dest: Path) -> None:
    if not src.exists():
        return
    if dest.exists():
        print(f"keep existing backup {dest}", flush=True)
        return
    shutil.copy2(src, dest)
    print(f"backed up {src.name} -> {dest.name}", flush=True)


def _discover(dir_path: Path) -> list[Path]:
    paths = sorted(dir_path.glob("*_00Z.nc"))
    have = [p.name.split("_")[0] for p in paths]
    if have != EXPECT:
        raise SystemExit(f"expected inits {EXPECT} in {dir_path}, found {have}")
    for p in paths:
        if p.stat().st_size < 1_000_000:
            raise SystemExit(f"forecast looks corrupt (too small): {p} {p.stat().st_size}B")
    return paths


def _describe_fc(path: Path, label: str) -> None:
    ds = xr.open_dataset(path)
    try:
        lat = ds.latitude.values
        lon = ds.longitude.values
        print(
            f"{label} {path.name} nlat={lat.size} nlon={lon.size} "
            f"lat=[{float(lat.min()):.1f},{float(lat.max()):.1f}] "
            f"lon=[{float(lon.min()):.1f},{float(lon.max()):.1f}] "
            f"time={int(ds.sizes['time'])}",
            flush=True,
        )
    finally:
        ds.close()


def _score_dir(paths: list[Path], cache: TruthCache) -> list[dict]:
    results = []
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
                pred0 = fc[list(fc.data_vars)[0]].isel(time=step)
                row["pred_grid"] = {
                    "nlat": int(pred0.latitude.size),
                    "nlon": int(pred0.longitude.size),
                    "lon_min": float(pred0.longitude.min()),
                    "lon_max": float(pred0.longitude.max()),
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
                    print(
                        f"{path.parent.parent.name} {init} +{lead}h {var} "
                        f"rmse={row['variables'][var].get('rmse')}",
                        flush=True,
                    )
                results.append(row)
        finally:
            fc.close()
    return results


def _card(phase: str, checkpoint: str, paths: list[Path], results: list[dict]) -> dict:
    box = _eval_africa_box()
    domain_box = {"lat": list(box[0]), "lon": list(box[1])} if box else DOMAIN_BOX
    return {
        "phase": phase,
        "status": "candidate" if "prune" in phase else "baseline_rescored",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checkpoint": checkpoint,
        "domain": "africa",
        "domain_box": domain_box,
        "eval_grid": "321x301",
        "variables": VARS,
        "leads_hours": LEADS,
        "truth_source": "gcs",
        "truth_bucket": TRUTH_BUCKET,
        "inits_scored": [p.name.split("_")[0] for p in paths],
        "forecasts": [str(p) for p in paths],
        "results": results,
    }


def main() -> int:
    if not SMOKE_GATE.exists():
        print(f"warning: smoke gate missing at {SMOKE_GATE}", flush=True)
    print(f"smoke forecasts left untouched: {SMOKE_FC}", flush=True)
    _backup_if_needed(A2_OUT, A2_BACKUP)
    _backup_if_needed(GATE_OUT, GATE_BACKUP)

    a1_dir = A1_DIR if A1_DIR.is_dir() else A1_FALLBACK
    a1_paths = _discover(a1_dir)
    a2_paths = _discover(A2_DIR)
    _describe_fc(a1_paths[0], "A1")
    _describe_fc(a2_paths[0], "A2")
    box = _eval_africa_box()
    print(f"eval.yaml africa box lat={box[0]} lon={box[1]}", flush=True)
    print(f"GCS_OPTS={GCS_OPTS} bucket={TRUTH_BUCKET}", flush=True)

    cache = TruthCache()
    try:
        a1_results = _score_dir(a1_paths, cache)
        a2_results = _score_dir(a2_paths, cache)
    finally:
        cache.close()

    a1_card = _card(
        "trackA_coarsen_extend_15k_gcs_rescore",
        "models/trackA_coarsen_full_extend_runs/.../inference-last.ckpt",
        a1_paths,
        a1_results,
    )
    a2_card = _card(
        "trackA_a2_prune_round1_full",
        "models/teacher_pruned_full.ckpt",
        a2_paths,
        a2_results,
    )
    write_phase0_scorecard(a1_card, A1_OUT)
    write_phase0_scorecard(a2_card, A2_OUT)
    print(f"wrote {A1_OUT}", flush=True)
    print(f"wrote {A2_OUT}", flush=True)

    gate_cfg = CFG.get("gate") or {}
    report = compare_scorecards(
        a1_card,
        a2_card,
        variables=VARS,
        leads_hours=LEADS,
        max_rmse_degradation_pct=float(gate_cfg.get("max_rmse_degradation_pct", 3.0)),
    )
    report["step"] = "prune_full"
    report["truth_source"] = "gcs"
    report["truth_bucket"] = TRUTH_BUCKET
    report["domain"] = "africa"
    report["domain_box"] = a2_card["domain_box"]
    report["eval_grid"] = "321x301"
    report["baseline_scorecard_path"] = str(A1_OUT)
    report["candidate_scorecard_path"] = str(A2_OUT)
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

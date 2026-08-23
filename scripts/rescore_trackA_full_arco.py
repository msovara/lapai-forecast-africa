#!/usr/bin/env python3
"""Rescore Track A FULL forecasts vs public ARCO ERA5 (anonymous GCS).

Does not rerun forecasts. Writes:
  reports/TRACKA_PRUNE_FULL_SCORECARD.json
  reports/TRACKA_A2_GATE_FULL.json
Leaves smoke TRACKA_A2_GATE.json / trackA_prune/forecasts untouched.

Team bucket gs://code4earth/era5 is private (needs ADC). WeatherBench2
1959-2023_01_10 ends 2023-01-10 so it cannot cover inits 15/22/29 Jan.
Public ARCO ERA5 0.25° hourly covers those dates.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "dummy")
os.environ.setdefault("LAPAI_GCS_TOKEN", "anon")

import numpy as np
import xarray as xr

from evaluation.gcs_era5_truth import normalize_lon, subset_africa
from evaluation.phase0_scorecard import (
    _resolve_fc_var,
    _score_point,
    forecast_step_and_valid_time,
    write_phase0_scorecard,
)
from evaluation.run_scorecard import _read_eval_config
from evaluation.trackA_gate import run_coarsen_gate, write_gate_report

ROOT = Path("/local/Mthetho/lapai-forecast")
FORECAST_DIR = ROOT / "data/processed/trackA_prune_full/forecasts"
SCORECARD = ROOT / "reports/TRACKA_PRUNE_FULL_SCORECARD.json"
GATE_OUT = ROOT / "reports/TRACKA_A2_GATE_FULL.json"
CFG_PATH = ROOT / "configs/trackA_prune_full.yaml"

ARCO_PATH = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ARCO_VARS = {
    "t2m": "2m_temperature",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "tp": "total_precipitation",
}
GCS_OPTS = {"token": "anon"}


class ArcoTruth:
    def __init__(self) -> None:
        print(f"opening {ARCO_PATH} token=anon", flush=True)
        self.ds = xr.open_zarr(ARCO_PATH, storage_options=GCS_OPTS, consolidated=True)
        self._cubes: dict[str, xr.DataArray] = {}

    def prefetch(self, times: list[np.datetime64]) -> None:
        uniq = sorted({np.datetime64(t, "ns") for t in times})
        print(f"prefetch {len(uniq)} valid times x {len(ARCO_VARS)} vars", flush=True)
        for var, name in ARCO_VARS.items():
            print(f"  loading {var} <- {name}", flush=True)
            da = self.ds[name]
            for drop in ("level", "number", "surface", "expver"):
                if drop in da.dims:
                    da = da.isel({drop: 0})
            cube = da.sel(time=uniq, method="nearest")
            if float(cube.latitude[0]) > float(cube.latitude[-1]):
                cube = cube.sel(latitude=slice(40, -40))
            else:
                cube = cube.sel(latitude=slice(-40, 40))
            cube = cube.load()
            cube = subset_africa(normalize_lon(cube))
            finite = float(np.isfinite(np.asarray(cube.values)).mean())
            print(
                f"    {var} sizes={dict(cube.sizes)} finite={finite:.3f} "
                f"t0={cube.time.values[0]} tN={cube.time.values[-1]}",
                flush=True,
            )
            if finite < 0.5:
                raise RuntimeError(f"ARCO {var} mostly non-finite; cannot score")
            self._cubes[var] = cube

    def at(self, var: str, when: np.datetime64) -> xr.DataArray:
        cube = self._cubes[var]
        sl = cube.sel(time=np.datetime64(when, "ns"), method="nearest")
        return sl.squeeze(drop=True)

    def close(self) -> None:
        self.ds.close()


def _scan_forecasts(paths: list[Path], leads: list[int], variables: list[str]):
    needed_times: list[np.datetime64] = []
    jobs = []
    for path in paths:
        init = path.name.split("_")[0]
        size = path.stat().st_size
        if size < 1_000_000:
            raise SystemExit(f"forecast looks corrupt (too small): {path} {size}B")
        fc = xr.open_dataset(path)
        try:
            print(f"forecast {path.name} size={size} times={fc.time.size} vars={list(fc.data_vars)}", flush=True)
            for lead in leads:
                step, vt = forecast_step_and_valid_time(fc, lead)
                needed_times.append(vt)
                jobs.append((path, init, lead, step, vt))
        finally:
            fc.close()
    return jobs, needed_times


def main() -> int:
    cfg = _read_eval_config(CFG_PATH)
    variables = list((cfg.get("gate") or {}).get("variables") or ["t2m", "u10", "v10", "tp"])
    leads = list((cfg.get("gate") or {}).get("leads_hours") or [24, 48])

    paths = sorted(FORECAST_DIR.glob("*_00Z.nc"))
    expect = ["20230101", "20230108", "20230115", "20230122", "20230129"]
    have = [p.name.split("_")[0] for p in paths]
    if have != expect:
        raise SystemExit(f"expected inits {expect}, found {have}")

    jobs, needed_times = _scan_forecasts(paths, leads, variables)
    cache = ArcoTruth()
    results = []
    try:
        cache.prefetch(needed_times)
        for path, init, lead, step, vt in jobs:
            fc = xr.open_dataset(path)
            try:
                row = {
                    "forecast": str(path),
                    "init_date": init,
                    "lead_hours": lead,
                    "valid_time": str(vt),
                    "variables": {},
                }
                for var in variables:
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
                        f"{init} +{lead}h {var} rmse={row['variables'][var].get('rmse')}",
                        flush=True,
                    )
                results.append(row)
            finally:
                fc.close()
    finally:
        cache.close()

    card = {
        "phase": "trackA_a2_prune_round1_full",
        "status": "candidate",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checkpoint": "models/teacher_pruned_full.ckpt",
        "domain": "africa",
        "variables": variables,
        "leads_hours": leads,
        "truth_source": "arco_era5_anon",
        "truth_uri": ARCO_PATH,
        "inits_scored": [p.name.split("_")[0] for p in paths],
        "forecasts": [str(p) for p in paths],
        "results": results,
    }
    write_phase0_scorecard(card, SCORECARD)
    print(f"wrote {SCORECARD}", flush=True)

    report = run_coarsen_gate(SCORECARD, config=cfg)
    report["step"] = "prune_full"
    report["truth_source"] = "arco_era5_anon"
    report["truth_uri"] = ARCO_PATH
    write_gate_report(report, GATE_OUT)
    n_ok = sum(1 for c in report["checks"] if c.get("passed"))
    print(f"GATE passed={report['passed']} ({n_ok}/{len(report['checks'])}) -> {GATE_OUT}", flush=True)
    for c in report["checks"]:
        if "error" in c:
            print(f"  FAIL {c}", flush=True)
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

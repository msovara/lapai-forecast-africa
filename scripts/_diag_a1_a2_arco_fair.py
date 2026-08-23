#!/usr/bin/env python3
"""Same-truth ARCO rescore of A1 vs A2, plus A2 cropped to A1 lon box.

Does not overwrite smoke scorecards or TRACKA_COARSEN_SCORECARD.json.
CPU only. Writes:
  reports/TRACKA_COARSEN_SCORECARD_ARCO.json
  reports/TRACKA_PRUNE_FULL_SCORECARD_ARCO_LON55.json
  reports/TRACKA_A2_GATE_ARCO_FAIR.json
  reports/TRACKA_A2_GATE_ARCO_FAIR_LON55.json
"""
from __future__ import annotations

import json
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
from evaluation.trackA_gate import compare_scorecards, write_gate_report

ROOT = Path("/local/Mthetho/lapai-forecast")
A1_DIR = ROOT / "data/processed/trackA_a1_windows/forecasts"
A2_DIR = ROOT / "data/processed/trackA_prune_full/forecasts"
A1_OUT = ROOT / "reports/TRACKA_COARSEN_SCORECARD_ARCO.json"
A2_CROP_OUT = ROOT / "reports/TRACKA_PRUNE_FULL_SCORECARD_ARCO_LON55.json"
GATE_MIXED_DOMAIN = ROOT / "reports/TRACKA_A2_GATE_ARCO_FAIR.json"
GATE_LON55 = ROOT / "reports/TRACKA_A2_GATE_ARCO_FAIR_LON55.json"

ARCO_PATH = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ARCO_VARS = {
    "t2m": "2m_temperature",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "tp": "total_precipitation",
}
GCS_OPTS = {"token": "anon"}
VARIABLES = ["t2m", "u10", "v10", "tp"]
LEADS = [24, 48]
A1_LON = slice(-20, 55)


class ArcoTruth:
    def __init__(self) -> None:
        print(f"opening {ARCO_PATH} token=anon", flush=True)
        self.ds = xr.open_zarr(ARCO_PATH, storage_options=GCS_OPTS, consolidated=True)
        self._cubes: dict[str, xr.DataArray] = {}

    def describe(self) -> None:
        print("ARCO dims", dict(self.ds.sizes), flush=True)
        for key in ARCO_VARS.values():
            da = self.ds[key]
            print(
                "ARCO",
                key,
                "dims",
                da.dims,
                "sizes",
                dict(da.sizes),
                "units",
                da.attrs.get("units"),
                "dtype",
                da.dtype,
                flush=True,
            )
        lat = np.asarray(self.ds.latitude.values)
        lon = np.asarray(self.ds.longitude.values)
        print(
            "ARCO lat",
            lat.size,
            float(lat.min()),
            float(lat.max()),
            "dlon_sample",
            float(np.diff(lon[:3])[0]) if lon.size > 2 else None,
            "lon0",
            float(lon[0]),
            "lonN",
            float(lon[-1]),
            flush=True,
        )
        t = self.ds.time.values
        print("ARCO time", t[0], "->", t[-1], "n", t.size, flush=True)

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
            tsel = cube.time.values
            print(
                f"    {var} sizes={dict(cube.sizes)} finite={float(np.isfinite(np.asarray(cube.values)).mean()):.3f} "
                f"min={float(np.nanmin(cube.values)):.6g} max={float(np.nanmax(cube.values)):.6g} "
                f"units={cube.attrs.get('units')} t_req0={uniq[0]} t_got0={tsel[0]} dt_h="
                f"{(np.datetime64(tsel[0], 'ns') - np.datetime64(uniq[0], 'ns')) / np.timedelta64(1, 'h')}",
                flush=True,
            )
            self._cubes[var] = cube

    def at(self, var: str, when: np.datetime64) -> xr.DataArray:
        cube = self._cubes[var]
        sl = cube.sel(time=np.datetime64(when, "ns"), method="nearest")
        return sl.squeeze(drop=True)

    def close(self) -> None:
        self.ds.close()


def crop_lon(da: xr.DataArray, lon: slice) -> xr.DataArray:
    if "longitude" not in da.dims:
        return da
    return da.sel(longitude=lon)


def score_dir(paths: list[Path], cache: ArcoTruth, *, lon_crop: slice | None, phase: str, ckpt: str) -> dict:
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
                for var in VARIABLES:
                    fc_name = _resolve_fc_var(fc, var)
                    if fc_name is None:
                        row["variables"][var] = {"error": "missing in forecast"}
                        continue
                    pred = fc[fc_name].isel(time=step).assign_coords(name=var)
                    if lon_crop is not None:
                        pred = crop_lon(pred, lon_crop)
                    truth = cache.at(var, vt)
                    if lon_crop is not None:
                        truth = crop_lon(truth, lon_crop)
                    scores = _score_point(pred, truth)
                    row["variables"][var] = {
                        k: (None if isinstance(v, float) and not np.isfinite(v) else float(v))
                        for k, v in scores.items()
                    }
                    print(
                        f"{phase} {init} +{lead}h {var} lon={getattr(lon_crop, 'stop', 'native')} "
                        f"rmse={row['variables'][var].get('rmse')}",
                        flush=True,
                    )
                results.append(row)
        finally:
            fc.close()
    return {
        "phase": phase,
        "status": "candidate",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checkpoint": ckpt,
        "domain": "africa" if lon_crop is None else "africa_lon-20_55",
        "variables": VARIABLES,
        "leads_hours": LEADS,
        "truth_source": "arco_era5_anon",
        "truth_uri": ARCO_PATH,
        "lon_crop": None if lon_crop is None else [lon_crop.start, lon_crop.stop],
        "inits_scored": [p.name.split("_")[0] for p in paths],
        "forecasts": [str(p) for p in paths],
        "results": results,
    }


def summarize_gate(report: dict, label: str) -> None:
    n_ok = sum(1 for c in report["checks"] if c.get("passed"))
    print(f"{label} passed={report['passed']} ({n_ok}/{len(report['checks'])})", flush=True)
    for c in report["checks"]:
        print(
            f"  {c.get('init_date')} +{c.get('lead_hours')}h {c.get('variable')}: "
            f"deg={c.get('degradation_pct')} pass={c.get('passed')} "
            f"base={c.get('baseline_rmse')} cand={c.get('candidate_rmse')}",
            flush=True,
        )


def main() -> int:
    a1 = sorted(A1_DIR.glob("*_00Z.nc"))
    a2 = sorted(A2_DIR.glob("*_00Z.nc"))
    expect = ["20230101", "20230108", "20230115", "20230122", "20230129"]
    if [p.name.split("_")[0] for p in a1] != expect:
        raise SystemExit(f"A1 missing/unexpected: {[p.name for p in a1]}")
    if [p.name.split("_")[0] for p in a2] != expect:
        raise SystemExit(f"A2 missing/unexpected: {[p.name for p in a2]}")
    for p in a1 + a2:
        if p.stat().st_size < 1_000_000:
            raise SystemExit(f"corrupt {p} {p.stat().st_size}B")

    needed = []
    for path in a1 + a2:
        fc = xr.open_dataset(path)
        try:
            for lead in LEADS:
                _, vt = forecast_step_and_valid_time(fc, lead)
                needed.append(vt)
        finally:
            fc.close()

    cache = ArcoTruth()
    try:
        cache.describe()
        cache.prefetch(needed)
        a1_card = score_dir(a1, cache, lon_crop=None, phase="a1_coarsen_arco", ckpt="a1_windows_nc")
        a2_native = score_dir(a2, cache, lon_crop=None, phase="a2_full_arco", ckpt="teacher_pruned_full.ckpt")
        a2_crop = score_dir(a2, cache, lon_crop=A1_LON, phase="a2_full_arco_lon55", ckpt="teacher_pruned_full.ckpt")
    finally:
        cache.close()

    write_phase0_scorecard(a1_card, A1_OUT)
    write_phase0_scorecard(a2_crop, A2_CROP_OUT)
    print(f"wrote {A1_OUT}", flush=True)
    print(f"wrote {A2_CROP_OUT}", flush=True)
    print("A2 native ARCO scores kept in-memory only (existing FULL scorecard already has -20:70)", flush=True)

    gate_native = compare_scorecards(
        a1_card,
        a2_native,
        variables=VARIABLES,
        leads_hours=LEADS,
        max_rmse_degradation_pct=3.0,
    )
    gate_native["step"] = "prune_full_arco_fair_mixed_domain"
    gate_native["truth_source"] = "arco_era5_anon"
    gate_native["note"] = "A1 lon -20:55 vs A2 lon -20:70, both ARCO 0.25deg nearest"
    write_gate_report(gate_native, GATE_MIXED_DOMAIN)
    summarize_gate(gate_native, "GATE mixed-domain ARCO")

    gate_crop = compare_scorecards(
        a1_card,
        a2_crop,
        variables=VARIABLES,
        leads_hours=LEADS,
        max_rmse_degradation_pct=3.0,
    )
    gate_crop["step"] = "prune_full_arco_fair_lon55"
    gate_crop["truth_source"] = "arco_era5_anon"
    gate_crop["note"] = "both cropped/native to lon -20:55, both ARCO 0.25deg nearest"
    write_gate_report(gate_crop, GATE_LON55)
    summarize_gate(gate_crop, "GATE lon-20:55 ARCO")

    # Compare A1 GCS (existing coarsen card) vs A1 ARCO to quantify truth shift.
    gcs_path = ROOT / "reports/TRACKA_COARSEN_SCORECARD.json"
    if gcs_path.exists():
        gcs = json.loads(gcs_path.read_text())
        print("\nA1 GCS vs A1 ARCO (truth-source delta, same forecast files):", flush=True)
        gcs_idx = {(r["init_date"], int(r["lead_hours"])): r["variables"] for r in gcs.get("results", [])}
        for row in a1_card["results"]:
            key = (row["init_date"], int(row["lead_hours"]))
            g = gcs_idx.get(key) or {}
            for var in VARIABLES:
                g_rmse = (g.get(var) or {}).get("rmse")
                a_rmse = (row["variables"].get(var) or {}).get("rmse")
                if g_rmse and a_rmse:
                    pct = 100.0 * (a_rmse - g_rmse) / g_rmse
                    print(
                        f"  {key[0]} +{key[1]}h {var}: gcs={g_rmse:.6g} arco={a_rmse:.6g} arco_vs_gcs={pct:.2f}%",
                        flush=True,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

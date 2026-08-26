#!/usr/bin/env python3
"""Rescore Track B student held-out NCs vs public ARCO ERA5 (no CDS / no ADC).

Used when ``gs://code4earth/era5`` ADC is unavailable on Cassava. Scores student
and K1 teacher fields at the same valid times so degradation is apples-to-apples.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "dummy")

from evaluation.gcs_era5_truth import normalize_lon, subset_africa  # noqa: E402
from evaluation.phase0_scorecard import (  # noqa: E402
    _resolve_fc_var,
    _score_point,
    forecast_step_and_valid_time,
)
from evaluation.trackB_held_out_jan2023 import (  # noqa: E402
    DEFAULT_INITS,
    STUDENT_SCORE_VARS,
    _aggregate_student,
    _degradation_pct,
)
from utils.era5_ondisk_ic import ARCO_DEFAULT  # noqa: E402

ARCO_VARS = {
    "t2m": "2m_temperature",
    "tp": "total_precipitation",
}
GCS_OPTS = {"token": "anon"}


class ArcoSurfaceTruth:
    def __init__(self, path: str = ARCO_DEFAULT) -> None:
        print(f"[arco-truth] opening {path}", flush=True)
        self.ds = xr.open_zarr(path, storage_options=GCS_OPTS, consolidated=True)
        self._cache: dict[str, xr.DataArray] = {}

    def at(self, var: str, when: np.datetime64) -> xr.DataArray:
        name = ARCO_VARS[var]
        key = f"{var}_{np.datetime64(when, 'h')}"
        if key in self._cache:
            return self._cache[key]
        da = self.ds[name]
        for drop in ("level", "number", "surface", "expver"):
            if drop in da.dims and da.sizes.get(drop, 0) == 1:
                da = da.isel({drop: 0})
        sl = da.sel(time=np.datetime64(when, "ns"), method="nearest")
        sl = sl.load().squeeze(drop=True)
        out = subset_africa(normalize_lon(sl))
        self._cache[key] = out
        return out

    def close(self) -> None:
        self.ds.close()


def _stud_nc(fc_dir: Path, init: str, lead: int) -> Path:
    if lead == 6:
        return fc_dir / f"{init}_00Z.nc"
    return fc_dir / f"{init}_00Z_L{lead:03d}h.nc"


def rescore(
    *,
    student_fc_dir: Path,
    teacher_fc_dir: Path,
    leads: tuple[int, ...],
    inits: tuple[str, ...],
    ckpt: str,
    student_json: Path | None,
) -> dict[str, Any]:
    truth = ArcoSurfaceTruth()
    results: list[dict[str, Any]] = []
    try:
        for init in inits:
            teach_path = teacher_fc_dir / f"{init}_00Z.nc"
            if not teach_path.is_file():
                results.append({"init_date": init, "error": f"missing teacher {teach_path.name}"})
                continue
            tfc = xr.open_dataset(teach_path)
            try:
                for lead in leads:
                    stud_path = _stud_nc(student_fc_dir, init, lead)
                    if not stud_path.is_file():
                        results.append(
                            {
                                "init_date": init,
                                "lead_hours": lead,
                                "error": f"missing student nc {stud_path.name}",
                            }
                        )
                        continue
                    sfc = xr.open_dataset(stud_path)
                    try:
                        if "time" in sfc.dims and sfc.sizes["time"] == 1:
                            step_s = 0
                            vt = sfc["time"].values[0]
                        else:
                            step_s, vt = forecast_step_and_valid_time(sfc, lead)
                        step_t, _ = forecast_step_and_valid_time(tfc, lead)
                        row: dict[str, Any] = {
                            "model": "student",
                            "init_date": init,
                            "lead_hours": lead,
                            "valid_time": str(vt),
                            "ckpt": ckpt,
                            "protocol": "analysis_forced_6h_step",
                            "truth_source": ARCO_DEFAULT,
                            "forecast_nc": str(stud_path),
                            "variables": {},
                        }
                        if student_json is not None and student_json.is_file():
                            # Preserve IC meta from inference JSON when present.
                            prev = json.loads(student_json.read_text(encoding="utf-8"))
                            for pr in prev.get("student_results") or []:
                                if pr.get("init_date") == init and int(pr.get("lead_hours") or -1) == lead:
                                    if "state_meta" in pr:
                                        row["state_meta"] = pr["state_meta"]
                                    break
                        for var in STUDENT_SCORE_VARS:
                            s_name = _resolve_fc_var(sfc, var)
                            t_name = _resolve_fc_var(tfc, var)
                            if s_name is None or t_name is None:
                                row["variables"][var] = {"error": "missing var"}
                                continue
                            pred = sfc[s_name].isel(time=step_s).assign_coords(name=var)
                            teach = tfc[t_name].isel(time=step_t).assign_coords(name=var)
                            # Field RMSE student vs teacher (no truth).
                            p = np.asarray(pred.values, dtype=np.float64)
                            t = np.asarray(teach.values, dtype=np.float64)
                            lat = np.asarray(pred.latitude.values, dtype=np.float64)
                            from lapai_inference.cache_schema import cosine_latitude_weights

                            w = np.asarray(cosine_latitude_weights(lat), dtype=np.float64).reshape(-1, 1)
                            mask = np.isfinite(p) & np.isfinite(t)
                            svt = float(
                                np.sqrt(
                                    np.sum(w * (p - t) ** 2 * mask) / max(np.sum(w * mask), 1e-12)
                                )
                            )
                            era5 = truth.at(var, np.datetime64(vt))
                            stud_scores = _score_point(pred, era5)
                            teach_scores = _score_point(teach, era5)
                            s_rmse = float(stud_scores["rmse"])
                            t_rmse = float(teach_scores["rmse"])
                            cell: dict[str, Any] = {
                                "student_rmse_vs_teacher_field": svt,
                                "student_rmse_vs_era5": s_rmse,
                                "student_acc": float(stud_scores.get("acc", float("nan"))),
                                "teacher_rmse_vs_era5": t_rmse,
                                "teacher_acc": float(teach_scores.get("acc", float("nan"))),
                                "degradation_pct_vs_teacher": _degradation_pct(s_rmse, t_rmse),
                            }
                            if var == "tp" and "pod_1mm" in stud_scores:
                                cell["student_pod_1mm"] = float(stud_scores["pod_1mm"])
                            if var == "tp" and "pod_1mm" in teach_scores:
                                cell["teacher_pod_1mm"] = float(teach_scores["pod_1mm"])
                            row["variables"][var] = cell
                            print(f"[arco-rescore] {init} +{lead}h {var} {cell}", flush=True)
                        results.append(row)
                    finally:
                        sfc.close()
            finally:
                tfc.close()
    finally:
        truth.close()

    return {
        "protocol": "trackB_held_out_jan2023",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_student": ckpt,
        "domain": "africa",
        "truth_source": ARCO_DEFAULT,
        "ic_source": "arco_era5",
        "student_leads_hours": list(leads),
        "student_variables_scored": list(STUDENT_SCORE_VARS),
        "caveats": [
            "Analysis-forced multi-lead only (not free-run).",
            "Student IC from public ARCO ERA5 pressure levels (no CDS).",
            "Surface truth for scoring also from public ARCO ERA5 (code4earth ADC unavailable on Cassava).",
            "tp out-of-scope for Track B promotion decisions; t2m is primary.",
        ],
        "student_results": results,
        "aggregate": {"student": _aggregate_student(results, leads=leads)},
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--student_fc_dir",
        type=Path,
        default=ROOT / "data/processed/trackB_student_heldout_v5/forecasts",
    )
    p.add_argument(
        "--teacher_fc_dir",
        type=Path,
        default=ROOT / "data/processed/trackA_prune_k1_a1bplus/forecasts",
    )
    p.add_argument("--leads", default="6,24")
    p.add_argument("--ckpt", default="models/student_global_stable_v5.ckpt")
    p.add_argument(
        "--student_json",
        type=Path,
        default=ROOT / "reports/TRACKB_HELD_OUT_JAN2023_V5.json",
        help="Inference JSON to copy state_meta from",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports/TRACKB_HELD_OUT_JAN2023_V5.json",
    )
    p.add_argument(
        "--canonical_out",
        type=Path,
        default=ROOT / "reports/TRACKB_HELD_OUT_JAN2023.json",
        help="Also write/update canonical held-out report path",
    )
    args = p.parse_args()
    leads = tuple(int(x) for x in args.leads.split(",") if x.strip())
    doc = rescore(
        student_fc_dir=args.student_fc_dir,
        teacher_fc_dir=args.teacher_fc_dir,
        leads=leads,
        inits=DEFAULT_INITS,
        ckpt=args.ckpt,
        student_json=args.student_json,
    )
    # Merge teacher multi-lead table from existing canonical report if present.
    if args.canonical_out.is_file():
        prev = json.loads(args.canonical_out.read_text(encoding="utf-8"))
        if prev.get("teacher_results"):
            doc["teacher_results"] = prev["teacher_results"]
            doc["baseline"] = prev.get("baseline")
            if "aggregate" in prev and "teacher" in prev["aggregate"]:
                doc.setdefault("aggregate", {})["teacher"] = prev["aggregate"]["teacher"]
            doc["leads_hours"] = prev.get("leads_hours")
            doc["truth_source_teacher"] = prev.get("truth_source")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    if args.canonical_out:
        args.canonical_out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        print(f"wrote {args.canonical_out}", flush=True)
    print(json.dumps(doc.get("aggregate"), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

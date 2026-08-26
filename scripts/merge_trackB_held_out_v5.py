#!/usr/bin/env python3
"""Merge v5 Cassava student held-out JSON with teacher table; rescore vs GCS ERA5."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.phase0_scorecard import _resolve_fc_var, _score_point, forecast_step_and_valid_time
from evaluation.trackB_held_out_jan2023 import (
    STUDENT_SCORE_VARS,
    TruthCache,
    _aggregate_student,
    _degradation_pct,
)

TEACHER_JSON = ROOT / "reports/TRACKB_HELD_OUT_JAN2023.json"
STUDENT_JSON = ROOT / "reports/TRACKB_HELD_OUT_JAN2023_V5.json"
STUDENT_FC = ROOT / "data/processed/trackB_student_heldout_v5/forecasts"
K1_FC = ROOT / "data/processed/trackA_prune_k1_a1bplus/forecasts"
OUT = ROOT / "reports/TRACKB_HELD_OUT_JAN2023_V5_MERGED.json"
CKPT = "models/student_global_stable_v5.ckpt"


def _stud_nc(init: str, lead: int) -> Path:
    if lead == 6:
        return STUDENT_FC / f"{init}_00Z.nc"
    return STUDENT_FC / f"{init}_00Z_L{lead:03d}h.nc"


def main() -> None:
    teacher_doc = json.loads(TEACHER_JSON.read_text(encoding="utf-8"))
    student_doc = json.loads(STUDENT_JSON.read_text(encoding="utf-8"))
    cache = TruthCache()
    merged_student = []
    leads_seen: set[int] = set()
    try:
        for row in student_doc.get("student_results") or []:
            init = row.get("init_date")
            lead = int(row.get("lead_hours") or 6)
            if not init or row.get("error"):
                merged_student.append(row)
                continue
            stud_path = Path(row.get("forecast_nc") or _stud_nc(init, lead))
            teach_path = K1_FC / f"{init}_00Z.nc"
            if not stud_path.is_file() or not teach_path.is_file():
                print(f"[merge] skip {init} L{lead}: missing nc", flush=True)
                merged_student.append(row)
                continue
            sfc = xr.open_dataset(stud_path)
            tfc = xr.open_dataset(teach_path)
            try:
                # Student NC usually has a single time at valid_time.
                if "time" in sfc.dims and sfc.sizes["time"] == 1:
                    step_s = 0
                    vt = sfc["time"].values[0]
                else:
                    step_s, vt = forecast_step_and_valid_time(sfc, lead)
                step_t, _ = forecast_step_and_valid_time(tfc, lead)
                out_row = dict(row)
                out_row["model"] = "student_v5"
                out_row["ckpt"] = CKPT
                out_row["variables"] = {}
                leads_seen.add(lead)
                for var in STUDENT_SCORE_VARS:
                    s_name = _resolve_fc_var(sfc, var)
                    t_name = _resolve_fc_var(tfc, var)
                    if s_name is None or t_name is None:
                        out_row["variables"][var] = {"error": "missing var"}
                        continue
                    pred = sfc[s_name].isel(time=step_s).assign_coords(name=var)
                    teach = tfc[t_name].isel(time=step_t).assign_coords(name=var)
                    truth = cache.at(var, np.datetime64(vt))
                    stud_scores = _score_point(pred, truth)
                    teach_scores = _score_point(teach, truth)
                    s_rmse = float(stud_scores["rmse"])
                    t_rmse = float(teach_scores["rmse"])
                    prev = (row.get("variables") or {}).get(var) or {}
                    cell = {
                        "student_rmse_vs_era5": s_rmse,
                        "student_acc": float(stud_scores.get("acc", float("nan"))),
                        "teacher_rmse_vs_era5": t_rmse,
                        "teacher_acc": float(teach_scores.get("acc", float("nan"))),
                        "degradation_pct_vs_teacher": _degradation_pct(s_rmse, t_rmse),
                        "student_rmse_vs_teacher_field": prev.get(
                            "student_rmse_vs_teacher_field"
                        ),
                    }
                    if var == "tp" and "pod_1mm" in stud_scores:
                        cell["student_pod_1mm"] = float(stud_scores["pod_1mm"])
                    if var == "tp" and "pod_1mm" in teach_scores:
                        cell["teacher_pod_1mm"] = float(teach_scores["pod_1mm"])
                    out_row["variables"][var] = cell
                    print(f"[merge] {init} +{lead}h {var} {cell}", flush=True)
                out_row["forecast_nc"] = str(stud_path)
                out_row["valid_time"] = str(vt)
                merged_student.append(out_row)
            finally:
                sfc.close()
                tfc.close()
    finally:
        cache.close()

    leads_t = tuple(sorted(leads_seen)) or (6,)
    stud_agg = _aggregate_student(merged_student, leads=leads_t)
    out_doc = {
        "protocol": "trackB_held_out_jan2023",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": teacher_doc.get("baseline"),
        "candidate_student": CKPT,
        "baseline_v4_for_comparison": "models/student_global_stable_v4.ckpt",
        "domain": teacher_doc.get("domain", "africa"),
        "truth_source": "gcs://code4earth/era5",
        "leads_hours": list(leads_t),
        "teacher_results": teacher_doc.get("teacher_results"),
        "student_results": merged_student,
        "aggregate": {
            "teacher": (teacher_doc.get("aggregate") or {}).get("teacher"),
            "student": stud_agg,
            "student_6h": {k: v for k, v in stud_agg.items() if k in STUDENT_SCORE_VARS},
            "v4_student_6h_reference": {
                "note": "see reports/TRACKB_HELD_OUT_JAN2023_V4_MERGED.json aggregate.student_6h",
                "tp_acc_approx": 0.0,
                "tp_pod_1mm": 0.0,
                "tp_rmse": 0.000389,
                "t2m_deg_pct": 23.9,
            },
        },
        "protocol_notes": {
            "teacher_scored": "prior local GCS ADC (v3 report)",
            "student_inferred": "Cassava GPU1; analysis-forced CDS IC; v5 tp-recovery ckpt",
            "student_era5_rescored": "local GCS ADC after student NCs",
            "tp_decision": "OUT-OF-SCOPE for Cout=3 student head after one v5 attempt (ACC≈0, POD₁ₘₘ=0, field identically zero)",
            "multi_lead": "analysis-forced code path ready (--student_leads); +6h scored; +24h blocked on CDS 18Z auth on Cassava; free-run not supported",
        },
    }
    OUT.write_text(json.dumps(out_doc, indent=2), encoding="utf-8")
    print("wrote", OUT)
    print(json.dumps(out_doc["aggregate"], indent=2))


if __name__ == "__main__":
    main()

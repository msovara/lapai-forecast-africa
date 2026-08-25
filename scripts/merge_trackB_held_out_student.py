#!/usr/bin/env python3
"""Merge Cassava student held-out JSON with local teacher table; rescore student NCs vs GCS."""

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
STUDENT_JSON = ROOT / "reports/TRACKB_HELD_OUT_JAN2023_STUDENT.json"
STUDENT_FC = ROOT / "data/processed/trackB_student_heldout/forecasts"
K1_FC = ROOT / "data/processed/trackA_prune_k1_a1bplus/forecasts"
OUT = ROOT / "reports/TRACKB_HELD_OUT_JAN2023.json"


def main() -> None:
    teacher_doc = json.loads(TEACHER_JSON.read_text(encoding="utf-8"))
    student_doc = json.loads(STUDENT_JSON.read_text(encoding="utf-8"))
    cache = TruthCache()
    merged_student = []
    try:
        for row in student_doc.get("student_results") or []:
            init = row["init_date"]
            stud_path = STUDENT_FC / f"{init}_00Z.nc"
            teach_path = K1_FC / f"{init}_00Z.nc"
            if not stud_path.is_file() or not teach_path.is_file():
                merged_student.append(row)
                continue
            sfc = xr.open_dataset(stud_path)
            tfc = xr.open_dataset(teach_path)
            try:
                # Student NC has a single valid-time stamp at +6h.
                step_s, vt = forecast_step_and_valid_time(sfc, 6)
                step_t, _ = forecast_step_and_valid_time(tfc, 6)
                out_row = dict(row)
                out_row["variables"] = {}
                for var in STUDENT_SCORE_VARS:
                    s_name = _resolve_fc_var(sfc, var)
                    t_name = _resolve_fc_var(tfc, var)
                    if s_name is None or t_name is None:
                        out_row["variables"][var] = {"error": "missing var"}
                        continue
                    pred = sfc[s_name].isel(time=step_s).assign_coords(name=var)
                    teach = tfc[t_name].isel(time=step_t).assign_coords(name=var)
                    truth = cache.at(var, vt)
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
                    out_row["variables"][var] = cell
                    print(f"[merge] {init} +6h {var} {cell}", flush=True)
                out_row["forecast_nc"] = str(stud_path)
                out_row["valid_time"] = str(vt)
                merged_student.append(out_row)
            finally:
                sfc.close()
                tfc.close()
    finally:
        cache.close()

    teacher_doc["student_results"] = merged_student
    teacher_doc["candidate_student"] = "models/student_global_stable_v3.ckpt"
    teacher_doc["aggregate"]["student_6h"] = _aggregate_student(merged_student)
    teacher_doc["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    teacher_doc["protocol_notes"] = {
        "teacher_scored": "local GCS ADC",
        "student_inferred": "Cassava GPU1 lapai-anemoi; CDS IC cache; Africa regrid from 1deg",
        "student_era5_rescored": "local GCS ADC after scp of student NCs",
    }
    OUT.write_text(json.dumps(teacher_doc, indent=2), encoding="utf-8")
    print("wrote", OUT)
    print(json.dumps(teacher_doc["aggregate"], indent=2))


if __name__ == "__main__":
    main()

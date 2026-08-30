#!/usr/bin/env python3
"""Score AF-matched (and init) persistence baselines for African t2m.

Uses the same ARCO Africa + cosine-latitude RMSE/ACC/bias methodology as
``evaluation/trackB_t2m_expanded.py``. Student and K1 metrics are taken from
``reports/TRACKB_T2M_EXPANDED.json`` (no student re-inference).

Baselines
---------
* ``persistence_af``:  T_hat(valid) = T(IC) with IC = init+(L−6)h  (fair vs AF student)
* ``persistence_init``: T_hat(valid) = T(init 00Z)               (classic issued-time persist)
* ``climatology`` (optional): mean ERA5 at same MM-DD HH over --clim_years

Usage (Cassava)::

  cd /local/Mthetho/lapai-forecast
  source /local/Mthetho/envs/lapai-credit/bin/activate
  export MKL_INTERFACE_LAYER=GNU LAPAI_GCS_TOKEN=anon GOOGLE_CLOUD_PROJECT=dummy
  python -u evaluation/score_t2m_baselines.py --with_climatology
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "dummy")
os.environ.setdefault("LAPAI_GCS_TOKEN", "anon")

from evaluation.trackB_t2m_expanded import (  # noqa: E402
    ArcoSurfaceTruth,
    _score_t2m,
    season_of,
)
from utils.era5_ondisk_ic import ARCO_DEFAULT  # noqa: E402


def _parse_ic(ic_date: str, ic_time: str) -> np.datetime64:
    return np.datetime64(datetime.strptime(f"{ic_date}{ic_time}", "%Y%m%d%H%M"))


def _parse_valid(valid_time: str) -> np.datetime64:
    # "2023-01-01T06:00:00.000000"
    return np.datetime64(valid_time)


def _init_00z(init_date: str) -> np.datetime64:
    return np.datetime64(datetime.strptime(init_date + "0000", "%Y%m%d%H%M"))


def _shift_year(when: np.datetime64, year: int) -> np.datetime64:
    dt = when.astype("datetime64[s]").astype(datetime)
    # Feb 29 → Feb 28 in non-leap years
    try:
        return np.datetime64(dt.replace(year=year))
    except ValueError:
        return np.datetime64(dt.replace(year=year, day=28))


def score_baselines(
    *,
    expanded_json: Path,
    arco_path: str,
    with_climatology: bool,
    clim_years: tuple[int, ...],
) -> dict[str, Any]:
    blob = json.loads(expanded_json.read_text(encoding="utf-8"))
    student_rows = blob["student_results"]
    k1_rows = blob.get("k1_results") or []

    truth = ArcoSurfaceTruth(arco_path)
    clim_cache: dict[tuple[int, int, int], Any] = {}  # (mm,dd,hh) -> DataArray

    baseline_rows: list[dict[str, Any]] = []
    try:
        for i, r in enumerate(student_rows):
            init_date = r["init_date"]
            lead = int(r["lead_hours"])
            valid = _parse_valid(r["valid_time"])
            ic = _parse_ic(r["ic_date"], r["ic_time"])
            init0 = _init_00z(init_date)

            truth_v = truth.at("t2m", valid)
            pred_af = truth.at("t2m", ic)
            pred_init = truth.at("t2m", init0)

            s_af = _score_t2m(pred_af, truth_v)
            s_init = _score_t2m(pred_init, truth_v)

            row: dict[str, Any] = {
                "init_date": init_date,
                "season": r.get("season") or season_of(init_date),
                "lead_hours": lead,
                "valid_time": r["valid_time"],
                "ic_date": r["ic_date"],
                "ic_time": r["ic_time"],
                "student": {
                    "rmse": float(r["variables"]["t2m"]["student_rmse_vs_era5"]),
                    "acc": float(r["variables"]["t2m"]["student_acc"]),
                    "bias": float(r["variables"]["t2m"]["student_bias"]),
                },
                "persistence_af": s_af,
                "persistence_init": s_init,
                "k1_available": bool(r.get("k1_available")),
            }

            # teacher fields on student row when present
            t2 = r["variables"]["t2m"]
            if t2.get("teacher_rmse_vs_era5") is not None and r.get("k1_available"):
                row["k1"] = {
                    "rmse": float(t2["teacher_rmse_vs_era5"]),
                    "acc": float(t2["teacher_acc"]),
                    "bias": float(t2["teacher_bias"]),
                }

            if with_climatology:
                dt = valid.astype("datetime64[s]").astype(datetime)
                key = (dt.month, dt.day, dt.hour)
                if key not in clim_cache:
                    fields = []
                    for y in clim_years:
                        fields.append(truth.at("t2m", _shift_year(valid, y)))
                    # align to first field grid
                    stacked = np.stack([np.asarray(f.values, dtype=np.float64) for f in fields], axis=0)
                    clim_vals = np.nanmean(stacked, axis=0)
                    clim_da = fields[0].copy(data=clim_vals)
                    clim_cache[key] = clim_da
                row["climatology"] = _score_t2m(clim_cache[key], truth_v)

            baseline_rows.append(row)
            if (i + 1) % 20 == 0 or i == 0:
                print(
                    f"[{i+1}/{len(student_rows)}] {init_date} L+{lead}h "
                    f"pers_af_rmse={s_af['rmse']:.3f} student={row['student']['rmse']:.3f}",
                    flush=True,
                )
    finally:
        truth.close()

    # Fill K1 from dedicated k1_results if missing on student rows
    k1_lookup = {
        (r["init_date"], int(r["lead_hours"])): r["variables"]["t2m"] for r in k1_rows
    }
    for row in baseline_rows:
        if "k1" not in row:
            k = k1_lookup.get((row["init_date"], row["lead_hours"]))
            if k is not None:
                row["k1"] = {
                    "rmse": float(k["rmse"]),
                    "acc": float(k["acc"]),
                    "bias": float(k["bias"]),
                }

    def _agg(name: str) -> dict[str, Any]:
        by_lead: dict[int, list[dict[str, float]]] = defaultdict(list)
        for row in baseline_rows:
            if name == "k1" and "k1" not in row:
                continue
            by_lead[row["lead_hours"]].append(row[name] if name != "student" else row["student"])
        out: dict[str, Any] = {}
        for L, vals in sorted(by_lead.items()):
            rmses = [v["rmse"] for v in vals]
            accs = [v["acc"] for v in vals]
            biases = [v["bias"] for v in vals]
            out[str(L)] = {
                "n": len(vals),
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
                "acc_mean": float(np.mean(accs)),
                "bias_mean": float(np.mean(biases)),
            }
        return out

    models = ["persistence_af", "persistence_init", "student"]
    if with_climatology:
        models.append("climatology")
    models.append("k1")

    aggregate = {m: _agg(m) for m in models}

    # Skill score vs AF persistence: 1 - RMSE_model/RMSE_pers
    skill_vs_pers: dict[str, Any] = {}
    for L in sorted({r["lead_hours"] for r in baseline_rows}):
        pers = [r["persistence_af"]["rmse"] for r in baseline_rows if r["lead_hours"] == L]
        stu = [r["student"]["rmse"] for r in baseline_rows if r["lead_hours"] == L]
        pers_m, stu_m = float(np.mean(pers)), float(np.mean(stu))
        skill_vs_pers[str(L)] = {
            "persistence_af_rmse": pers_m,
            "student_rmse": stu_m,
            "student_skill_vs_persistence_af": float(1.0 - stu_m / pers_m) if pers_m > 0 else float("nan"),
        }

    report = {
        "protocol": "trackB_t2m_baselines_vs_expanded",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "expanded_json": str(expanded_json),
        "arco_path": arco_path,
        "definitions": {
            "persistence_af": "T_hat(valid)=T(IC); IC=init+(L-6)h — matched to AF student protocol",
            "persistence_init": "T_hat(valid)=T(init 00Z)",
            "climatology": (
                f"mean ERA5 same MM-DD HH over years {list(clim_years)}"
                if with_climatology
                else "not computed"
            ),
            "student": "from TRACKB_T2M_EXPANDED.json (v5 AF)",
            "k1": "from expanded JSON where available (n≈3 inits)",
        },
        "with_climatology": with_climatology,
        "clim_years": list(clim_years) if with_climatology else [],
        "n_rows": len(baseline_rows),
        "aggregate": aggregate,
        "skill_vs_persistence_af": skill_vs_pers,
        "rows": baseline_rows,
    }
    return report


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# African t2m baselines (persistence / student / K1)",
        "",
        f"**Protocol:** {report['protocol']}",
        f"**Generated:** {report['timestamp_utc']}",
        "",
        "## Definitions",
        "",
    ]
    for k, v in report["definitions"].items():
        lines.append(f"- **{k}:** {v}")
    lines += ["", "## Aggregate RMSE (K)", "", "| Lead | Pers(AF) | Pers(init) | Student | Clim | K1 | Student skill vs Pers(AF) |", "|-----:|---------:|-----------:|--------:|-----:|---:|--------------------------:|"]
    leads = sorted(int(L) for L in report["aggregate"]["student"])
    for L in leads:
        Ls = str(L)
        agg = report["aggregate"]
        pers = agg["persistence_af"][Ls]["rmse_mean"]
        pinit = agg["persistence_init"][Ls]["rmse_mean"]
        stu = agg["student"][Ls]["rmse_mean"]
        clim = agg.get("climatology", {}).get(Ls, {}).get("rmse_mean")
        k1 = agg.get("k1", {}).get(Ls, {}).get("rmse_mean")
        skill = report["skill_vs_persistence_af"][Ls]["student_skill_vs_persistence_af"]
        clim_s = f"{clim:.3f}" if clim is not None else "—"
        k1_s = f"{k1:.3f}" if k1 is not None else "—"
        lines.append(
            f"| +{L} h | {pers:.3f} | {pinit:.3f} | **{stu:.3f}** | {clim_s} | {k1_s} | {100*skill:.1f}% |"
        )
    lines += ["", f"Rows: {report['n_rows']}.", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--expanded_json",
        type=Path,
        default=_REPO_ROOT / "reports" / "TRACKB_T2M_EXPANDED.json",
    )
    p.add_argument("--arco_path", default=ARCO_DEFAULT)
    p.add_argument("--with_climatology", action="store_true")
    p.add_argument(
        "--clim_years",
        default="2018,2019,2020,2021,2022",
        help="Comma years for same-calendar climatology (exclude eval year 2023)",
    )
    p.add_argument(
        "--out_json",
        type=Path,
        default=_REPO_ROOT / "reports" / "TRACKB_T2M_BASELINES.json",
    )
    p.add_argument(
        "--out_md",
        type=Path,
        default=_REPO_ROOT / "reports" / "TRACKB_T2M_BASELINES.md",
    )
    args = p.parse_args()
    clim_years = tuple(int(x) for x in args.clim_years.split(",") if x.strip())

    report = score_baselines(
        expanded_json=args.expanded_json,
        arco_path=args.arco_path,
        with_climatology=args.with_climatology,
        clim_years=clim_years,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, args.out_md)
    print(f"wrote {args.out_json}", flush=True)
    print(f"wrote {args.out_md}", flush=True)
    print("skill_vs_persistence_af:", json.dumps(report["skill_vs_persistence_af"], indent=2), flush=True)


if __name__ == "__main__":
    main()

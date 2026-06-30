#!/usr/bin/env python3
"""Track A closure: forecast coarsened teacher + score (Lengau) / gate vs Phase 0."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.run_scorecard import _read_eval_config  # noqa: E402
from evaluation.trackA_gate import run_coarsen_gate, write_gate_report  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Track A forecast + score + optional gate")
    p.add_argument("--config", type=Path, default=_REPO_ROOT / "configs" / "trackA_coarsen.yaml")
    p.add_argument("--forecast-only", action="store_true")
    p.add_argument("--score-only", action="store_true")
    p.add_argument("--gate-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = _read_eval_config(args.config)
    fc = cfg.get("forecast") or {}
    inits = fc.get("inits") or []
    forecast_dir = _REPO_ROOT / fc.get("forecast_dir", "data/processed/trackA/forecasts")
    scorecard = _REPO_ROOT / fc.get("scorecard_out", "reports/TRACKA_COARSEN_SCORECARD.json")
    checkpoint = fc.get("checkpoint", "models/teacher_coarsened.ckpt")
    teacher = fc.get("teacher_config", "configs/teacher_n320_gt6.yaml")

    py = sys.executable
    if not args.gate_only and not args.score_only:
        for init in inits:
            cmd = [
                py,
                str(_REPO_ROOT / "scripts" / "run_phase0_forecast.py"),
                "--init",
                str(init),
                "--teacher-config",
                teacher,
                "--checkpoint",
                str(_REPO_ROOT / checkpoint),
                "--output",
                str(forecast_dir / f"{init}_00Z.nc"),
                "--cds-offline",
            ]
            if args.dry_run:
                cmd.append("--dry-run")
            print("exec:", " ".join(cmd))
            if not args.dry_run:
                subprocess.check_call(cmd)

    if not args.forecast_only and not args.gate_only:
        cmd = [
            py,
            str(_REPO_ROOT / "scripts" / "run_phase0_closure.py"),
            "--score-only",
            "--forecast-dir",
            str(forecast_dir),
            "--out",
            str(scorecard),
        ]
        if args.dry_run:
            print("exec:", " ".join(cmd), "(skipped in dry-run; needs GCS on laptop)")
        elif not args.forecast_only:
            subprocess.check_call(cmd)

    if not args.forecast_only and not args.score_only:
        if args.dry_run:
            print(f"# gate would compare {scorecard} vs Phase 0 baseline")
            return 0
        report = run_coarsen_gate(scorecard)
        gate_out = _REPO_ROOT / "reports" / "TRACKA_A1_GATE.json"
        write_gate_report(report, gate_out)
        print(f"Track A gate passed={report['passed']} -> {gate_out}")
        return 0 if report["passed"] else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

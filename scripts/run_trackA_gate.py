#!/usr/bin/env python3
"""Check Track A coarsen gate: candidate scorecard vs Phase 0 baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.trackA_gate import (  # noqa: E402
    load_trackA_coarsen_config,
    run_coarsen_gate,
    write_gate_report,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Track A acceptance gate (A1 coarsen / A2 prune)")
    p.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Scorecard JSON for coarsened/pruned model forecasts",
    )
    p.add_argument("--baseline", type=Path, default=None, help="Override baseline scorecard path")
    p.add_argument("--config", type=Path, default=_REPO_ROOT / "configs" / "trackA_coarsen.yaml")
    p.add_argument("--out", type=Path, default=_REPO_ROOT / "reports" / "TRACKA_A1_GATE.json")
    args = p.parse_args()

    cfg = load_trackA_coarsen_config(args.config)
    if args.out == _REPO_ROOT / "reports" / "TRACKA_A1_GATE.json" and "prune" in args.config.name:
        args.out = _REPO_ROOT / "reports" / "TRACKA_A2_GATE.json"

    report = run_coarsen_gate(
        args.candidate,
        baseline_scorecard=args.baseline,
        config=cfg,
    )
    # Tag step for A2 reports
    if "prune" in args.config.name:
        report["step"] = "prune"
    write_gate_report(report, args.out)

    print(f"# Track A coarsen gate: passed={report['passed']}")
    print(f"# variables={report['variables']} leads={report['leads_hours']} max_deg={report['max_rmse_degradation_pct']}%")
    print(f"# wrote {args.out}")
    print()
    print("| init | lead | var | base RMSE | cand RMSE | deg % | pass |")
    print("|---|---|---|---|---|---|---|")
    for row in report["checks"]:
        if "error" in row:
            print(f"| {row.get('init_date','?')} | +{row.get('lead_hours','?')}h | {row.get('variable','?')} | ERROR | ERROR | - | FAIL |")
            continue
        mark = "OK" if row["passed"] else "FAIL"
        print(
            f"| {row['init_date']} | +{row['lead_hours']}h | {row['variable']} | "
            f"{row['baseline_rmse']:.4g} | {row['candidate_rmse']:.4g} | "
            f"{row['degradation_pct']:.2f} | {mark} |"
        )

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

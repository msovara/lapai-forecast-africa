"""Track A driver — Anemoi coarsening + scorecard-gated acceptance."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from evaluation.run_scorecard import _read_eval_config
from evaluation.trackA_gate import load_trackA_coarsen_config, run_coarsen_gate, write_gate_report


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def suggested_invocation(step: str, config: Path) -> str:
    rr = repo_root()
    cfg = _read_eval_config(config) if config.exists() else {}
    gate = cfg.get("gate") or {}
    return (
        f"# Track A operator hints (step={step})\n"
        f"cd {rr}\n"
        f"export LAPAI_PYTHON=${{LAPAI_PYTHON:-/home/msovara/lustre/dev/lapai-anemoi/bin/python}}\n"
        f"bash scripts/lapai_inference_gate.sh   # smoke + verify --strict + inference --dry-run\n"
        f"# A1 coarsen fine-tune (when Hydra config ready):\n"
        f"# python training/train_trackA.py --step coarsen --anemoi-train --config-name=<hydra>\n"
        f"# After coarsened forecasts exist:\n"
        f"# python scripts/run_phase0_closure.py --score-only --forecast-dir data/processed/trackA/forecasts\n"
        f"# python scripts/run_trackA_gate.py --candidate reports/TRACKA_COARSEN_SCORECARD.json\n"
        f"# Gate contract: variables={gate.get('variables', ['t2m','u10','v10'])} "
        f"leads={gate.get('leads_hours', [24,48])} "
        f"max_deg={gate.get('max_rmse_degradation_pct', 5.0)}% vs "
        f"{gate.get('baseline_scorecard', 'reports/PHASE0_BASELINE_SCORECARD.json')}\n"
        f"# Config: {config}\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Track A: Anemoi coarsening + CRPS-gated pruning")
    p.add_argument("--step", choices=["coarsen", "prune"], required=True)
    p.add_argument("--config", type=Path, default=Path("configs/trackA_coarsen.yaml"))
    p.add_argument("--suggest-invocation", action="store_true", help="Print bash-style next steps and exit 0")
    p.add_argument(
        "--check-gate",
        type=Path,
        metavar="SCORECARD",
        help="Coarsen step: compare candidate scorecard JSON vs Phase 0 baseline",
    )
    p.add_argument(
        "--gate-out",
        type=Path,
        default=Path("reports/TRACKA_A1_GATE.json"),
        help="Write gate report JSON when using --check-gate",
    )
    p.add_argument(
        "--anemoi-train",
        nargs=argparse.REMAINDER,
        metavar="ARGS",
        help="Run anemoi-training train ARGS when tokens follow",
    )
    args = p.parse_args()

    if args.step == "coarsen" and not args.config.name.startswith("trackA_coarsen"):
        pass  # allow override
    if args.step == "prune" and str(args.config) == "configs/trackA_coarsen.yaml":
        args.config = Path("configs/trackA_prune.yaml")

    if args.suggest_invocation:
        print(suggested_invocation(args.step, args.config))
        return

    if args.check_gate is not None:
        if args.step != "coarsen":
            raise SystemExit("--check-gate only applies to --step coarsen")
        report = run_coarsen_gate(args.check_gate, config=load_trackA_coarsen_config(args.config))
        write_gate_report(report, repo_root() / args.gate_out)
        print(f"Track A coarsen gate passed={report['passed']} -> {args.gate_out}")
        raise SystemExit(0 if report["passed"] else 1)

    if args.anemoi_train is not None and len(args.anemoi_train) > 0:
        exe = shutil.which("anemoi-training")
        if exe:
            cmd = [exe, "train", *args.anemoi_train]
        else:
            cmd = [sys.executable, "-m", "anemoi.training", "train", *args.anemoi_train]
        print("exec:", " ".join(cmd))
        raise SystemExit(subprocess.call(cmd))

    cfg = _read_eval_config(args.config) if args.config.exists() else {}
    gate = cfg.get("gate") or {}
    print(
        "Track A stub: integrate anemoi-training here; PBS uses lustre LAPAI_PYTHON.\n"
        f"  step={args.step} config={args.config}\n"
        f"  coarsen gate: {gate.get('variables')} @ leads {gate.get('leads_hours')} "
        f"(max {gate.get('max_rmse_degradation_pct', 5.0)}% RMSE vs Phase 0 baseline)\n"
        "Run: python training/train_trackA.py --step coarsen --suggest-invocation\n"
        "Gate:  python training/train_trackA.py --step coarsen --check-gate reports/TRACKA_COARSEN_SCORECARD.json\n"
        "Or:    python scripts/run_trackA_gate.py --candidate reports/TRACKA_COARSEN_SCORECARD.json\n"
        "Grid coarsening helpers: utils/grid_coarsen.py"
    )


if __name__ == "__main__":
    main()

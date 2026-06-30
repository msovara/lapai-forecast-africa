"""Track A driver — Anemoi coarsening + scorecard-gated acceptance."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.run_scorecard import _read_eval_config
from evaluation.trackA_gate import load_trackA_coarsen_config, run_coarsen_gate, write_gate_report
from scripts.convert_inference_to_warmstart_ckpt import convert_inference_to_warmstart, default_warmstart_path


def repo_root() -> Path:
    return _REPO_ROOT


def suggested_invocation(step: str, config: Path) -> str:
    rr = repo_root()
    cfg = _read_eval_config(config) if config.exists() else {}
    gate = cfg.get("gate") or {}
    return (
        f"# Track A operator hints (step={step})\n"
        f"cd {rr}\n"
        f"export LAPAI_PYTHON=${{LAPAI_PYTHON:-/home/msovara/lustre/dev/lapai-anemoi/bin/python}}\n"
        f"bash scripts/lapai_inference_gate.sh   # smoke + verify --strict + inference --dry-run\n"
        f"# A1 coarsen fine-tune (after anemoi.dataset is set in {config}):\n"
        f"# python training/train_trackA.py --step coarsen --config {config}\n"
        f"# After coarsened forecasts exist:\n"
        f"# python scripts/run_phase0_closure.py --score-only --forecast-dir data/processed/trackA/forecasts\n"
        f"# python scripts/run_trackA_gate.py --candidate reports/TRACKA_COARSEN_SCORECARD.json\n"
        f"# Gate contract: variables={gate.get('variables', ['t2m','u10','v10'])} "
        f"leads={gate.get('leads_hours', [24,48])} "
        f"max_deg={gate.get('max_rmse_degradation_pct', 5.0)}% vs "
        f"{gate.get('baseline_scorecard', 'reports/PHASE0_BASELINE_SCORECARD.json')}\n"
        f"# Config: {config}\n"
    )


def _anemoi_train_cmd(extra: list[str]) -> list[str]:
    exe = shutil.which("anemoi-training")
    if exe:
        return [exe, "train", *extra]
    return [sys.executable, "-m", "anemoi.training", "train", *extra]


def _run_coarsen_anemoi(cfg: dict[str, Any]) -> int:
    """Launch anemoi-training when configs/trackA_coarsen.yaml anemoi block is populated."""
    anemoi = cfg.get("anemoi") or {}
    dataset = anemoi.get("dataset")
    if not dataset:
        print(
            "Track A coarsen: set anemoi.dataset in configs/trackA_coarsen.yaml "
            "(ERA5 N96 Zarr path on Lengau) before real training."
        )
        return 2

    warm_start = anemoi.get(
        "warm_start",
        str(repo_root() / "models" / "teacher_n320_gt6" / "inference.ckpt"),
    )
    output_root = anemoi.get("output_root", str(repo_root() / "models" / "trackA_coarsen_runs"))
    processor_res = int(anemoi.get("processor_resolution", 6))  # TriNodes: 6 ≈ O48
    max_steps = int(anemoi.get("max_steps", 500))
    max_epochs = anemoi.get("max_epochs")
    load_weights_only = anemoi.get("load_weights_only", True)

    rr = repo_root()
    dataset_path = Path(dataset)
    if not dataset_path.is_absolute():
        dataset_path = rr / dataset_path
    warm_start_path = Path(warm_start)
    if not warm_start_path.is_absolute():
        warm_start_path = rr / warm_start_path
    warm_start_training = anemoi.get("warm_start_training")
    if warm_start_training:
        warm_start_path = Path(warm_start_training)
        if not warm_start_path.is_absolute():
            warm_start_path = rr / warm_start_path
    else:
        warm_start_path = convert_inference_to_warmstart(
            warm_start_path,
            default_warmstart_path(warm_start_path),
        )
    output_root_path = Path(output_root)
    if not output_root_path.is_absolute():
        output_root_path = rr / output_root_path
    output_root_path.mkdir(parents=True, exist_ok=True)
    mlflow_uri = f"file://{(output_root_path / 'mlruns').resolve()}"

    overrides = [
        "data=zarr",
        "graph=multi_scale",
        "model=gnn",
        f"system.input.dataset={dataset_path.as_posix()}",
        "system.input.graph=null",
        f"system.input.warm_start={warm_start_path.as_posix()}",
        f"system.output.root={output_root_path.as_posix()}",
        f"graph.nodes.hidden.node_builder.resolution={processor_res}",
        "training.transfer_learning=True",
        f"training.load_weights_only={'True' if load_weights_only else 'False'}",
        f"training.max_steps={max_steps}",
        "dataloader.batch_size.training=1",
        "dataloader.batch_size.validation=1",
        "dataloader.limit_batches.validation=1",
        "training.num_sanity_val_steps=0",
        "+diagnostics.plot.callbacks=[]",
        f"diagnostics.log.mlflow.tracking_uri={mlflow_uri}",
    ]
    if max_epochs is not None:
        overrides.append(f"training.max_epochs={int(max_epochs)}")
    overrides.extend(str(x) for x in (anemoi.get("hydra_overrides") or []))

    cmd = _anemoi_train_cmd(overrides)
    print("exec:", " ".join(cmd))
    return subprocess.call(cmd)


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
        cmd = _anemoi_train_cmd(list(args.anemoi_train))
        print("exec:", " ".join(cmd))
        raise SystemExit(subprocess.call(cmd))

    cfg = _read_eval_config(args.config) if args.config.exists() else {}

    if args.step == "coarsen":
        raise SystemExit(_run_coarsen_anemoi(cfg))

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

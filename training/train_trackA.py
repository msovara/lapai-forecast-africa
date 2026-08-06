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
    if step == "prune":
        return (
            f"# Track A A2 prune operator hints\n"
            f"cd {rr}\n"
            f"export LAPAI_PYTHON=${{LAPAI_PYTHON:-/home/msovara/lustre/dev/lapai-anemoi/bin/python}}\n"
            f"# PREREQ — GraphTransformer O96 (A1b), not GNN A1:\n"
            f"#   LAPAI_TRACKA_CONFIG=configs/trackA_gt_coarsen.yaml qsub -v LAPAI_TRACKA_CONFIG,LAPAI_SKIP_GATE=1 pbs/trackA_full.pbs\n"
            f"#   ln -sfn .../inference-last.ckpt models/teacher_gt_coarsened.ckpt\n"
            f"# Round-1 soft-mask + recovery fine-tune:\n"
            f"#   $LAPAI_PYTHON training/train_trackA.py --step prune --config {config}\n"
            f"#   qsub pbs/trackA_prune.pbs\n"
            f"# Gate vs A1 scorecard (<={gate.get('max_rmse_degradation_pct', 3.0)}% RMSE proxy, vars={gate.get('variables')}):\n"
            f"#   $LAPAI_PYTHON scripts/run_trackA_gate.py --candidate reports/TRACKA_PRUNE_SCORECARD.json \\\n"
            f"#     --config {config} --out reports/TRACKA_A2_GATE.json\n"
            f"# See reports/TRACKA_A2_START.md\n"
        )
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
    model_name = str(anemoi.get("model", "gnn"))

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
        f"model={model_name}",
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


def _run_prune_anemoi(cfg: dict[str, Any]) -> int:
    """Soft-mask attention heads on a GraphTransformer ckpt, then recovery fine-tune."""
    from utils.head_prune import prune_checkpoint, write_prune_report

    anemoi = cfg.get("anemoi") or {}
    prune_cfg = cfg.get("prune") or {}
    rr = repo_root()

    warm = Path(anemoi.get("warm_start", "models/teacher_gt_coarsened.ckpt"))
    if not warm.is_absolute():
        warm = rr / warm
    if not warm.exists():
        print(
            "ERROR: A2 prune warm_start missing:",
            warm,
            "\nFine-tune GraphTransformer on O96 first:",
            "configs/trackA_gt_coarsen.yaml → models/teacher_gt_coarsened.ckpt",
            "\n(A1 GNN teacher_coarsened.ckpt has no attention heads.)",
        )
        return 2

    masked = Path(anemoi.get("masked_ckpt", "models/teacher_pruned_masked.ckpt"))
    if not masked.is_absolute():
        masked = rr / masked
    fraction = float(prune_cfg.get("fraction_per_round", cfg.get("prune_fraction_per_round", 0.10)))
    scope = str(prune_cfg.get("scope", "processor"))
    if isinstance(scope, list):
        scope = scope[0] if len(scope) == 1 else "processor"
    importance = str(prune_cfg.get("importance", "weight_l1"))

    print(f"[A2 prune] masking heads on {warm} → {masked} (fraction={fraction}, scope={scope})")
    report = prune_checkpoint(
        warm,
        masked,
        fraction=fraction,
        scope=scope if scope != "all" else "processor",
        importance=importance,
    )
    report_path = rr / "reports" / f"TRACKA_A2_PRUNE_ROUND{int(prune_cfg.get('round', 1))}.json"
    write_prune_report(report, report_path)
    print(f"[A2 prune] wrote {report_path}")
    print(f"[A2 prune] dropped_heads sample:", list(report["dropped_heads"].items())[:3])

    # Recovery fine-tune from masked weights.
    ft_cfg = dict(cfg)
    ft_anemoi = dict(anemoi)
    ft_anemoi["warm_start"] = str(masked)
    ft_anemoi["model"] = anemoi.get("model", "graphtransformer")
    ft_cfg["anemoi"] = ft_anemoi
    return _run_coarsen_anemoi(ft_cfg)


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
        report = run_coarsen_gate(args.check_gate, config=load_trackA_coarsen_config(args.config))
        out = repo_root() / args.gate_out
        if args.step == "prune" and args.gate_out == Path("reports/TRACKA_A1_GATE.json"):
            out = repo_root() / "reports" / "TRACKA_A2_GATE.json"
        write_gate_report(report, out)
        print(f"Track A gate passed={report['passed']} -> {out}")
        raise SystemExit(0 if report["passed"] else 1)

    if args.anemoi_train is not None and len(args.anemoi_train) > 0:
        cmd = _anemoi_train_cmd(list(args.anemoi_train))
        print("exec:", " ".join(cmd))
        raise SystemExit(subprocess.call(cmd))

    cfg = _read_eval_config(args.config) if args.config.exists() else {}

    if args.step == "coarsen":
        raise SystemExit(_run_coarsen_anemoi(cfg))

    if args.step == "prune":
        raise SystemExit(_run_prune_anemoi(cfg))

    raise SystemExit(f"unknown step {args.step}")


if __name__ == "__main__":
    main()

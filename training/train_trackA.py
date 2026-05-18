"""Track A driver — Anemoi wiring on HPC; prints contract until Hydra configs are fused."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def suggested_invocation(step: str, config: str) -> str:
    rr = repo_root()
    return (
        f"# Track A operator hints (step={step})\n"
        f"cd {rr}\n"
        f"conda activate lapai-anemoi   # or LAPAI_CONDA_TRACK_A override — see docs/LENGAU.md\n"
        f"python scripts/smoke_teacher_ckpt.py     # after downloading teacher weights\n"
        f"python scripts/run_aifs_inference.py --dry-run   # then run without --dry-run for anemoi-inference\n"
        f"bash scripts/invoke_anemoi_training_example.sh   # read HF / ECMWF anemoi-training examples\n"
        f"# This stub reads: {config}\n"
        f"# Full integration: anemoi-training train ... with Hydra trees that honour trackA_*.yaml\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Track A: Anemoi coarsening + CRPS-gated pruning")
    p.add_argument("--step", choices=["coarsen", "prune"], required=True)
    p.add_argument("--config", type=str, default="configs/trackA_coarsen.yaml")
    p.add_argument(
        "--suggest-invocation",
        action="store_true",
        help="Print bash-style next steps and exit 0",
    )
    p.add_argument(
        "--anemoi-train",
        nargs=argparse.REMAINDER,
        metavar="ARGS",
        help="If any tokens follow, run: python -m anemoi.training train ARGS (pass-through)",
    )
    args = p.parse_args()

    if args.suggest_invocation:
        print(suggested_invocation(args.step, args.config))
        return

    if args.anemoi_train is not None and len(args.anemoi_train) > 0:
        cmd = [sys.executable, "-m", "anemoi.training", "train", *args.anemoi_train]
        print("exec:", " ".join(cmd))
        raise SystemExit(subprocess.call(cmd))

    print(
        "Track A stub: integrate anemoi-training here; PBS defaults use LAPAI_CONDA_TRACK_A "
        "(isolated lapai-anemoi from environment-anemoi.yml).\n"
        f"  step={args.step} config={args.config}\n"
        "Teacher HF pin: configs/teacher_aifs.yaml — download ckpt via scripts/download_teacher_ckpt.py.\n"
        "Smoke inference: scripts/run_aifs_inference.py (--dry-run first).\n"
        "Run: python training/train_trackA.py --step coarsen --suggest-invocation\n"
        "Or delegate: python training/train_trackA.py --step coarsen --anemoi-train --config-name=...\n"
        "Grid coarsening helpers live in utils/grid_coarsen.py for offline ERA5 stacks."
    )


if __name__ == "__main__":
    main()

"""Track A driver - integrate with ECMWF Anemoi on HPC; local stub prints contract."""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description="Track A: Anemoi coarsening + CRPS-gated pruning")
    p.add_argument("--step", choices=["coarsen", "prune"], required=True)
    p.add_argument("--config", type=str, default="configs/trackA_coarsen.yaml")
    args = p.parse_args()

    print(
        "Track A stub: integrate anemoi-training here; PBS defaults use LAPAI_CONDA_TRACK_A "
        "(isolated lapai-anemoi from environment-anemoi.yml).\n"
        f"  step={args.step} config={args.config}\n"
        "Teacher HF pin: configs/teacher_aifs.yaml — download ckpt via scripts/download_teacher_ckpt.py.\n"
        "Replace this entrypoint with anemoi-training when challenge recipe / configs are wired.\n"
        "Grid coarsening helpers live in utils/grid_coarsen.py for offline ERA5 stacks."
    )


if __name__ == "__main__":
    main()

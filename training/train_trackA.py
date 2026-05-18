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
        "(CHPC anemoi-training).\n"
        f"  step={args.step} config={args.config}\n"
        "Replace this entrypoint with anemoi-training after ECMWF challenge recipe access.\n"
        "Grid coarsening helpers live in utils/grid_coarsen.py for offline ERA5 stacks."
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# EXAMPLES ONLY — uncomment and customise once you deploy ECMWF Anemoi
# Hydra / training YAML packs on Lengau. See:
#   https://huggingface.co/ecmwf/aifs-single-1.0 → "How to train AIFS Single v1.0"
# -----------------------------------------------------------------------------
#
# Typical HF recipe (adapt DATASETS_PATH / OUTPUT_PATH / PRETRAINING_RUN_ID):
#
# export DATASETS_PATH=/mnt/lustre/users/$USER/lapai/datasets-zarr
# export OUTPUT_PATH=/mnt/lustre/users/$USER/lapai/anemoi-runs
#
# conda activate lapai-anemoi
# anemoi-training train --config-name=config_pretraining.yaml
#
# export PRETRAINING_RUN_ID=<run_id_from_checkpoint_folder>
# anemoi-training train --config-name=config_finetuning.yaml
#
# LapAI Track A (this repo) should eventually call the same driver with a
# config that applies coarsen + prune from configs/trackA_*.yaml — see PLAN.md.
# -----------------------------------------------------------------------------

echo "This file is documentation only. Read the comments or HF model card for real commands."
exit 0

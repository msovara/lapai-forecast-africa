#!/bin/bash
# Medium K1 teacher cache fill from 2020-2021 ERA5 (background-friendly).
set -euo pipefail
cd /local/Mthetho/lapai-forecast
export MKL_INTERFACE_LAYER=GNU
export CUDA_VISIBLE_DEVICES=1
source /local/Mthetho/miniforge3/etc/profile.d/conda.sh
conda activate /local/Mthetho/envs/lapai-anemoi
pip install -e . --no-deps -q

OUT=data/processed/lapai/teacher_k1_cache.zarr
mkdir -p logs
python -u training/build_teacher_feature_cache.py \
  --era5 data/processed/lapai/era5_n96_2020_2021.zarr \
  --teacher models/teacher_pruned.ckpt \
  --out "$OUT" \
  --samples 32 \
  --stride 8 \
  --start 0 \
  --lead_steps 1 \
  --overwrite \
  2>&1 | tee logs/trackB_teacher_cache_medium.log

echo "EXIT:$?"
du -sh "$OUT"

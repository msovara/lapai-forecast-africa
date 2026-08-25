#!/bin/bash
# Build smoke K1 teacher feature cache on Cassava GPU1 (anemoi env).
set -euo pipefail
cd /local/Mthetho/lapai-forecast
export MKL_INTERFACE_LAYER=GNU
export CUDA_VISIBLE_DEVICES=1
source /local/Mthetho/miniforge3/etc/profile.d/conda.sh
conda activate /local/Mthetho/envs/lapai-anemoi

# Ensure editable package sees fixed cache_schema
pip install -e . --no-deps -q

OUT=data/processed/lapai/teacher_k1_cache_smoke.zarr
mkdir -p logs data/processed/lapai
python -u training/build_teacher_feature_cache.py \
  --era5 data/processed/lapai/era5_n96_smoke.zarr \
  --teacher models/teacher_pruned.ckpt \
  --out "$OUT" \
  --samples 8 \
  --stride 12 \
  --start 0 \
  --lead_steps 1 \
  --overwrite \
  2>&1 | tee logs/trackB_teacher_cache_smoke.log

echo "EXIT:$?"
du -sh "$OUT"

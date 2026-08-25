#!/bin/bash
# Expand K1 teacher feature cache to T=256 (GPU time cheap).
set -euo pipefail
cd /local/Mthetho/lapai-forecast
export MKL_INTERFACE_LAYER=GNU
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
source /local/Mthetho/miniforge3/etc/profile.d/conda.sh
conda activate /local/Mthetho/envs/lapai-anemoi

LOG=/local/Mthetho/logs/trackB_teacher_cache_t256.log
OUT=data/processed/lapai/teacher_k1_cache_t256.zarr
ERA5=data/processed/lapai/era5_n96_2020_2021.zarr

echo "[cache-t256] start $(date -Iseconds)" | tee "$LOG"
if [[ ! -d "$ERA5" ]]; then
  echo "[cache-t256] missing $ERA5" | tee -a "$LOG"
  exit 1
fi

python -u training/build_teacher_feature_cache.py \
  --era5 "$ERA5" \
  --teacher models/teacher_pruned.ckpt \
  --out "$OUT" \
  --samples 256 \
  --stride 4 \
  --overwrite \
  --device cuda \
  2>&1 | tee -a "$LOG"

echo "[cache-t256] done $(date -Iseconds)" | tee -a "$LOG"

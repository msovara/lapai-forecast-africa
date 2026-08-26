#!/bin/bash
# Held-out Jan-2023 student 6h for v4 (Cassava GPU1). Teacher multi-lead can be skipped.
set -euo pipefail
cd /local/Mthetho/lapai-forecast
export MKL_INTERFACE_LAYER=GNU
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
source /local/Mthetho/miniforge3/etc/profile.d/conda.sh
# anemoi has earthkit/CDS + often netCDF; credit has torch. Prefer anemoi if both work.
conda activate /local/Mthetho/envs/lapai-anemoi

LOG=/local/Mthetho/logs/trackB_held_out_v4.log
CKPT="${1:-models/student_global_stable_v4.ckpt}"
OUT=reports/TRACKB_HELD_OUT_JAN2023_V4_STUDENT.json

echo "[heldout-v4] start $(date -Iseconds) ckpt=$CKPT" | tee "$LOG"
pip install -e . --no-deps -q

python -u evaluation/trackB_held_out_jan2023.py \
  --student_ckpt "$CKPT" \
  --device cuda \
  --skip_teacher \
  --no_era5 \
  --student_forecast_dir data/processed/trackB_student_heldout_v4/forecasts \
  --out "$OUT" \
  2>&1 | tee -a "$LOG"

echo "[heldout-v4] done $(date -Iseconds)" | tee -a "$LOG"

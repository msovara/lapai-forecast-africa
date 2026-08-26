#!/bin/bash
# Retry analysis-forced +24h only (needs cdsapi for 18Z IC download).
set -euo pipefail
cd /local/Mthetho/lapai-forecast
export MKL_INTERFACE_LAYER=GNU
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
source /local/Mthetho/miniforge3/etc/profile.d/conda.sh
conda activate /local/Mthetho/envs/lapai-anemoi
pip install -e . --no-deps -q
pip install -q cdsapi || true
python -c 'import cdsapi; print("cdsapi ok")'

LOG=/local/Mthetho/logs/trackB_held_out_v5_L24.log
CKPT=models/student_global_stable_v5.ckpt
echo "[heldout-L24] start $(date -Iseconds)" | tee "$LOG"
python -u evaluation/trackB_held_out_jan2023.py \
  --student_ckpt "$CKPT" \
  --device cuda \
  --skip_teacher \
  --no_era5 \
  --student_leads 24 \
  --allow_download \
  --student_forecast_dir data/processed/trackB_student_heldout_v5/forecasts \
  --out reports/TRACKB_HELD_OUT_JAN2023_V5_L24.json \
  2>&1 | tee -a "$LOG"
echo "[heldout-L24] done $(date -Iseconds)" | tee -a "$LOG"

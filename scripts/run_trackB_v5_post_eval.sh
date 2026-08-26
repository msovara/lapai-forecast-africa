#!/bin/bash
# After v5 tp-recovery train: held-out Jan-2023 student (+ optional multi-lead).
set -euo pipefail
cd /local/Mthetho/lapai-forecast
export MKL_INTERFACE_LAYER=GNU
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
source /local/Mthetho/miniforge3/etc/profile.d/conda.sh

CKPT="${1:-models/student_global_stable_v5.ckpt}"
HELD_LOG=/local/Mthetho/logs/trackB_held_out_v5.log
# Analysis-forced multi-lead: 6h from 00Z IC; 24h from 18Z IC (CDS may download).
STUDENT_LEADS="${STUDENT_LEADS:-6,24}"

echo "[post-v5] start $(date -Iseconds) ckpt=$CKPT leads=$STUDENT_LEADS" | tee /local/Mthetho/logs/trackB_v5_post.log

conda activate /local/Mthetho/envs/lapai-anemoi
pip install -e . --no-deps -q
echo "[heldout] start $(date -Iseconds)" | tee "$HELD_LOG"

set +e
python -u evaluation/trackB_held_out_jan2023.py \
  --student_ckpt "$CKPT" \
  --device cuda \
  --skip_teacher \
  --student_leads "$STUDENT_LEADS" \
  --allow_download \
  --student_forecast_dir data/processed/trackB_student_heldout_v5/forecasts \
  --out reports/TRACKB_HELD_OUT_JAN2023_V5.json \
  2>&1 | tee -a "$HELD_LOG"
rc=${PIPESTATUS[0]}
set -e
if [[ $rc -ne 0 ]]; then
  echo "[heldout] ERA5 path failed (rc=$rc); retry --no_era5 (6h only if 18Z missing)" | tee -a "$HELD_LOG"
  python -u evaluation/trackB_held_out_jan2023.py \
    --student_ckpt "$CKPT" \
    --device cuda \
    --skip_teacher \
    --no_era5 \
    --student_leads "$STUDENT_LEADS" \
    --allow_download \
    --student_forecast_dir data/processed/trackB_student_heldout_v5/forecasts \
    --out reports/TRACKB_HELD_OUT_JAN2023_V5_STUDENT.json \
    2>&1 | tee -a "$HELD_LOG"
fi

echo "[post-v5] done $(date -Iseconds)" | tee -a /local/Mthetho/logs/trackB_v5_post.log

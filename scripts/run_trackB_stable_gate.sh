#!/bin/bash
set -eo pipefail
cd /local/Mthetho/lapai-forecast
export MKL_INTERFACE_LAYER=GNU
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
source /local/Mthetho/miniforge3/etc/profile.d/conda.sh
conda activate /local/Mthetho/envs/lapai-credit
LOG=/local/Mthetho/logs/trackB_stable_gate.log
CKPT=models/student_global_stable.ckpt
echo "[trackB-gate] start $(date -Iseconds) ckpt=$CKPT" | tee "$LOG"
python -u evaluation/trackB_gate.py \
  --ckpt "$CKPT" \
  --cache data/processed/lapai/teacher_k1_cache.zarr \
  --device cuda \
  --scorecard_out reports/TRACKB_STUDENT_SCORECARD.json \
  --gate_out reports/TRACKB_GATE.json \
  2>&1 | tee -a "$LOG"
echo "[trackB-gate] done $(date -Iseconds)" | tee -a "$LOG"

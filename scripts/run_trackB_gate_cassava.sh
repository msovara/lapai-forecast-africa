#!/bin/bash
# Track B MVP gate on Cassava GPU1 (student vs K1 teacher on cache).
set -euo pipefail
cd /local/Mthetho/lapai-forecast
export MKL_INTERFACE_LAYER=GNU
export CUDA_VISIBLE_DEVICES=1
export PATH="/local/Mthetho/envs/lapai-credit/bin:$PATH"
PY=/local/Mthetho/envs/lapai-credit/bin/python

mkdir -p logs reports
$PY -m pip install -e . --no-deps -q

echo "[trackB_gate] start $(date -Is)" | tee logs/trackB_gate.log
$PY -u evaluation/trackB_gate.py \
  --ckpt models/student_global.ckpt \
  --cache data/processed/lapai/teacher_k1_cache.zarr \
  --extra_cache data/processed/lapai/teacher_k1_cache_smoke.zarr \
  --device cuda \
  --domain global \
  --max_degradation_pct 15 \
  --scorecard_out reports/TRACKB_STUDENT_SCORECARD.json \
  --gate_out reports/TRACKB_GATE.json \
  2>&1 | tee -a logs/trackB_gate.log

echo "[trackB_gate] exit=$? finished $(date -Is)" | tee -a logs/trackB_gate.log

#!/bin/bash
# Smoke-train Track B student against K1 teacher feature cache (lapai-credit, GPU1).
set -euo pipefail
cd /local/Mthetho/lapai-forecast
export MKL_INTERFACE_LAYER=GNU
export CUDA_VISIBLE_DEVICES=1
source /local/Mthetho/miniforge3/etc/profile.d/conda.sh
conda activate /local/Mthetho/envs/lapai-credit
pip install -e . --no-deps -q

CACHE=data/processed/lapai/teacher_k1_cache_smoke.zarr
python -u training/train_student.py \
  --config configs/student_global.yaml \
  --cache "$CACHE" \
  --epochs 1 \
  --steps_per_epoch 2 \
  --batch_size 1 \
  --device cuda \
  --out models/student_cache_smoke.pt \
  2>&1 | tee logs/trackB_train_cache_smoke.log

echo "EXIT:$?"

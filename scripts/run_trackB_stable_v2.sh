#!/bin/bash
# Track B student stable v2: L_A-only, more epochs, prefer T=128 cache.
set -euo pipefail
cd /local/Mthetho/lapai-forecast
export MKL_INTERFACE_LAYER=GNU
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
source /local/Mthetho/miniforge3/etc/profile.d/conda.sh
conda activate /local/Mthetho/envs/lapai-credit

LOG=/local/Mthetho/logs/trackB_student_stable_v2_train.log
GATE_LOG=/local/Mthetho/logs/trackB_stable_v2_gate.log
CKPT=models/student_global_stable_v2.ckpt

CACHE=data/processed/lapai/teacher_k1_cache.zarr
if [[ -d data/processed/lapai/teacher_k1_cache_t128.zarr ]]; then
  CACHE=data/processed/lapai/teacher_k1_cache_t128.zarr
elif [[ -d data/processed/lapai/teacher_k1_cache_t256.zarr ]]; then
  CACHE=data/processed/lapai/teacher_k1_cache_t256.zarr
fi

echo "[trackB-stable-v2] start $(date -Iseconds)" | tee "$LOG"
echo "[trackB-stable-v2] cache=$CACHE ckpt=$CKPT" | tee -a "$LOG"
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)' | tee -a "$LOG"

python -u training/train_student.py \
  --config configs/student_global.yaml \
  --cache "$CACHE" \
  --epochs 80 \
  --device cuda \
  --batch_size 4 \
  --steps_per_epoch 64 \
  --out "$CKPT" \
  2>&1 | tee -a "$LOG"

echo "[trackB-stable-v2] train done $(date -Iseconds)" | tee -a "$LOG"

echo "[trackB-gate-v2] start $(date -Iseconds)" | tee "$GATE_LOG"
python -u evaluation/trackB_gate.py \
  --ckpt "$CKPT" \
  --cache "$CACHE" \
  --device cuda \
  --domain global \
  2>&1 | tee -a "$GATE_LOG"
echo "[trackB-gate-v2] done $(date -Iseconds)" | tee -a "$GATE_LOG"

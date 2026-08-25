#!/bin/bash
# Track B student stable v3: resume v2, channel-weighted L_A, prefer T=256 cache.
set -euo pipefail
cd /local/Mthetho/lapai-forecast
export MKL_INTERFACE_LAYER=GNU
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
source /local/Mthetho/miniforge3/etc/profile.d/conda.sh
conda activate /local/Mthetho/envs/lapai-credit

LOG=/local/Mthetho/logs/trackB_student_stable_v3_train.log
GATE_LOG=/local/Mthetho/logs/trackB_stable_v3_gate.log
CKPT=models/student_global_stable_v3.ckpt
RESUME=models/student_global_stable_v2.ckpt

CACHE=data/processed/lapai/teacher_k1_cache_t128.zarr
if [[ -d data/processed/lapai/teacher_k1_cache_t256.zarr ]]; then
  CACHE=data/processed/lapai/teacher_k1_cache_t256.zarr
elif [[ -d data/processed/lapai/teacher_k1_cache.zarr ]]; then
  CACHE=data/processed/lapai/teacher_k1_cache.zarr
fi

echo "[trackB-stable-v3] start $(date -Iseconds)" | tee "$LOG"
echo "[trackB-stable-v3] cache=$CACHE resume=$RESUME -> $CKPT" | tee -a "$LOG"
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)' | tee -a "$LOG"

if [[ ! -f "$RESUME" ]]; then
  echo "[trackB-stable-v3] missing resume ckpt $RESUME" | tee -a "$LOG"
  exit 1
fi

python -u training/train_student.py \
  --config configs/student_global.yaml \
  --cache "$CACHE" \
  --resume "$RESUME" \
  --epochs 80 \
  --device cuda \
  --batch_size 4 \
  --steps_per_epoch 64 \
  --out "$CKPT" \
  2>&1 | tee -a "$LOG"

echo "[trackB-stable-v3] train done $(date -Iseconds)" | tee -a "$LOG"

echo "[trackB-gate-v3] start $(date -Iseconds)" | tee "$GATE_LOG"
python -u evaluation/trackB_gate.py \
  --ckpt "$CKPT" \
  --cache "$CACHE" \
  --device cuda \
  --domain global \
  2>&1 | tee -a "$GATE_LOG"
echo "[trackB-gate-v3] done $(date -Iseconds)" | tee -a "$GATE_LOG"

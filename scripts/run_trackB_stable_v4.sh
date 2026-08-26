#!/bin/bash
# Track B student stable v4: generalization vs held-out Jan-2023.
# Resume v3; full-year cache; Africa mix; precip log1p; input norm; softplus tp.
set -euo pipefail
cd /local/Mthetho/lapai-forecast
export MKL_INTERFACE_LAYER=GNU
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
source /local/Mthetho/miniforge3/etc/profile.d/conda.sh
conda activate /local/Mthetho/envs/lapai-credit

LOG=/local/Mthetho/logs/trackB_student_stable_v4_train.log
GATE_LOG=/local/Mthetho/logs/trackB_stable_v4_gate.log
CKPT=models/student_global_stable_v4.ckpt
RESUME=models/student_global_stable_v3.ckpt
CFG=configs/student_global_v4.yaml

CACHE=data/processed/lapai/teacher_k1_cache_full2020_2021.zarr
if [[ ! -d "$CACHE" ]]; then
  if [[ -d data/processed/lapai/teacher_k1_cache_t256.zarr ]]; then
    echo "[trackB-stable-v4] WARN: full cache missing; falling back to t256" | tee "$LOG"
    CACHE=data/processed/lapai/teacher_k1_cache_t256.zarr
  else
    echo "[trackB-stable-v4] missing cache" | tee "$LOG"
    exit 1
  fi
fi

echo "[trackB-stable-v4] start $(date -Iseconds)" | tee "$LOG"
echo "[trackB-stable-v4] cache=$CACHE resume=$RESUME cfg=$CFG -> $CKPT" | tee -a "$LOG"
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)' | tee -a "$LOG"

if [[ ! -f "$RESUME" ]]; then
  echo "[trackB-stable-v4] missing resume ckpt $RESUME" | tee -a "$LOG"
  exit 1
fi

pip install -e . --no-deps -q

python -u training/train_student.py \
  --config "$CFG" \
  --cache "$CACHE" \
  --resume "$RESUME" \
  --epochs 100 \
  --device cuda \
  --batch_size 4 \
  --steps_per_epoch 64 \
  --out "$CKPT" \
  2>&1 | tee -a "$LOG"

echo "[trackB-stable-v4] train done $(date -Iseconds)" | tee -a "$LOG"

echo "[trackB-gate-v4] start $(date -Iseconds)" | tee "$GATE_LOG"
python -u evaluation/trackB_gate.py \
  --ckpt "$CKPT" \
  --cache "$CACHE" \
  --device cuda \
  --domain global \
  --scorecard_out reports/TRACKB_STUDENT_SCORECARD_V4.json \
  --gate_out reports/TRACKB_GATE_V4.json \
  2>&1 | tee -a "$GATE_LOG"
# Also score Africa box on cache (regression check aligned with held-out domain).
python -u evaluation/trackB_gate.py \
  --ckpt "$CKPT" \
  --cache "$CACHE" \
  --device cuda \
  --domain africa \
  --scorecard_out reports/TRACKB_STUDENT_SCORECARD_V4_AFRICA.json \
  --gate_out reports/TRACKB_GATE_V4_AFRICA.json \
  2>&1 | tee -a "$GATE_LOG"
echo "[trackB-gate-v4] done $(date -Iseconds)" | tee -a "$GATE_LOG"

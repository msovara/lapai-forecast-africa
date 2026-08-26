#!/bin/bash
# Track B student stable v5: ONE focused tp-recovery attempt from v4.
# Keep Africa mix + input norm; stronger asymmetric precip loss; β/γ off.
set -euo pipefail
cd /local/Mthetho/lapai-forecast
export MKL_INTERFACE_LAYER=GNU
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
source /local/Mthetho/miniforge3/etc/profile.d/conda.sh
conda activate /local/Mthetho/envs/lapai-credit

LOG=/local/Mthetho/logs/trackB_student_stable_v5_train.log
GATE_LOG=/local/Mthetho/logs/trackB_stable_v5_gate.log
CKPT=models/student_global_stable_v5.ckpt
RESUME=models/student_global_stable_v4.ckpt
CFG=configs/student_global_v5.yaml

CACHE=data/processed/lapai/teacher_k1_cache_full2020_2021.zarr
if [[ ! -d "$CACHE" ]]; then
  echo "[trackB-stable-v5] missing cache $CACHE" | tee "$LOG"
  exit 1
fi

echo "[trackB-stable-v5] start $(date -Iseconds)" | tee "$LOG"
echo "[trackB-stable-v5] cache=$CACHE resume=$RESUME cfg=$CFG -> $CKPT" | tee -a "$LOG"
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)' | tee -a "$LOG"

if [[ ! -f "$RESUME" ]]; then
  echo "[trackB-stable-v5] missing resume ckpt $RESUME" | tee -a "$LOG"
  exit 1
fi

pip install -e . --no-deps -q

python -u training/train_student.py \
  --config "$CFG" \
  --cache "$CACHE" \
  --resume "$RESUME" \
  --epochs 80 \
  --device cuda \
  --batch_size 4 \
  --steps_per_epoch 64 \
  --out "$CKPT" \
  2>&1 | tee -a "$LOG"

echo "[trackB-stable-v5] train done $(date -Iseconds)" | tee -a "$LOG"

echo "[trackB-gate-v5] start $(date -Iseconds)" | tee "$GATE_LOG"
python -u evaluation/trackB_gate.py \
  --ckpt "$CKPT" \
  --cache "$CACHE" \
  --device cuda \
  --domain global \
  --scorecard_out reports/TRACKB_STUDENT_SCORECARD_V5.json \
  --gate_out reports/TRACKB_GATE_V5.json \
  2>&1 | tee -a "$GATE_LOG"
python -u evaluation/trackB_gate.py \
  --ckpt "$CKPT" \
  --cache "$CACHE" \
  --device cuda \
  --domain africa \
  --scorecard_out reports/TRACKB_STUDENT_SCORECARD_V5_AFRICA.json \
  --gate_out reports/TRACKB_GATE_V5_AFRICA.json \
  2>&1 | tee -a "$GATE_LOG"
echo "[trackB-gate-v5] done $(date -Iseconds)" | tee -a "$GATE_LOG"

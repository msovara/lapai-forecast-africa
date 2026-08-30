#!/bin/bash
# Track B v6-lite: t2m RMSE continue-train from frozen v5 (GPU1 only).
# Does NOT overwrite student_global_stable_v5.ckpt.
set -euo pipefail
cd /local/Mthetho/lapai-forecast
export MKL_INTERFACE_LAYER=GNU
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
source /local/Mthetho/miniforge3/etc/profile.d/conda.sh
conda activate /local/Mthetho/envs/lapai-credit

LOG=/local/Mthetho/logs/trackB_v6_t2m_rmse_train.log
CKPT=models/student_global_stable_v5_t2mRMSE.ckpt
RESUME=models/student_global_stable_v5.ckpt
CFG=configs/student_global_v6_t2m.yaml
CACHE=data/processed/lapai/teacher_k1_cache_full2020_2021.zarr

mkdir -p /local/Mthetho/logs

if [[ ! -d "$CACHE" ]]; then
  echo "[v6-t2m] missing cache $CACHE" | tee "$LOG"
  exit 1
fi
if [[ ! -f "$RESUME" ]]; then
  echo "[v6-t2m] missing resume ckpt $RESUME" | tee "$LOG"
  exit 1
fi

echo "[v6-t2m] start $(date -Iseconds)" | tee "$LOG"
echo "[v6-t2m] cache=$CACHE resume=$RESUME cfg=$CFG -> $CKPT" | tee -a "$LOG"
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)' | tee -a "$LOG"

pip install -e . --no-deps -q

python -u training/train_student.py \
  --config "$CFG" \
  --cache "$CACHE" \
  --resume "$RESUME" \
  --epochs 25 \
  --device cuda \
  --batch_size 4 \
  --steps_per_epoch 64 \
  --out "$CKPT" \
  2>&1 | tee -a "$LOG"

echo "[v6-t2m] train done $(date -Iseconds)" | tee -a "$LOG"
ls -la "$CKPT" | tee -a "$LOG"

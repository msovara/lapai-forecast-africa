#!/bin/bash
set -eo pipefail
cd /local/Mthetho/lapai-forecast
export MKL_INTERFACE_LAYER=GNU
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
source /local/Mthetho/miniforge3/etc/profile.d/conda.sh
conda activate /local/Mthetho/envs/lapai-credit
LOG=/local/Mthetho/logs/trackB_student_stable_train.log
echo "[trackB-stable] start $(date -Iseconds)" | tee "$LOG"
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)' | tee -a "$LOG"
python -u training/train_student.py \
  --config configs/student_global.yaml \
  --cache data/processed/lapai/teacher_k1_cache.zarr \
  --epochs 25 \
  --device cuda \
  --batch_size 4 \
  --steps_per_epoch 32 \
  --out models/student_global_stable.ckpt \
  2>&1 | tee -a "$LOG"
echo "[trackB-stable] done $(date -Iseconds)" | tee -a "$LOG"

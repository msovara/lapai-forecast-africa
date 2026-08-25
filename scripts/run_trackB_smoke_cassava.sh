#!/usr/bin/env bash
# Track B smoke on Cassava GPU1 — synthetic batches (no teacher cache required).
# Full distill needs lapai_cache Zarr + preferred lapai-credit env; see reports/TRACKB_START.md.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-GNU}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export KMP_DUPLICATE_LIB_OK=TRUE

source /local/Mthetho/miniforge3/etc/profile.d/conda.sh
# Prefer Track B env when present; else anemoi stack (torch+cuda) for smoke only.
set +u
if [[ -d /local/Mthetho/envs/lapai-credit ]]; then
  conda activate /local/Mthetho/envs/lapai-credit
else
  conda activate /local/Mthetho/envs/lapai-anemoi
fi
set -u
cd /local/Mthetho/lapai-forecast

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES} $(date -Is)"
nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-gpu=name,memory.used,memory.total --format=csv,noheader || true

test -e models/teacher_pruned.ckpt
echo "teacher_pruned -> $(readlink -f models/teacher_pruned.ckpt)"

python -u training/train_student.py \
  --config configs/student_global.yaml \
  --epochs "${EPOCHS:-1}" \
  --steps_per_epoch "${STEPS:-4}" \
  --batch_size "${BATCH:-2}" \
  --device cuda \
  --out "${OUT:-models/student_smoke.pt}"

echo "[Track B smoke] done $(date -Is)"

#!/usr/bin/env bash
# Track B held-out student 6h on Cassava (anemoi env: earthkit+netCDF4+torch+gcsfs).
set -euo pipefail
cd /local/Mthetho/lapai-forecast
export CUDA_VISIBLE_DEVICES=1
export MKL_INTERFACE_LAYER=GNU
source /local/Mthetho/miniforge3/etc/profile.d/conda.sh
conda activate /local/Mthetho/envs/lapai-anemoi

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
import scipy
print("scipy", scipy.__version__)
PY

pip install -e . --no-deps -q

LOG=/local/Mthetho/logs/trackB_held_out_jan2023_student.log
mkdir -p /local/Mthetho/logs
echo "=== $(date -u) student held-out start ===" | tee "$LOG"

python -u evaluation/trackB_held_out_jan2023.py \
  --skip_teacher \
  --no_era5 \
  --student_ckpt models/student_global_stable_v3.ckpt \
  --device cuda \
  --student_forecast_dir data/processed/trackB_student_heldout/forecasts \
  --out reports/TRACKB_HELD_OUT_JAN2023_STUDENT.json \
  2>&1 | tee -a "$LOG"

echo "EXIT:${PIPESTATUS[0]}" | tee -a "$LOG"

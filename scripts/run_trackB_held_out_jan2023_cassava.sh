#!/usr/bin/env bash
# Run Track B held-out Jan-2023 on Cassava GPU1.
# Prefer anemoi for netCDF4 + earthkit CDS; fall back / also use credit for torch if needed.
set -euo pipefail
cd /local/Mthetho/lapai-forecast
export CUDA_VISIBLE_DEVICES=1
export MKL_INTERFACE_LAYER=GNU
source /local/Mthetho/miniforge3/etc/profile.d/conda.sh

# Merge path: anemoi has netCDF4/earthkit/GCS; credit has the student torch stack.
# Try credit first if torch+cuda present with netcdf; else anemoi for teacher-only then credit for student.
conda activate /local/Mthetho/envs/lapai-credit
python -c "import torch; assert torch.cuda.is_available(); print('credit cuda', torch.cuda.get_device_name(0))"

# Ensure editable import
pip install -e . --no-deps -q

LOG=/local/Mthetho/logs/trackB_held_out_jan2023.log
mkdir -p /local/Mthetho/logs
echo "=== $(date -u) held-out start ===" | tee "$LOG"

# If netCDF4 missing in credit, install lightly or use anemoi for teacher JSON merge.
python - <<'PY' 2>>"$LOG" || true
import importlib.util
print("netCDF4", bool(importlib.util.find_spec("netCDF4")))
print("earthkit", bool(importlib.util.find_spec("earthkit")))
print("gcsfs", bool(importlib.util.find_spec("gcsfs")))
PY

python -u evaluation/trackB_held_out_jan2023.py \
  --student_ckpt models/student_global_stable_v3.ckpt \
  --device cuda \
  --out reports/TRACKB_HELD_OUT_JAN2023.json \
  2>&1 | tee -a "$LOG"

echo "EXIT:${PIPESTATUS[0]}" | tee -a "$LOG"

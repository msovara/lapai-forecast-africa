#!/bin/bash
# After mask fix + v4 ckpt: re-run cache gate + held-out Jan-2023 (with ERA5 if possible).
set -euo pipefail
cd /local/Mthetho/lapai-forecast
export MKL_INTERFACE_LAYER=GNU
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
source /local/Mthetho/miniforge3/etc/profile.d/conda.sh

CKPT=models/student_global_stable_v4.ckpt
CACHE=data/processed/lapai/teacher_k1_cache_full2020_2021.zarr
GATE_LOG=/local/Mthetho/logs/trackB_stable_v4_gate.log
HELD_LOG=/local/Mthetho/logs/trackB_held_out_v4.log

echo "[post-v4] start $(date -Iseconds)" | tee /local/Mthetho/logs/trackB_v4_post.log

# Gate uses credit env (torch)
conda activate /local/Mthetho/envs/lapai-credit
pip install -e . --no-deps -q
echo "[gate] global $(date -Iseconds)" | tee "$GATE_LOG"
python -u evaluation/trackB_gate.py \
  --ckpt "$CKPT" \
  --cache "$CACHE" \
  --device cuda \
  --domain global \
  --scorecard_out reports/TRACKB_STUDENT_SCORECARD_V4.json \
  --gate_out reports/TRACKB_GATE_V4.json \
  2>&1 | tee -a "$GATE_LOG"

echo "[gate] africa $(date -Iseconds)" | tee -a "$GATE_LOG"
python -u evaluation/trackB_gate.py \
  --ckpt "$CKPT" \
  --cache "$CACHE" \
  --device cuda \
  --domain africa \
  --scorecard_out reports/TRACKB_STUDENT_SCORECARD_V4_AFRICA.json \
  --gate_out reports/TRACKB_GATE_V4_AFRICA.json \
  2>&1 | tee -a "$GATE_LOG"

# Held-out needs anemoi (CDS/earthkit + often gcsfs)
conda activate /local/Mthetho/envs/lapai-anemoi
pip install -e . --no-deps -q
echo "[heldout] start $(date -Iseconds) ckpt=$CKPT" | tee "$HELD_LOG"

# Prefer full ERA5 GCS scoring; fall back to --no_era5 + teacher-field only.
set +e
python -u evaluation/trackB_held_out_jan2023.py \
  --student_ckpt "$CKPT" \
  --device cuda \
  --skip_teacher \
  --student_forecast_dir data/processed/trackB_student_heldout_v4/forecasts \
  --out reports/TRACKB_HELD_OUT_JAN2023_V4.json \
  2>&1 | tee -a "$HELD_LOG"
rc=${PIPESTATUS[0]}
set -e
if [[ $rc -ne 0 ]]; then
  echo "[heldout] ERA5 path failed (rc=$rc); retry --no_era5" | tee -a "$HELD_LOG"
  python -u evaluation/trackB_held_out_jan2023.py \
    --student_ckpt "$CKPT" \
    --device cuda \
    --skip_teacher \
    --no_era5 \
    --student_forecast_dir data/processed/trackB_student_heldout_v4/forecasts \
    --out reports/TRACKB_HELD_OUT_JAN2023_V4_STUDENT.json \
    2>&1 | tee -a "$HELD_LOG"
fi

echo "[post-v4] done $(date -Iseconds)" | tee -a /local/Mthetho/logs/trackB_v4_post.log

#!/usr/bin/env bash
# Track B held-out student leads {6,24} on Cassava GPU1 using ARCO ERA5 ICs (no CDS).
set -eo pipefail
cd /local/Mthetho/lapai-forecast
export CUDA_VISIBLE_DEVICES=1
export MKL_INTERFACE_LAYER=GNU
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-dummy}"
# Prefer ADC for code4earth truth; ARCO IC uses anon inside era5_ondisk_ic.
unset LAPAI_GCS_TOKEN || true

source /local/Mthetho/miniforge3/etc/profile.d/conda.sh
# anemoi has gcsfs (ARCO IC + optional GCS truth); credit lacks gcsfs.
# NOTE: avoid `set -u` around conda activate (MKL backup refs unbound vars).
conda activate /local/Mthetho/envs/lapai-anemoi
set -u
python -c "import torch,gcsfs; assert torch.cuda.is_available(); print('anemoi cuda', torch.cuda.get_device_name(0), 'gcsfs', gcsfs.__version__)"
pip install -e . --no-deps -q

CKPT=models/student_global_stable_v5.ckpt
if [[ ! -f "$CKPT" ]]; then
  CKPT=models/student_global_stable_v4.ckpt
fi
echo "Using ckpt=$CKPT"

LOG=/local/Mthetho/logs/trackB_held_out_jan2023_arco_l6_24.log
mkdir -p /local/Mthetho/logs data/cache/student_ic_arco
echo "=== $(date -u) held-out ARCO IC leads 6,24 start ===" | tee "$LOG"

python -u evaluation/trackB_held_out_jan2023.py \
  --student_ckpt "$CKPT" \
  --device cuda \
  --student_leads 6,24 \
  --ic_source arco \
  --skip_teacher \
  --student_forecast_dir data/processed/trackB_student_heldout_v5/forecasts \
  --out reports/TRACKB_HELD_OUT_JAN2023_V5.json \
  2>&1 | tee -a "$LOG"

echo "EXIT:${PIPESTATUS[0]}" | tee -a "$LOG"

#!/usr/bin/env bash
# Track B production t2m expanded eval on Cassava GPU1 (v5 freeze, ARCO IC).
# Analysis-forced leads 6/12/18/24 across Jan/Apr/Jul/Oct 2023.
# ~61 inits (every 2 days) — inside the 50–100 target; ARCO IC cache reuses prior hits.
set -eo pipefail
cd /local/Mthetho/lapai-forecast
export CUDA_VISIBLE_DEVICES=1
export MKL_INTERFACE_LAYER=GNU
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-dummy}"
unset LAPAI_GCS_TOKEN || true

source /local/Mthetho/miniforge3/etc/profile.d/conda.sh
# anemoi env has gcsfs for ARCO IC + truth
conda activate /local/Mthetho/envs/lapai-anemoi
set -u

python -c "import torch,gcsfs; assert torch.cuda.is_available(); print('cuda', torch.cuda.get_device_name(0), 'gcsfs', gcsfs.__version__)"
pip install -e . --no-deps -q

CKPT=models/student_global_stable_v5.ckpt
test -f "$CKPT"

LOG=/local/Mthetho/logs/trackB_t2m_expanded.log
mkdir -p /local/Mthetho/logs data/cache/student_ic_arco reports/figures
echo "=== $(date -u) trackB t2m expanded START ===" | tee "$LOG"

# Seasonal months only; skip per-lead NC writes (JSON + spatial figures are the deliverable).
python -u evaluation/trackB_t2m_expanded.py \
  --ckpt "$CKPT" \
  --device cuda \
  --ic_source arco \
  --months 1,4,7,10 \
  --step_days 2 \
  --leads 6,12,18,24 \
  --no_write_nc \
  --figures_dir reports/figures \
  --out_json reports/TRACKB_T2M_EXPANDED.json \
  --out_md reports/TRACKB_T2M_EXPANDED.md \
  2>&1 | tee -a "$LOG"

echo "EXIT:${PIPESTATUS[0]} $(date -u)" | tee -a "$LOG"

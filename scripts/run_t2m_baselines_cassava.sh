#!/usr/bin/env bash
# Score AF persistence (+ optional climatology) baselines on Cassava.
set -euo pipefail
cd /local/Mthetho/lapai-forecast
# shellcheck disable=SC1091
source /local/Mthetho/envs/lapai-credit/bin/activate || true
export MKL_INTERFACE_LAYER=GNU
export LAPAI_GCS_TOKEN=anon
export GOOGLE_CLOUD_PROJECT=dummy
# Avoid unbound-var traps from conda activate on some shells
set +u
LOG=/local/Mthetho/logs/trackB_t2m_baselines.log
mkdir -p "$(dirname "$LOG")"
echo "[baselines] start $(date -Is)" | tee -a "$LOG"
python -u evaluation/score_t2m_baselines.py \
  --expanded_json reports/TRACKB_T2M_EXPANDED.json \
  --with_climatology \
  --clim_years 2018,2019,2020,2021,2022 \
  2>&1 | tee -a "$LOG"
echo "[baselines] done $(date -Is)" | tee -a "$LOG"

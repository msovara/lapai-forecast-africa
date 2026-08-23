#!/usr/bin/env bash
# Rescore existing Track A FULL 240h forecasts vs public ARCO ERA5.
# Does not rerun forecasts. Does not touch smoke A2 artifacts.
export MKL_INTERFACE_LAYER=GNU
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export KMP_DUPLICATE_LIB_OK=TRUE
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-dummy}"
export LAPAI_GCS_TOKEN=anon
set -euo pipefail

source /local/Mthetho/miniforge3/etc/profile.d/conda.sh
conda activate /local/Mthetho/envs/lapai-anemoi
cd /local/Mthetho/lapai-forecast

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES} $(date -Is)"
echo "[A2 rescore full] CPU scoring vs ARCO ERA5 (anon GCS); CUDA_VISIBLE_DEVICES set only as a safety fence"

INITS=(20230101 20230108 20230115 20230122 20230129)
for init in "${INITS[@]}"; do
  f="data/processed/trackA_prune_full/forecasts/${init}_00Z.nc"
  if [[ ! -f "${f}" ]]; then
    echo "MISSING forecast ${f}" >&2
    exit 1
  fi
  bytes=$(stat -c%s "${f}")
  echo "[A2 rescore full] keep ${f} bytes=${bytes}"
  if [[ "${bytes}" -lt 1000000 ]]; then
    echo "forecast looks corrupt: ${f}" >&2
    exit 1
  fi
done

test -e models/teacher_pruned_full.ckpt
CKPT_REAL=$(readlink -f models/teacher_pruned_full.ckpt)
echo "[A2 rescore full] ckpt=${CKPT_REAL}"
echo "${CKPT_REAL}" | grep -q '31676a83-4919-4741-acaa-118fced72eab'
echo "${CKPT_REAL}" | grep -q 'trackA_prune_runs_full'

# Safety: do not write smoke paths
test -d data/processed/trackA_prune_full/forecasts
python -u scripts/rescore_trackA_full_arco.py
echo "[A2 rescore full] done $(date -Is)"

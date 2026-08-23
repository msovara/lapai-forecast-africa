#!/usr/bin/env bash
# A1b unpruned GT forecast (CDS IC) — fair baseline for A2 prune-only gate.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-GNU}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export KMP_DUPLICATE_LIB_OK=TRUE

source /local/Mthetho/miniforge3/etc/profile.d/conda.sh
conda activate /local/Mthetho/envs/lapai-anemoi
cd /local/Mthetho/lapai-forecast

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES} $(date -Is)"
test -e models/teacher_gt_coarsened.ckpt

if [[ ! -f data/processed/lapai/n320_latlons.npy ]]; then
  echo "[A1b fc] extract N320 lat/lons"
  python -u scripts/extract_grid_latlons.py \
    --ckpt models/teacher_n320_gt6/inference.ckpt \
    --out data/processed/lapai/n320_latlons.npy
fi

mkdir -p data/processed/trackA_gt_coarsen/forecasts
INITS=(20230101 20230108 20230115 20230122 20230129)

for init in "${INITS[@]}"; do
  out="data/processed/trackA_gt_coarsen/forecasts/${init}_00Z.nc"
  if [[ -f "${out}" ]]; then
    echo "[A1b fc] skip existing ${out}"
    continue
  fi
  echo "[A1b fc] forecasting ${init}"
  python -u scripts/run_phase0_forecast.py \
    --config configs/trackA_prune.yaml \
    --init "${init}" \
    --teacher-config configs/teacher_n320_gt6.yaml \
    --checkpoint models/teacher_gt_coarsened.ckpt \
    --ic-source cds \
    --cds-cache-dir /home/ubuntu/.cache/aifs-africa/era5 \
    --cds-offline \
    --output "${out}"
done

echo "[A1b fc] done $(date -Is)"

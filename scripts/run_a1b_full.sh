#!/usr/bin/env bash
# Export MKL before set -u / conda activate. Intel/conda scripts
# reference MKL_INTERFACE_LAYER and fail under nounset if unset.
export MKL_INTERFACE_LAYER=GNU
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
set -euo pipefail
source /local/Mthetho/miniforge3/etc/profile.d/conda.sh
conda activate /local/Mthetho/envs/lapai-anemoi
cd /local/Mthetho/lapai-forecast
echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES} $(date -Is)"
test -d data/processed/lapai/era5_n96_2020_2021.zarr
exec python -u training/train_trackA.py --step coarsen --config configs/trackA_gt_coarsen.yaml

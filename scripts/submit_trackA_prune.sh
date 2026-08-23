#!/usr/bin/env bash
# Sync-safe submit for A2 prune on Lengau.
set -euo pipefail
cd /home/msovara/repos/lapai-forecast || exit 1
CR=$(printf '\r')
for f in pbs/trackA_prune.pbs pbs/inc_conda_lengau.sh configs/trackA_prune.yaml training/train_trackA.py utils/head_prune.py; do
  [[ -f "$f" ]] && sed -i "s/${CR}\$//" "$f" || true
done

echo "=== prereqs ==="
ls -la models/teacher_gt_coarsened.ckpt
test -d data/processed/lapai/era5_n96_2020_2021.zarr
grep -E 'model:|dataset:|warm_start:|max_steps:|num_heads' configs/trackA_prune.yaml | head -20

# Prefer gpu2006 (A1b succeeded there); fall back gpu2005
HOST="${LAPAI_PRUNE_HOST:-gpu2006}"
sed -i "s/host=gpu200[56]/host=${HOST}/" pbs/trackA_prune.pbs
sed -i "s/${CR}\$//" pbs/trackA_prune.pbs
grep -E '^#PBS -l select|^#PBS -N' pbs/trackA_prune.pbs

echo "=== qsub ==="
qsub pbs/trackA_prune.pbs
sleep 2
qstat -u msovara

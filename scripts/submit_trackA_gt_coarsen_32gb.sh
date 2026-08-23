#!/usr/bin/env bash
# Dual-submit Track A A1b GraphTransformer O96 fine-tune (A2 prereq).
set -euo pipefail
cd /home/msovara/repos/lapai-forecast || exit 1
CR=$(printf '\r')
# Windows uploads leave CRLF; sourcing a CRLF .sh under set -e → exit 127 ($'\r').
for f in pbs/trackA_full.pbs pbs/inc_conda_lengau.sh configs/trackA_gt_coarsen.yaml; do
  sed -i "s/${CR}\$//" "$f" 2>/dev/null || true
done

export LAPAI_TRACKA_CONFIG=configs/trackA_gt_coarsen.yaml
export LAPAI_SKIP_GATE=1

echo "=== A1b GT coarsen config ==="
grep -E 'model:|max_steps|warm_start|num_heads|num_channels' "$LAPAI_TRACKA_CONFIG" | head -20
ls -la models/teacher_n320_gt6/inference.ckpt
test -d data/processed/lapai/era5_n96_2020_2021.zarr || {
  echo "ERROR: missing O96 Zarr" >&2
  exit 1
}

submit_one() {
  local host=$1
  local pbs="pbs/trackA_gt_${host}.pbs"
  cp pbs/trackA_full.pbs "$pbs"
  sed -i "s/host=gpu200[56]/host=${host}/" "$pbs"
  sed -i "s/${CR}\$//" "$pbs"
  sed -i "s/#PBS -N lapai_trackA_full/#PBS -N lapai_gtA1b_${host}/" "$pbs"
  echo "=== submit $host ==="
  grep -E '^#PBS -l select|^#PBS -N' "$pbs"
  qsub -v LAPAI_TRACKA_CONFIG,LAPAI_SKIP_GATE "$pbs"
}

submit_one gpu2005
submit_one gpu2006
sleep 2
qstat -u msovara | head -20

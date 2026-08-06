#!/usr/bin/env bash
# Dual-submit Track A EXTEND train on 32 GiB V100s (gpu2005 / gpu2006).
set -euo pipefail
cd /home/msovara/repos/lapai-forecast || exit 1

CR=$(printf '\r')
sed -i "s/${CR}\$//" pbs/trackA_full.pbs configs/trackA_coarsen_full_extend.yaml 2>/dev/null || true

export LAPAI_TRACKA_CONFIG=configs/trackA_coarsen_full_extend.yaml
export LAPAI_SKIP_GATE=1

echo "=== extend config ==="
grep -E 'max_steps|warm_start|output_root' "$LAPAI_TRACKA_CONFIG" | head -20

echo "=== ckpt present? ==="
ls -la models/teacher_coarsened.ckpt || {
  echo "ERROR: models/teacher_coarsened.ckpt missing" >&2
  exit 1
}

echo "=== 32GB occupancy ==="
for n in gpu2005 gpu2006; do
  echo "--- $n ---"
  pbsnodes "$n" 2>&1 | grep -E 'state =|resources_assigned.ngpus|resources_available.ngpus|jobs =' \
    | sed 's/^[[:space:]]*/  /' | cut -c1-200
done

submit_one() {
  local host=$1
  local pbs="pbs/trackA_full_${host}.pbs"
  cp pbs/trackA_full.pbs "$pbs"
  sed -i "s/host=gpu200[56]/host=${host}/" "$pbs"
  sed -i "s/${CR}\$//" "$pbs"
  sed -i "s/#PBS -N lapai_trackA_full/#PBS -N lapai_trkA_ext_${host}/" "$pbs"
  echo "=== submit $host ==="
  grep -E '^#PBS -l select|^#PBS -N' "$pbs"
  JID=$(qsub -v LAPAI_TRACKA_CONFIG,LAPAI_SKIP_GATE "$pbs")
  echo "job: $JID"
}

submit_one gpu2005
submit_one gpu2006

sleep 3
qstat -u msovara | head -30

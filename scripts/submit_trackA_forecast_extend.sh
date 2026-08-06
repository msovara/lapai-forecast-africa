#!/usr/bin/env bash
# Forecast with current teacher_coarsened.ckpt (extend 15k), dual-queue 32GB.
set -euo pipefail
cd /home/msovara/repos/lapai-forecast || exit 1
CR=$(printf '\r')

echo "ckpt -> $(readlink -f models/teacher_coarsened.ckpt)"
ls -la models/teacher_coarsened.ckpt

sed -i "s/${CR}\$//" pbs/trackA_forecast.pbs configs/trackA_coarsen_full.yaml

submit_for() {
  local host=$1
  local tmp=pbs/trackA_forecast_${host}.pbs
  sed "s/host=gpu200[56]/host=${host}/" pbs/trackA_forecast.pbs > "$tmp"
  sed -i "s/^#PBS -N .*/#PBS -N lapai_fc_ext_${host}/" "$tmp"
  sed -i "s/^#PBS -o .*/#PBS -o lapai_fc_ext_${host}.o/" "$tmp"
  sed -i "s/^#PBS -e .*/#PBS -e lapai_fc_ext_${host}.e/" "$tmp"
  sed -i "s/${CR}\$//" "$tmp"
  qsub "$tmp"
}

echo "=== occupancy ==="
for n in gpu2005 gpu2006; do
  ag=$(pbsnodes "$n" 2>&1 | grep 'resources_assigned.ngpus' | awk '{print $3}')
  echo "$n assigned_ngpus=${ag:-?} / 3"
done

J5=$(submit_for gpu2005); echo "gpu2005 -> $J5"
J6=$(submit_for gpu2006); echo "gpu2006 -> $J6"
sleep 3
qstat -u msovara | head -20

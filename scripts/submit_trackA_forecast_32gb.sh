#!/usr/bin/env bash
# Point teacher_coarsened.ckpt at the Aug 2026 full train, dual-queue forecast
# on gpu2005/gpu2006, keep whichever starts first.
cd /home/msovara/repos/lapai-forecast || exit 1
CR=$(printf '\r')

CKPT_NEW=models/trackA_coarsen_full_runs/checkpoint/5882930f-8b8c-4c32-af22-c5ca249c728d/inference-last.ckpt
if [[ ! -f "$CKPT_NEW" ]]; then
  echo "ERROR: missing $CKPT_NEW" >&2
  exit 1
fi

ln -sfn "$(pwd)/$CKPT_NEW" models/teacher_coarsened.ckpt
echo "teacher_coarsened.ckpt -> $(readlink -f models/teacher_coarsened.ckpt)"
ls -la models/teacher_coarsened.ckpt "$CKPT_NEW"

sed -i "s/${CR}\$//" pbs/trackA_forecast.pbs configs/trackA_coarsen_full.yaml

submit_for() {
  local host=$1
  local tmp=pbs/trackA_forecast_${host}.pbs
  sed "s/host=gpu200[56]/host=${host}/" pbs/trackA_forecast.pbs > "$tmp"
  sed -i "s/^#PBS -N .*/#PBS -N lapai_fc_${host}/" "$tmp"
  sed -i "s/^#PBS -o .*/#PBS -o lapai_fc_${host}.o/" "$tmp"
  sed -i "s/^#PBS -e .*/#PBS -e lapai_fc_${host}.e/" "$tmp"
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
J5N=${J5%%.*}; J6N=${J6%%.*}

WINNER=""; WHOST=""; LOSER=""
for i in $(seq 1 2160); do
  s5=$(qstat -x -f "$J5N" 2>/dev/null | grep -E '^\s*job_state' | awk '{print $3}')
  s6=$(qstat -x -f "$J6N" 2>/dev/null | grep -E '^\s*job_state' | awk '{print $3}')
  [ $((i % 6)) -eq 1 ] && echo "t=$((i*10))s gpu2005=$s5 gpu2006=$s6"
  if [ "$s5" = "R" ] || [ "$s5" = "E" ]; then WINNER=$J5N; LOSER=$J6N; WHOST=gpu2005; break; fi
  if [ "$s6" = "R" ] || [ "$s6" = "E" ]; then WINNER=$J6N; LOSER=$J5N; WHOST=gpu2006; break; fi
  if [ "$s5" = "F" ] && [ "$s6" = "F" ]; then
    echo BOTH_FINISHED_EARLY
    qstat -x -f "$J5N" "$J6N" 2>&1 | grep -Ei 'Exit_status|comment|exec_vnode' | sed 's/^[[:space:]]*/  /'
    exit 1
  fi
  sleep 10
done

if [ -z "$WINNER" ]; then
  echo TIMEOUT_WAITING_FOR_SLOT
  qstat -u msovara
  exit 2
fi

echo "WINNER=$WINNER HOST=$WHOST — deleting $LOSER"
qdel -W force "$LOSER" 2>/dev/null || qdel "$LOSER" 2>/dev/null || true

OUT=/home/msovara/lapai_fc_${WHOST}.o${WINNER}
for i in $(seq 1 720); do
  st=$(qstat -x -f "$WINNER" 2>/dev/null | grep -E '^\s*job_state' | awk '{print $3}')
  [ -f "$OUT" ] || OUT=/home/msovara/repos/lapai-forecast/lapai_fc_${WHOST}.o${WINNER}
  if [ -f "$OUT" ] && grep -qiE 'CUDA out of memory|OutOfMemoryError|Error|Traceback' "$OUT"; then
    # only hard-fail on OOM; print other errors but keep watching until F
    if grep -qiE 'CUDA out of memory|OutOfMemoryError' "$OUT"; then
      echo OOM_DETECTED
      tail -40 "$OUT" | tr '\r' '\n' | tail -40
      exit 3
    fi
  fi
  if [ $((i % 5)) -eq 1 ]; then
    echo "heartbeat t=$((i*30))s state=$st"
    [ -f "$OUT" ] && tr '\r' '\n' < "$OUT" | grep -E 'Init:|exec:|Wrote|forecast: date=|ERROR|OK' | tail -6
  fi
  if [ "$st" = "F" ]; then
    echo FORECAST_JOB_FINISHED
    qstat -x -f "$WINNER" 2>&1 | sed -e ':a' -e '$!N' -e 's/\n\t//' -e 'ta' -e 'P' -e 'D' \
      | grep -Ei 'Exit_status|comment|resources_used.walltime|exec_vnode' | sed 's/^[[:space:]]*/  /' | cut -c1-200
    echo "=== forecast NetCDFs ==="
    ls -la data/processed/trackA/forecasts/*_00Z.nc 2>/dev/null
    [ -f "$OUT" ] && tail -30 "$OUT" | tr '\r' '\n' | tail -30
    exit 0
  fi
  sleep 30
done
echo FORECAST_WATCH_TIMEOUT
exit 2

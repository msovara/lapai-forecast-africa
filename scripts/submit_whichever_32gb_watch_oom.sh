#!/usr/bin/env bash
# Submit Track A full train to BOTH 32 GiB hosts; keep whichever starts first.
cd /home/msovara/repos/lapai-forecast || exit 1
CR=$(printf '\r')
sed -i "s/${CR}\$//" pbs/trackA_full.pbs

# Drop the single-host queued job if still waiting
qdel -W force 7353484 2>/dev/null || qdel 7353484 2>/dev/null || true
sleep 2

submit_for() {
  local host=$1
  local tmp=pbs/trackA_full_${host}.pbs
  sed "s/host=gpu200[56]/host=${host}/" pbs/trackA_full.pbs > "$tmp"
  # unique job name so outputs don't collide
  sed -i "s/^#PBS -N .*/#PBS -N lapai_tA_${host}/" "$tmp"
  sed -i "s/^#PBS -o .*/#PBS -o lapai_tA_${host}.o/" "$tmp"
  sed -i "s/^#PBS -e .*/#PBS -e lapai_tA_${host}.e/" "$tmp"
  sed -i "s/${CR}\$//" "$tmp"
  local jid
  jid=$(qsub "$tmp")
  echo "$host $jid"
}

echo "=== occupancy ==="
for n in gpu2005 gpu2006; do
  ag=$(pbsnodes "$n" 2>&1 | grep 'resources_assigned.ngpus' | awk '{print $3}')
  echo "$n assigned_ngpus=$ag / 3"
done

echo "=== submit both ==="
read -r H5 J5 <<< "$(submit_for gpu2005)"
read -r H6 J6 <<< "$(submit_for gpu2006)"
echo "gpu2005 -> $J5"
echo "gpu2006 -> $J6"
J5N=${J5%%.*}
J6N=${J6%%.*}

# Wait until one is Running; delete the other
WINNER=""
for i in $(seq 1 180); do
  s5=$(qstat -x -f "$J5N" 2>/dev/null | grep -E '^\s*job_state' | awk '{print $3}')
  s6=$(qstat -x -f "$J6N" 2>/dev/null | grep -E '^\s*job_state' | awk '{print $3}')
  echo "t=${i}0s gpu2005=$s5 gpu2006=$s6"
  if [ "$s5" = "R" ] || [ "$s5" = "E" ]; then
    WINNER=$J5N; LOSER=$J6N; WHOST=gpu2005
    break
  fi
  if [ "$s6" = "R" ] || [ "$s6" = "E" ]; then
    WINNER=$J6N; LOSER=$J5N; WHOST=gpu2006
    break
  fi
  # both finished without running? stop
  if [ "$s5" = "F" ] && [ "$s6" = "F" ]; then
    echo "BOTH FINISHED without Running — check Exit_status"
    qstat -x -f "$J5N" "$J6N" 2>&1 | grep -Ei 'Job Id|job_state|Exit_status|comment|exec_vnode' | sed 's/^[[:space:]]*/  /'
    exit 1
  fi
  sleep 10
done

if [ -z "$WINNER" ]; then
  echo "TIMEOUT waiting for a slot (30 min). Jobs left queued: $J5 $J6"
  qstat -u msovara
  exit 2
fi

echo "WINNER=$WINNER on $WHOST — deleting loser $LOSER"
qdel -W force "$LOSER" 2>/dev/null || qdel "$LOSER" 2>/dev/null || true
sleep 3
qstat -u msovara

# Watch for OOM / first training progress (up to ~40 min after start)
OUT=""
for cand in \
  "/home/msovara/lapai_tA_${WHOST}.o${WINNER}" \
  "/home/msovara/repos/lapai-forecast/lapai_tA_${WHOST}.o${WINNER}" \
  "/home/msovara/repos/lapai-forecast/lapai_tA_${WHOST}.o"; do
  [ -f "$cand" ] && OUT=$cand
done

echo "=== OOM / progress watch (winner=$WINNER host=$WHOST) ==="
seen_train=0
for i in $(seq 1 120); do
  st=$(qstat -x -f "$WINNER" 2>/dev/null | grep -E '^\s*job_state' | awk '{print $3}')
  # refresh output path
  for cand in \
    "/home/msovara/lapai_tA_${WHOST}.o${WINNER}" \
    "/home/msovara/repos/lapai-forecast/lapai_tA_${WHOST}.o${WINNER}" \
    "/home/msovara/repos/lapai-forecast/lapai_tA_${WHOST}.o"; do
    [ -f "$cand" ] && OUT=$cand
  done

  if [ -n "$OUT" ] && [ -f "$OUT" ]; then
    if grep -qiE 'CUDA out of memory|OutOfMemoryError|torch.cuda.OutOfMemoryError|oom_kill|Killed' "$OUT"; then
      echo "OOM_DETECTED in $OUT"
      tail -60 "$OUT"
      exit 3
    fi
    if grep -qE 'Epoch |max_steps|train_multi_dataset_loss|Saving.*checkpoint|Trainer.fit' "$OUT"; then
      if [ "$seen_train" -eq 0 ]; then
        echo "TRAINING_STARTED"
        grep -E 'Epoch |max_steps|train_multi_dataset_loss|host=|cuda_available|ERROR|Error' "$OUT" | tail -30
        seen_train=1
      fi
    fi
    # periodic heartbeat
    if [ $((i % 6)) -eq 0 ]; then
      echo "--- heartbeat t=${i}0s state=$st bytes=$(wc -c < "$OUT") ---"
      tail -5 "$OUT" | tr '\r' '\n' | tail -5
    fi
  else
    echo "t=${i}0s state=$st (waiting for output file)"
  fi

  if [ "$st" = "F" ]; then
    echo "JOB_FINISHED"
    qstat -x -f "$WINNER" 2>&1 | sed -e ':a' -e '$!N' -e 's/\n\t//' -e 'ta' -e 'P' -e 'D' \
      | grep -Ei 'job_state|Exit_status|exec_vnode|comment|resources_used.walltime' | sed 's/^[[:space:]]*/  /' | cut -c1-200
    if [ -n "$OUT" ]; then
      echo "--- last 80 lines of $OUT ---"
      tail -80 "$OUT" | tr '\r' '\n' | tail -80
      if grep -qiE 'CUDA out of memory|OutOfMemoryError|torch.cuda.OutOfMemoryError' "$OUT"; then
        echo "OOM_DETECTED_AT_END"
        exit 3
      fi
    fi
    exit 0
  fi
  sleep 10
done

echo "WATCH_TIMEOUT after ~20 min of running — job still $st; no OOM seen in output so far"
[ -n "$OUT" ] && tail -30 "$OUT" | tr '\r' '\n' | tail -30
echo "WINNER_JOB=$WINNER HOST=$WHOST OUT=$OUT"

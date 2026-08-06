#!/usr/bin/env bash
# Continue wait for dual-queued Track A jobs; keep winner; watch OOM.
J5=7353532
J6=7353533
cd /home/msovara/repos/lapai-forecast || exit 1

echo "=== who holds the 32GB GPUs ==="
for n in gpu2005 gpu2006; do
  echo "--- $n ---"
  pbsnodes "$n" 2>&1 | grep -E 'state =|resources_assigned.ngpus|jobs =' | sed 's/^[[:space:]]*/  /' | cut -c1-220
done
echo
echo "=== their walltimes / elapsed ==="
for j in $(pbsnodes gpu2005 gpu2006 2>/dev/null | grep -oE '[0-9]+\.sched01' | sort -u); do
  qstat -f "$j" 2>/dev/null | sed -e ':a' -e '$!N' -e 's/\n\t//' -e 'ta' -e 'P' -e 'D' \
    | grep -Ei 'Job Id|Job_Name|Job_Owner|job_state|Resource_List.walltime|resources_used.walltime|exec_vnode' \
    | sed 's/^[[:space:]]*/  /' | cut -c1-180
  echo
done

echo "=== wait for 7353532 or 7353533 to start (up to 6h) ==="
WINNER=""; WHOST=""; LOSER=""
for i in $(seq 1 2160); do
  s5=$(qstat -x -f "$J5" 2>/dev/null | grep -E '^\s*job_state' | awk '{print $3}')
  s6=$(qstat -x -f "$J6" 2>/dev/null | grep -E '^\s*job_state' | awk '{print $3}')
  if [ $((i % 6)) -eq 1 ]; then
    echo "t=$((i*10))s gpu2005=$s5 gpu2006=$s6"
  fi
  if [ "$s5" = "R" ] || [ "$s5" = "E" ]; then
    WINNER=$J5; LOSER=$J6; WHOST=gpu2005; break
  fi
  if [ "$s6" = "R" ] || [ "$s6" = "E" ]; then
    WINNER=$J6; LOSER=$J5; WHOST=gpu2006; break
  fi
  if [ "$s5" = "F" ] && [ "$s6" = "F" ]; then
    echo "BOTH_FINISHED_WITHOUT_RUN"
    qstat -x -f "$J5" "$J6" 2>&1 | grep -Ei 'Job Id|Exit_status|comment|exec_vnode' | sed 's/^[[:space:]]*/  /'
    exit 1
  fi
  # if one finished failed while other still Q, keep waiting on the Q one
  sleep 10
done

if [ -z "$WINNER" ]; then
  echo "TIMEOUT_6H still queued"
  qstat -u msovara
  exit 2
fi

echo "WINNER=$WINNER HOST=$WHOST — deleting loser $LOSER"
qdel -W force "$LOSER" 2>/dev/null || qdel "$LOSER" 2>/dev/null || true

# OOM watch for up to 12h walltime (poll every 30s)
OUT=""
seen_train=0
for i in $(seq 1 1440); do
  st=$(qstat -x -f "$WINNER" 2>/dev/null | grep -E '^\s*job_state' | awk '{print $3}')
  for cand in \
    "/home/msovara/lapai_tA_${WHOST}.o${WINNER}" \
    "/home/msovara/repos/lapai-forecast/lapai_tA_${WHOST}.o${WINNER}" \
    "/home/msovara/repos/lapai-forecast/lapai_tA_${WHOST}.o"; do
    [ -f "$cand" ] && OUT=$cand
  done

  if [ -n "$OUT" ] && [ -f "$OUT" ]; then
    if grep -qiE 'CUDA out of memory|OutOfMemoryError|torch.cuda.OutOfMemoryError|oom_kill' "$OUT"; then
      echo "OOM_DETECTED"
      tail -80 "$OUT" | tr '\r' '\n' | tail -80
      exit 3
    fi
    if [ "$seen_train" -eq 0 ] && grep -qE 'Epoch |train_multi_dataset_loss|Trainer.fit|max_steps' "$OUT"; then
      echo "TRAINING_STARTED host=$WHOST job=$WINNER"
      grep -E 'host=|cuda_available|Epoch |train_multi_dataset_loss|ERROR|Error|OOM' "$OUT" | tr '\r' '\n' | tail -40
      seen_train=1
    fi
    if [ $((i % 10)) -eq 0 ]; then
      echo "heartbeat t=$((i*30))s state=$st bytes=$(wc -c < "$OUT") last:"
      tail -3 "$OUT" | tr '\r' '\n' | tail -3
    fi
  else
    [ $((i % 10)) -eq 0 ] && echo "heartbeat t=$((i*30))s state=$st (no output yet)"
  fi

  if [ "$st" = "F" ]; then
    echo "JOB_FINISHED"
    qstat -x -f "$WINNER" 2>&1 | sed -e ':a' -e '$!N' -e 's/\n\t//' -e 'ta' -e 'P' -e 'D' \
      | grep -Ei 'Exit_status|exec_vnode|comment|resources_used.walltime' | sed 's/^[[:space:]]*/  /' | cut -c1-200
    [ -n "$OUT" ] && tail -60 "$OUT" | tr '\r' '\n' | tail -60
    grep -qiE 'CUDA out of memory|OutOfMemoryError' "$OUT" 2>/dev/null && echo OOM_AT_END && exit 3
    exit 0
  fi
  sleep 30
done
echo "WATCH_ENDED still running; no OOM seen"
echo "WINNER=$WINNER HOST=$WHOST OUT=$OUT"

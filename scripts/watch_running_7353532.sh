#!/usr/bin/env bash
# Watch running Track A full train for OOM / completion.
J=7353532
OUT=/home/msovara/lapai_tA_gpu2005.o7353532
echo "watching job=$J out=$OUT"
for i in $(seq 1 1440); do
  st=$(qstat -x -f "$J" 2>/dev/null | grep -E '^\s*job_state' | awk '{print $3}')
  if [ -f "$OUT" ]; then
    if grep -qiE 'CUDA out of memory|OutOfMemoryError|torch.cuda.OutOfMemoryError|oom_kill' "$OUT"; then
      echo "OOM_DETECTED"
      tail -60 "$OUT" | tr '\r' '\n' | tail -60
      exit 3
    fi
    if [ $((i % 5)) -eq 1 ]; then
      echo "heartbeat t=$((i*30))s state=$st bytes=$(wc -c < "$OUT")"
      tr '\r' '\n' < "$OUT" | grep -E 'Epoch |train_multi_dataset_loss|ERROR|Error|Saving.*checkpoint|max_steps' | tail -4
    fi
  else
    echo "heartbeat t=$((i*30))s state=$st (no out yet)"
  fi
  if [ "$st" = "F" ]; then
    echo "JOB_FINISHED"
    qstat -x -f "$J" 2>&1 | sed -e ':a' -e '$!N' -e 's/\n\t//' -e 'ta' -e 'P' -e 'D' \
      | grep -Ei 'Exit_status|exec_vnode|comment|resources_used.walltime' | sed 's/^[[:space:]]*/  /' | cut -c1-200
    tr '\r' '\n' < "$OUT" | tail -40
    grep -qiE 'CUDA out of memory|OutOfMemoryError' "$OUT" && echo OOM_AT_END && exit 3
    exit 0
  fi
  sleep 30
done
echo "WATCH_TIMEOUT still state=$st"
exit 2

#!/bin/bash
# Usage: poll_job.sh <JOBID> <max_iters> <sleep_secs>
JOBID="${1:-7313652}"
MAX="${2:-20}"
SLP="${3:-15}"
for i in $(seq 1 "$MAX"); do
    echo "--- poll $i $(date '+%H:%M:%S') ---"
    LINE=$(qstat "$JOBID" 2>/dev/null | tail -1)
    if [ -z "$LINE" ] || echo "$LINE" | grep -qi 'unknown'; then
        echo "job $JOBID no longer in queue (finished or unknown)"
        break
    fi
    echo "$LINE"
    STATE=$(echo "$LINE" | awk '{print $5}')
    if [ "$STATE" = "R" ]; then
        echo "RUNNING — listing partial output:"
        ls -lt /home/msovara/lustre/anemoi-weather-quest/train_jra3q_anemoi_gpu_smoke.o 2>/dev/null || true
        tail -30 /home/msovara/lustre/anemoi-weather-quest/train_jra3q_anemoi_gpu_smoke.o 2>/dev/null || true
        break
    fi
    sleep "$SLP"
done

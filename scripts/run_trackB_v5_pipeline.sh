#!/bin/bash
# One-shot Track B v5: train tp recovery → gate → held-out (ACC/POD).
set -euo pipefail
cd /local/Mthetho/lapai-forecast
LOG=/local/Mthetho/logs/trackB_v5_pipeline.log

echo "[pipeline-v5] start $(date -Iseconds)" | tee "$LOG"
bash scripts/run_trackB_stable_v5.sh 2>&1 | tee -a "$LOG"
bash scripts/run_trackB_v5_post_eval.sh 2>&1 | tee -a "$LOG"
echo "[pipeline-v5] done $(date -Iseconds)" | tee -a "$LOG"

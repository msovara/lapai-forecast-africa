#!/usr/bin/env bash
set -euo pipefail
cd ~/repos/lapai-forecast
export LAPAI_PYTHON="${LAPAI_PYTHON:-/home/msovara/lustre/dev/lapai-anemoi/bin/python}"

echo "=== tracejob 7357693 (exit) ==="
tracejob -n 2 7357693 2>&1 | grep -Ei 'Exit_status|Job queued|started|ended|resources_used' | tail -30 || true

echo "=== find logs ==="
find . -maxdepth 2 -name 'lapai_*' -type f 2>/dev/null | head -40
ls -lt ~/lapai_* 2>/dev/null | head -10 || true

CKPT_DIR=models/trackA_coarsen_full_extend_runs/checkpoint/b828d680-6d0e-4665-a5da-b9d0c8d56649
echo "=== ckpt dir ==="
ls -la "$CKPT_DIR"

LAST=""
if [ -f "$CKPT_DIR/inference-last.ckpt" ]; then
  LAST="$CKPT_DIR/inference-last.ckpt"
elif ls "$CKPT_DIR"/inference-anemoi-by_time*.ckpt >/dev/null 2>&1; then
  LAST=$(ls -t "$CKPT_DIR"/inference-anemoi-by_time*.ckpt | head -1)
fi
echo "LAST=$LAST"
if [ -z "$LAST" ] || [ ! -f "$LAST" ]; then
  echo "ERROR: no inference ckpt in extend run" >&2
  exit 1
fi

# Point teacher_coarsened at new extend ckpt (absolute)
ABS=$(readlink -f "$LAST")
ln -sfn "$ABS" models/teacher_coarsened.ckpt
echo "=== teacher_coarsened now ==="
ls -la models/teacher_coarsened.ckpt
readlink -f models/teacher_coarsened.ckpt

# Confirm step count from filename if present
basename "$ABS"

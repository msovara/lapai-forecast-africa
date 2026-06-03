#!/usr/bin/env bash
# Local / login-node gate before GPU inference (LapAI Phase 0).
# Run from anywhere:
#   bash scripts/lapai_inference_gate.sh
#   bash scripts/lapai_inference_gate.sh configs/teacher_n320_gt6.yaml
#   LAPAI_TEACHER_CONFIG=configs/teacher_n320_gt6.yaml bash scripts/lapai_inference_gate.sh
#
# Steps: smoke ckpt → strict verify stack → inference dry-run.
# Requires: conda env lapai-anemoi (teacher weights on disk, anemoi-inference on PATH).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
trap 'echo "[lapai] gate FAILED. See reports/RUNBOOK_BASELINE_LENGAU.md section 7."' ERR
cd "$ROOT"

# Teacher config: first positional arg > env LAPAI_TEACHER_CONFIG > scripts' own default.
TEACHER_CONFIG="${1:-${LAPAI_TEACHER_CONFIG:-}}"
if [[ -n "$TEACHER_CONFIG" ]]; then
  echo "[lapai] teacher config: $TEACHER_CONFIG"
fi

python scripts/smoke_teacher_ckpt.py ${TEACHER_CONFIG:+--config "$TEACHER_CONFIG"}
python scripts/verify_inference_stack.py --strict ${TEACHER_CONFIG:+--config "$TEACHER_CONFIG"}
python scripts/run_aifs_inference.py --dry-run ${TEACHER_CONFIG:+--teacher-config "$TEACHER_CONFIG"}

echo "OK: gate passed. Next: python scripts/run_aifs_inference.py  or  qsub pbs/inference_aifs_teacher.pbs"

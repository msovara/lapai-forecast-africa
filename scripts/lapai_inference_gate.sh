#!/usr/bin/env bash
# Local / login-node gate before GPU inference (LapAI Phase 0 / Track A).
# Run from repo root:
#   bash scripts/lapai_inference_gate.sh
#   LAPAI_TEACHER_CONFIG=configs/teacher_n320_gt6.yaml bash scripts/lapai_inference_gate.sh
#
# Uses LAPAI_PYTHON when set (Lengau lustre venv); otherwise `python` on PATH.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
trap 'echo "[lapai] gate FAILED. See reports/RUNBOOK_BASELINE_LENGAU.md section 7."' ERR
cd "$ROOT"

LAPAI_PYTHON="${LAPAI_PYTHON:-python}"
if ! command -v "${LAPAI_PYTHON}" >/dev/null 2>&1; then
  echo "[lapai] LAPAI_PYTHON not found: ${LAPAI_PYTHON}" >&2
  exit 1
fi
export PATH="$(dirname "${LAPAI_PYTHON}"):${PATH}"

TEACHER_CONFIG="${1:-${LAPAI_TEACHER_CONFIG:-}}"
if [[ -n "$TEACHER_CONFIG" ]]; then
  echo "[lapai] teacher config: $TEACHER_CONFIG"
fi
echo "[lapai] python: ${LAPAI_PYTHON}"

"${LAPAI_PYTHON}" scripts/smoke_teacher_ckpt.py ${TEACHER_CONFIG:+--config "$TEACHER_CONFIG"}
"${LAPAI_PYTHON}" scripts/verify_inference_stack.py --strict ${TEACHER_CONFIG:+--config "$TEACHER_CONFIG"}
"${LAPAI_PYTHON}" scripts/run_aifs_inference.py --dry-run ${TEACHER_CONFIG:+--teacher-config "$TEACHER_CONFIG"}

echo "OK: gate passed. Next: python scripts/run_aifs_inference.py  or  qsub pbs/inference_aifs_teacher.pbs"

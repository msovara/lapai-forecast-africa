#!/usr/bin/env bash
# Local / login-node gate before GPU inference (LapAI Phase 0).
# Run from anywhere:
#   bash scripts/lapai_inference_gate.sh
#
# Steps: smoke ckpt → strict verify stack → inference dry-run.
# Requires: conda env lapai-anemoi (teacher weights on disk, anemoi-inference on PATH).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python scripts/smoke_teacher_ckpt.py
python scripts/verify_inference_stack.py --strict
python scripts/run_aifs_inference.py --dry-run

echo "OK: gate passed. Next: python scripts/run_aifs_inference.py  or  qsub pbs/inference_aifs_teacher.pbs"

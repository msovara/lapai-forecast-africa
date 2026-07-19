#!/usr/bin/env bash
# Offline-friendly Phase 0 readiness check on Lengau (no git/network required).
set -euo pipefail

REPO="${LAPAI_REPO:-$HOME/repos/lapai-forecast}"
cd "$REPO" || { echo "FAIL: missing repo $REPO"; exit 1; }

echo "=== LapAI Phase 0 verification ==="
echo "host: $(hostname)"
echo "repo: $REPO"
echo

FAIL=0
check_file() {
  if [[ -f "$1" ]]; then
    echo "OK   $1"
  else
    echo "MISS $1"
    FAIL=1
  fi
}

check_file configs/phase0_baseline.yaml
check_file scripts/run_phase0_forecast.py
check_file scripts/run_phase0_closure.py
check_file pbs/phase0_baseline_lengau.pbs
check_file evaluation/phase0_scorecard.py
check_file evaluation/gcs_era5_truth.py
check_file utils/eval_forecast_io.py
check_file reports/PHASE0_CLOSURE.md
echo

CKPT_REPO="$REPO/models/teacher_n320_gt6/inference.ckpt"
CKPT_HOME="$HOME/models/teacher_n320_gt6/inference.ckpt"
if [[ -f "$CKPT_REPO" ]]; then
  echo "OK   checkpoint $CKPT_REPO ($(du -h "$CKPT_REPO" | cut -f1))"
elif [[ -f "$CKPT_HOME" ]]; then
  echo "WARN checkpoint not in repo; found $CKPT_HOME ($(du -h "$CKPT_HOME" | cut -f1))"
  echo "     symlink or copy: mkdir -p models/teacher_n320_gt6 && ln -sf $CKPT_HOME models/teacher_n320_gt6/inference.ckpt"
else
  echo "MISS checkpoint (expected $CKPT_REPO or $CKPT_HOME)"
  FAIL=1
fi
echo

if [[ -f scripts/run_n320_gt6_opendata_forecast.py ]]; then
  echo "OK   scripts/run_n320_gt6_opendata_forecast.py"
else
  echo "MISS scripts/run_n320_gt6_opendata_forecast.py (required by run_phase0_forecast.py)"
  FAIL=1
fi

if grep -q '_VAR_ALIASES' evaluation/eval_skill.py 2>/dev/null; then
  echo "OK   evaluation/eval_skill.py alias patch"
else
  echo "WARN evaluation/eval_skill.py missing 2t->t2m alias (scp updated eval_skill.py from Windows)"
fi
echo

module purge 2>/dev/null || true
module load chpc/python/anaconda/3-2024.10.1 2>/dev/null || true
# shellcheck source=/dev/null
source /home/apps/chpc/bio/anaconda3-2024.10.1/etc/profile.d/conda.sh
conda activate lapai-anemoi

echo "=== conda lapai-anemoi ==="
python --version
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
from evaluation.phase0_scorecard import load_phase0_config
c = load_phase0_config()
print("inits", c.get("inits"))
print("lead_time_hours", c.get("lead_time_hours"))
PY
echo

echo "=== dry-run closure plan ==="
python scripts/run_phase0_closure.py --dry-run
echo

FC="../aifs-africa/output/20230101_00Z.nc"
if [[ -f "$FC" ]]; then
  echo "OK   partial forecast $FC ($(du -h "$FC" | cut -f1))"
  echo "     score-only smoke (needs GCS if truth=gcs):"
  echo "     python scripts/run_phase0_closure.py --score-only --forecast-dir ../aifs-africa/output --inits 20230101 --leads 24"
else
  echo "INFO no smoke forecast at $FC"
fi
echo

if [[ "$FAIL" -eq 0 ]]; then
  echo "RESULT: READY (offline files OK). GPU job: qsub pbs/phase0_baseline_lengau.pbs"
  echo "NOTE: full inference needs GPU node; GCS scoring needs outbound network when available."
else
  echo "RESULT: NOT READY — fix MISS items above."
  exit 1
fi

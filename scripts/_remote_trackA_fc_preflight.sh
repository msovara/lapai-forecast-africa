#!/bin/bash
# Track A forecast preflight: verify coarsened ckpt, grid-bridge coords, CDS cache,
# and Phase 0 baseline before submitting pbs/trackA_forecast.pbs. Safe (read-only
# except for the teacher_coarsened.ckpt symlink).
set +e
R="$HOME/repos/lapai-forecast"
echo "REPO=$R"
if [[ ! -d "$R" ]]; then
  echo "NO_REPO at $R"
  exit 2
fi
cd "$R" || exit 2

echo "--- coarsen run ckpts (newest first)"
CK=$(ls -t models/trackA_coarsen_runs/checkpoint/*/inference-last.ckpt 2>/dev/null | head -1)
ls -t models/trackA_coarsen_runs/checkpoint/*/inference-last.ckpt 2>&1 | head -3
if [[ -n "$CK" ]]; then
  ln -sf "$(readlink -f "$CK")" models/teacher_coarsened.ckpt
  echo "SYMLINKED teacher_coarsened.ckpt -> $CK"
else
  echo "NO_COARSENED_CKPT"
fi
echo "--- coarsened link"
ls -lh models/teacher_coarsened.ckpt 2>&1
readlink -f models/teacher_coarsened.ckpt 2>&1

echo "--- n320_latlons (grid bridge source coords)"
ls -lh data/processed/lapai/n320_latlons.npy 2>&1

echo "--- CDS cache (offline ICs)"
ls ~/.cache/aifs-africa/era5 2>&1 | head -20
du -sh ~/.cache/aifs-africa/era5 2>&1

echo "--- Phase 0 baseline scorecard"
ls -lh reports/PHASE0_BASELINE_SCORECARD*.json 2>&1

echo "--- Phase 0 forecasts present"
ls -1 data/processed/phase0/forecasts/*.nc 2>&1 | head

echo "--- forecast output dir"
ls -1 data/processed/trackA/forecasts/*.nc 2>&1 | head
mkdir -p data/processed/trackA/forecasts && echo "ensured data/processed/trackA/forecasts"

echo "--- pbs script python target"
grep -n "LAPAI_PYTHON" pbs/trackA_forecast.pbs 2>&1 | head -3

echo "DONE_PREFLIGHT"

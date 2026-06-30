#!/bin/bash
set -euo pipefail
cd ~/repos/lapai-forecast
CK=$(ls -t models/trackA_coarsen_runs/checkpoint/*/inference-last.ckpt 2>/dev/null | head -1)
echo "CKPT=${CK}"
if [[ -z "${CK}" ]]; then
  echo "ERROR: no inference-last.ckpt found" >&2
  exit 1
fi
ln -sf "$(readlink -f "${CK}")" models/teacher_coarsened.ckpt
ls -lh models/teacher_coarsened.ckpt
echo "--- CDS cache ---"
if [[ -d ~/.cache/aifs-africa/era5 ]]; then
  ls ~/.cache/aifs-africa/era5 | head
  du -sh ~/.cache/aifs-africa/era5
else
  echo "MISSING: ~/.cache/aifs-africa/era5"
fi
echo "--- track forecast pbs ---"
ls pbs/ | grep -i track || echo "(none)"
echo "--- phase0 forecasts present ---"
ls -1 data/processed/phase0/forecasts/*.nc 2>/dev/null | head || echo "(none)"

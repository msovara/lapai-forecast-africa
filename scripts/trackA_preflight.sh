#!/bin/bash
# Track A submit preflight: confirm dataset, warm-start ckpt, env, and PBS are in place.
set -uo pipefail

REPO="${LAPAI_REPO:-$HOME/repos/lapai-forecast}"
cd "$REPO" || { echo "MISSING repo: $REPO"; exit 1; }
echo "=== repo: $REPO ==="

echo "--- smoke zarr dataset ---"
DS="data/processed/lapai/era5_n96_smoke.zarr"
if [ -d "$DS" ]; then
  echo "  FOUND $DS"
  du -sh "$DS" 2>/dev/null | sed 's/^/  /'
  ls "$DS" | sed 's/^/    /' | head -20
else
  echo "  MISSING $DS"
fi

echo "--- teacher inference checkpoint ---"
CK="models/teacher_n320_gt6/inference.ckpt"
if [ -f "$CK" ]; then
  echo "  FOUND $CK"
  ls -lh "$CK" | sed 's/^/  /'
else
  echo "  MISSING $CK"
fi

echo "--- trackA.pbs python target ---"
grep -n "LAPAI_PYTHON" pbs/trackA.pbs | sed 's/^/  /'

echo "--- queue state ---"
qstat -u msovara 2>/dev/null | tail -10

echo "=== preflight done ==="

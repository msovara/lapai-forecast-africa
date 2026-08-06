#!/usr/bin/env bash
# Preflight for the full Track A train on Lengau: normalise line endings on the
# freshly-copied files and confirm every prerequisite trackA_full.pbs will touch.
cd /home/msovara/repos/lapai-forecast || exit 1

CR=$(printf '\r')

echo "=== normalise CRLF on uploaded files ==="
for f in configs/trackA_coarsen_full.yaml pbs/trackA_full.pbs; do
  sed -i "s/${CR}\$//" "$f"
  n=$(grep -c "${CR}" "$f")
  echo "  $f : $n CR remaining"
done
echo

echo "=== md5 (compare against laptop) ==="
md5sum configs/trackA_coarsen_full.yaml pbs/trackA_full.pbs
echo

echo "=== training Zarr via repo path ==="
ls -la data/processed/lapai/era5_n96_2020_2021.zarr
test -d data/processed/lapai/era5_n96_2020_2021.zarr \
  && echo "  -d guard: PASS" || echo "  -d guard: FAIL"
echo

echo "=== teacher warm-start ==="
ls -la models/teacher_n320_gt6/ 2>&1
echo

echo "=== existing coarsened ckpt (full run will overwrite this path) ==="
ls -la models/teacher_coarsened.ckpt 2>&1
echo

echo "=== PBS support scripts ==="
for f in pbs/inc_conda_lengau.sh scripts/lapai_inference_gate.sh training/train_trackA.py; do
  if [ -f "$f" ]; then
    echo "  $f : PRESENT ($(grep -c "${CR}" "$f") CR)"
  else
    echo "  $f : MISSING"
  fi
done
echo

echo "=== lustre python env ==="
P=/home/msovara/lustre/dev/lapai-anemoi/bin/python
if [ -x "$P" ]; then
  echo "  $P : executable"
  "$P" -c 'import anemoi.training as t; print("  anemoi-training", t.__version__)' 2>&1 | tail -2
  "$P" -c 'import torch; print("  torch", torch.__version__)' 2>&1 | tail -1
else
  echo "  $P : NOT EXECUTABLE"
fi
echo

echo "=== baseline scorecard for the gate ==="
ls -la reports/PHASE0_BASELINE_SCORECARD.json 2>&1
echo

echo "=== queue ==="
qstat -u "$USER" 2>&1 | tail -5
echo "(empty = nothing queued)"

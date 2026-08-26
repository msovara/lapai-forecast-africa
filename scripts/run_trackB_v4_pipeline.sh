#!/bin/bash
# Wait for full-year cache, then train v4 + gate + held-out student 6h.
set -euo pipefail
cd /local/Mthetho/lapai-forecast
LOG=/local/Mthetho/logs/trackB_v4_pipeline.log
CACHE=data/processed/lapai/teacher_k1_cache_full2020_2021.zarr
MARKER=/local/Mthetho/logs/trackB_teacher_cache_full2020_2021.log

echo "[pipeline-v4] start $(date -Iseconds)" | tee "$LOG"

echo "[pipeline-v4] waiting for cache builder..." | tee -a "$LOG"
while pgrep -f "training/build_teacher_feature_cache.py" >/dev/null 2>&1 \
   || pgrep -f "run_trackB_cache_full2020_2021.sh" >/dev/null 2>&1; do
  tail -1 "$MARKER" 2>/dev/null | tee -a "$LOG" || true
  sleep 120
done

if [[ ! -d "$CACHE/lapai_cache" ]] && [[ ! -d "$CACHE" ]]; then
  echo "[pipeline-v4] ERROR: missing $CACHE" | tee -a "$LOG"
  exit 1
fi

# Confirm done line or T>=700
if ! grep -q "\[cache\] done" "$MARKER" 2>/dev/null; then
  echo "[pipeline-v4] WARN: no done marker; checking zarr size" | tee -a "$LOG"
fi

echo "[pipeline-v4] cache ready $(date -Iseconds)" | tee -a "$LOG"
export MKL_INTERFACE_LAYER=GNU
source /local/Mthetho/miniforge3/etc/profile.d/conda.sh
conda activate /local/Mthetho/envs/lapai-anemoi
python - <<'PY' | tee -a "$LOG"
import zarr
from pathlib import Path
p = Path("data/processed/lapai/teacher_k1_cache_full2020_2021.zarr")
z = zarr.open(str(p), mode="r")
g = z["lapai_cache"] if "lapai_cache" in z else z
t = int(g["state_in"].shape[0])
print("cache T", t, "shape", g["state_in"].shape)
if t < 700:
    raise SystemExit(f"cache too small T={t}")
PY

bash scripts/run_trackB_stable_v4.sh 2>&1 | tee -a "$LOG"
bash scripts/run_trackB_held_out_v4_cassava.sh 2>&1 | tee -a "$LOG"

echo "[pipeline-v4] done $(date -Iseconds)" | tee -a "$LOG"

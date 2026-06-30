#!/bin/bash
# Validate the unified lustre env for Track A (training + inference parity).
set -uo pipefail

L="${LAPAI_PYTHON:-/home/msovara/lustre/dev/lapai-anemoi/bin/python}"

echo "=== lustre python ==="
ls -l "$L" 2>&1 || { echo "MISSING: $L"; exit 1; }

echo "=== anemoi stack versions ==="
"$L" - <<'PY'
import importlib.metadata as m
for p in ('anemoi-training','anemoi-models','anemoi-inference',
          'scikit-learn','torch','torch_geometric'):
    try:
        print(f"  {p:18s} {m.version(p)}")
    except Exception as e:
        print(f"  {p:18s} MISSING ({e})")
PY

echo "=== sklearn import + BallTree (glibc check) ==="
"$L" - <<'PY'
import sklearn
from sklearn.neighbors import BallTree
print("  sklearn import OK", sklearn.__version__)
PY

echo "=== anemoi.graphs import (training graph builder) ==="
"$L" -c 'from anemoi.graphs.nodes import AnemoiDatasetNodes; print("  anemoi.graphs OK")' 2>&1

echo "=== locate lapai-forecast repo ==="
for d in "$HOME/repos/lapai-forecast" "$HOME/lapai-forecast" \
         "$HOME/lustre/dev/lapai-forecast" "$HOME/lustre/lapai-forecast"; do
  if [ -d "$d" ]; then echo "  FOUND $d"; fi
done

echo "=== done ==="

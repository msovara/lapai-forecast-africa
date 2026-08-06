#!/usr/bin/env bash
set -euo pipefail
cd ~/repos/lapai-forecast
echo "=== qstat 7357693 ==="
qstat -f 7357693 2>&1 | grep -Ei 'job_state|exec_host|Resource_List.select|comment|resources_used|Exit_status' | head -25
echo "=== newest logs ==="
ls -lt lapai_trkA_ext*.o lapai_trackA_full.o 2>/dev/null | head -10 || true
ls -lt ~/*.o 2>/dev/null | head -10 || true
for L in lapai_trkA_ext_gpu2005.o lapai_trackA_full.o; do
  if [ -f "$L" ]; then
    echo "=== tail $L ==="
    tail -50 "$L"
  fi
done
# PBS often writes next to submit dir
find . -maxdepth 1 -name 'lapai_trkA_ext*.o' -o -name 'lapai_trackA_full.o' 2>/dev/null | head

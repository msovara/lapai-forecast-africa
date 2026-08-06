#!/usr/bin/env bash
# Poll Track A dry-run until finished, then print status + any output.
J="${1:-7353410}"
cd /home/msovara/repos/lapai-forecast || exit 1

for i in $(seq 1 36); do
  st=$(qstat -x -f "$J" 2>/dev/null | grep -E '^\s*job_state' | awk '{print $3}')
  echo "t=${i}0s state=${st:-?}"
  case "$st" in F|H) break ;; esac
  sleep 10
done

echo
echo "=== result ==="
qstat -x -f "$J" 2>&1 | sed -e ':a' -e '$!N' -e 's/\n\t//' -e 'ta' -e 'P' -e 'D' \
  | grep -Ei 'job_state|Exit_status|exec_vnode|comment|run_count|project|Variable_List' \
  | sed 's/^[[:space:]]*/  /' | cut -c1-220

echo
echo "=== output files ==="
for f in lapai_trackA_full.o lapai_trackA_full.e \
         "/home/msovara/lapai_trackA_full.o${J}" \
         "/home/msovara/repos/lapai-forecast/lapai_trackA_full.o${J}"; do
  if [ -f "$f" ]; then
    echo "--- $f (last 80 lines) ---"
    tail -80 "$f"
  fi
done
find /home/msovara -maxdepth 2 -name "lapai_trackA_full*" -newermt '-1 hour' 2>/dev/null

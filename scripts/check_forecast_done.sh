#!/usr/bin/env bash
for j in 7356037 7356038; do
  echo "=== $j ==="
  qstat -x -f "$j" 2>&1 | sed -e ':a' -e '$!N' -e 's/\n\t//' -e 'ta' -e 'P' -e 'D' \
    | grep -Ei 'Job_Name|job_state|Exit_status|exec_vnode|comment|resources_used.walltime|ctime|mtime' \
    | sed 's/^[[:space:]]*/  /' | cut -c1-200
  echo
done
echo "=== gpu2005 log tail ==="
tail -25 /home/msovara/lapai_fc_gpu2005.o7356037 | tr '\r' '\n' | tail -25
echo
echo "=== ckpt used ==="
readlink -f /home/msovara/repos/lapai-forecast/models/teacher_coarsened.ckpt
ls -la /home/msovara/repos/lapai-forecast/data/processed/trackA/forecasts/

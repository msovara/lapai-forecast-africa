#!/usr/bin/env bash
for j in 7356037 7356038; do
  echo "=== $j ==="
  qstat -x -f "$j" 2>&1 | sed -e ':a' -e '$!N' -e 's/\n\t//' -e 'ta' -e 'P' -e 'D' \
    | grep -Ei 'Job_Name|job_state|Exit_status|comment|exec_vnode|resources_used.walltime' \
    | sed 's/^[[:space:]]*/  /' | cut -c1-220
  echo
done
echo "=== tail gpu2005 log ==="
tail -25 /home/msovara/lapai_fc_gpu2005.o7356037 | tr '\r' '\n' | tail -25
echo
echo "=== ckpt used ==="
ls -la /home/msovara/repos/lapai-forecast/models/teacher_coarsened.ckpt
readlink -f /home/msovara/repos/lapai-forecast/models/teacher_coarsened.ckpt

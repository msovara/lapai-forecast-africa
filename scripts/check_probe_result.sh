#!/usr/bin/env bash
# What actually happened to the GPU probe job.
J="${1:-7353072}"

echo "=== qstat -x -f $J ==="
qstat -x -f "$J" 2>&1 | grep -Ei \
  'job_state|Exit_status|exec_host|exec_vnode|comment|run_count|Hold_Types|stime|mtime|obittime|Output_Path|resources_used' \
  | sed 's/^[[:space:]]*/  /' | cut -c1-220

echo
echo "=== search for output files anywhere obvious ==="
find /home/msovara -maxdepth 3 -name 'lapai_gpu_probe*' -newermt '-1 hour' 2>/dev/null
find /home/msovara -maxdepth 2 -name '*.o7353072' -o -maxdepth 2 -name '*7353072*' 2>/dev/null | head

echo
echo "=== PBS spool (if readable) ==="
ls -la /var/spool/pbs/spool/ 2>/dev/null | grep -i 7353072

echo
echo "=== full recent job history ==="
qstat -x -u msovara 2>&1 | tail -10

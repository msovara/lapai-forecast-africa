#!/usr/bin/env bash
# Diagnose GT A1b jobs (7358489 / 7358490) after one exits.
set -u
cd /home/msovara/repos/lapai-forecast || exit 1

echo "=== QUEUE ==="
qstat -u msovara 2>&1 || true
echo

echo "=== RECENT HISTORY ==="
qstat -x -u msovara 2>&1 | tail -30 || true
echo

for j in 7358489 7358490; do
  echo "=== JOB $j ==="
  qstat -x -f "$j" 2>&1 | sed -e ':a' -e '$!N' -e 's/\n\t//' -e 'ta' -e 'P' -e 'D' \
    | grep -Ei 'Job_Name|job_state|Exit_status|exec_host|comment|resources_used.walltime|resources_used.cpupercent|Output_Path|Error_Path|ctime|stime|obittime|Resource_List.select|run_count' \
    | sed 's/^[[:space:]]*/  /' | cut -c1-240
  echo
done

echo "=== PBS OUTPUT FILES (recent gtA1b / trackA) ==="
ls -lt /home/msovara/repos/lapai-forecast/lapai_gtA1b* \
       /home/msovara/repos/lapai-forecast/*.o7358489 \
       /home/msovara/repos/lapai-forecast/*.o7358490 \
       /home/msovara/repos/lapai-forecast/*.e7358489 \
       /home/msovara/repos/lapai-forecast/*.e7358490 \
       /home/msovara/*gtA1b* 2>/dev/null | head -40 || true
find /home/msovara -maxdepth 3 \( -name '*7358489*' -o -name '*7358490*' -o -name '*gtA1b*' \) 2>/dev/null | head -40
echo

echo "=== TAIL EXITED JOB LOGS ==="
for f in \
  /home/msovara/repos/lapai-forecast/lapai_gtA1b_gpu2005.o* \
  /home/msovara/repos/lapai-forecast/lapai_gtA1b_gpu2006.o* \
  /home/msovara/repos/lapai-forecast/lapai_gtA1b*.o7358489 \
  /home/msovara/repos/lapai-forecast/lapai_gtA1b*.o7358490 \
  /home/msovara/lapai_gtA1b*.o* \
  /home/msovara/repos/lapai-forecast/lapai_trackA*.o7358489 \
  /home/msovara/repos/lapai-forecast/lapai_trackA*.o7358490
do
  if [[ -f "$f" ]]; then
    echo "--- $f (tail -80) ---"
    tail -80 "$f"
    echo
  fi
done

echo "=== GT RUN DIRS / CKPTS ==="
ls -lt models/trackA_gt_coarsen_runs 2>/dev/null | head -20 || echo "no trackA_gt_coarsen_runs"
ls -lt models/teacher_gt* 2>/dev/null | head -10 || echo "no teacher_gt*"
find models -maxdepth 3 -name 'inference*.ckpt' -newer models/teacher_coarsened.ckpt 2>/dev/null | head -20 || true

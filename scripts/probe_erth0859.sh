#!/usr/bin/env bash
# Decisive test: same GPU probe, different project.
# RCHPC last ran a GPU job successfully on 30 June; ERTH0859 is actively working today.
# If this runs, RCHPC's allocation has lapsed. If it fails the same way, it is CHPC-side.
cd /home/msovara/repos/lapai-forecast || exit 1

echo "=== submitting probe under ERTH0859 ==="
JID=$(qsub -P ERTH0859 pbs/gpu_probe.pbs)
echo "job: $JID"
J="${JID%%.*}"
echo

for i in $(seq 1 20); do
  st=$(qstat -x -f "$J" 2>/dev/null | grep -E '^\s*job_state' | awk '{print $3}')
  echo "  t=${i}0s state=${st:-?}"
  [ "$st" = "F" ] && break
  sleep 10
done
echo

echo "=== result ==="
qstat -x -f "$J" 2>&1 | sed -e ':a' -e '$!N' -e 's/\n\t//' -e 'ta' -e 'P' -e 'D' \
  | grep -Ei 'job_state|Exit_status|exec_vnode|comment|run_count|project' \
  | sed 's/^[[:space:]]*/  /' | cut -c1-200
echo

echo "=== output file ==="
for f in "/home/msovara/repos/lapai-forecast/lapai_gpu_probe.o${J}" \
         "/home/msovara/lapai_gpu_probe.o${J}" \
         "/home/msovara/repos/lapai-forecast/lapai_gpu_probe.o"; do
  if [ -f "$f" ]; then
    echo "--- $f ---"
    cat "$f"
    exit 0
  fi
done
echo "  no output file found (job never produced a shell)"

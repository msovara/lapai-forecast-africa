#!/usr/bin/env bash
# Write GPU facts to a fixed path (PBS -k oe output was missing after reboot).
OUT=/home/msovara/repos/lapai-forecast/reports/gpu_probe_last.txt
{
  echo "=== $(date) host=$(hostname) job=${PBS_JOBID} ==="
  pbsnodes "$(hostname -s)" 2>/dev/null | grep -Ei 'node_type|ngpus|ncpus' | sed 's/^[[:space:]]*/  /'
  echo
  nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv 2>&1
  echo
  nvidia-smi 2>&1 | head -20
  echo
  echo PROBE_OK
} > "$OUT" 2>&1
cat "$OUT"

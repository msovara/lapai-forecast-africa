#!/usr/bin/env bash
# After reboot: drop stuck 32GB-pinned dry-run, probe GPU memory, report.
cd /home/msovara/repos/lapai-forecast || exit 1
CR=$(printf '\r')

# kill stuck dry-run
qdel -W force 7353410 2>/dev/null || qdel 7353410 2>/dev/null || true
sleep 2

# stage probe body
mkdir -p reports
cat > /tmp/gpu_probe_body.sh <<'EOF'
#!/usr/bin/env bash
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
EOF

# submit probe writing fixed output path via -o
JID=$(qsub -P ERTH0859 -N lapai_gpu_probe2 -q gpu_1 \
  -l select=1:ncpus=4:ngpus=1 -l walltime=00:05:00 \
  -j oe -k oe \
  -o /home/msovara/repos/lapai-forecast/reports/gpu_probe_pbs.o \
  -- /bin/bash /tmp/gpu_probe_body.sh)
echo "probe: $JID"
J="${JID%%.*}"

for i in $(seq 1 30); do
  st=$(qstat -x -f "$J" 2>/dev/null | grep -E '^\s*job_state' | awk '{print $3}')
  echo "t=${i}0s state=${st:-?}"
  [ "$st" = "F" ] && break
  sleep 10
done

echo "=== qstat ==="
qstat -x -f "$J" 2>&1 | sed -e ':a' -e '$!N' -e 's/\n\t//' -e 'ta' -e 'P' -e 'D' \
  | grep -Ei 'job_state|Exit_status|exec_vnode|comment|project' | sed 's/^[[:space:]]*/  /' | cut -c1-200

echo "=== fixed report ==="
cat reports/gpu_probe_last.txt 2>/dev/null || echo "(no gpu_probe_last.txt)"
echo "=== pbs -o ==="
cat reports/gpu_probe_pbs.o 2>/dev/null | tail -40 || echo "(no pbs o)"

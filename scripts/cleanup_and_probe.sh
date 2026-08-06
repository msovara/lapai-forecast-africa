#!/usr/bin/env bash
# 1. Force-clear the two system-held duplicate dry runs.
# 2. Submit the unpinned GPU probe to distinguish "32GB nodes are broken" from
#    "RCHPC allocation is the problem".
cd /home/msovara/repos/lapai-forecast || exit 1

CR=$(printf '\r')
sed -i "s/${CR}\$//" pbs/gpu_probe.pbs
echo "CR in pbs/gpu_probe.pbs: $(grep -c "${CR}" pbs/gpu_probe.pbs)"
echo

echo "=== force-deleting held duplicates ==="
for j in 7353070 7353071; do
  qdel -W force "$j" 2>&1 && echo "  $j : deleted" || echo "  $j : qdel returned $?"
done
sleep 5
echo

echo "=== queue after cleanup ==="
qstat -u msovara 2>&1
echo

echo "=== submitting unpinned GPU probe (5 min, ngpus=1, no node_type) ==="
qsub pbs/gpu_probe.pbs
echo "qsub exit: $?"
echo

sleep 10
echo "=== queue ==="
qstat -u msovara 2>&1

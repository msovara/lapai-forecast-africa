#!/usr/bin/env bash
# Check project allocation and group membership -- a depleted or wrong project makes
# PBS accept a job, then fail every run attempt until it system-holds it.

echo "=== groups (gpu_1 needs acl_groups=gpu) ==="
id
echo

echo "=== currently running job's project (this one works) ==="
qstat -f 7353059 2>/dev/null | grep -Ei 'project|Account_Name|queue|euser|egroup' | sed 's/^[[:space:]]*/  /'
echo

echo "=== held job's project ==="
qstat -f 7353070 2>/dev/null | grep -Ei 'project|Account_Name|queue|euser|egroup' | sed 's/^[[:space:]]*/  /'
echo

echo "=== allocation tools available ==="
for c in chpcallocation allocation mam-list-funds gbalance glsalloc sbank; do
  command -v "$c" >/dev/null 2>&1 && echo "  $c : $(command -v $c)" || echo "  $c : not found"
done
echo

echo "=== try allocation query ==="
if command -v chpcallocation >/dev/null 2>&1; then
  chpcallocation 2>&1 | head -20
fi
echo

echo "=== recent finished jobs (did anything run on gpu_1 lately?) ==="
qstat -x -u msovara 2>/dev/null | tail -15

#!/usr/bin/env bash
# Pull PBS's own history for the held Track A job to find why every run attempt failed.
J="${1:-7353070}"

echo "=== tracejob $J ==="
tracejob "$J" 2>&1 | grep -vi 'Log file.*not found' | tail -60

echo
echo "=== node 32GB detail (cpus/gpus actually free) ==="
for n in gpu2005 gpu2006; do
  echo "--- $n ---"
  pbsnodes "$n" 2>&1 | grep -Ei \
    'state|resources_available.ncpus|resources_available.ngpus|resources_assigned.ncpus|resources_assigned.ngpus|node_type|jobs =|comment' \
    | sed 's/^[[:space:]]*/  /' | cut -c1-200
done

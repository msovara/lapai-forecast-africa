#!/usr/bin/env bash
# Show the resource limits PBS enforces on the GPU queues, so trackA_full.pbs
# can be sized to something the scheduler will actually accept.
for q in gpu_1 gpu_2 gpu_3 gpu_4; do
  echo "=== $q ==="
  qstat -Qf "$q" 2>/dev/null | grep -Ei \
    'resources_max|resources_min|resources_default|max_run|max_queued|enabled|started|acl_group' \
    | sed 's/^[[:space:]]*/  /'
  echo
done

echo "=== node types available in gpu queues ==="
pbsnodes -a 2>/dev/null | grep -Ei 'resources_available.node_type|resources_available.ngpus' \
  | sort | uniq -c | sort -rn | head -20

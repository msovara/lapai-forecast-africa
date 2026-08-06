#!/usr/bin/env bash
# Compare the pinned 32GB nodes against the other GPU node types.
# Looking for (a) why the 32GB nodes sit idle, (b) whether another type is a safe
# substitute for a job tuned to a 32 GB V100.
for n in gpu2001 gpu2004 gpu2005 gpu2006 gpu4003; do
  echo "=== $n ==="
  pbsnodes "$n" 2>&1 | grep -Ei \
    'state|node_type|ncpus|ngpus|mem =|gpu_mem|comment|jobs|last_state_change|last_used|pcpus|resv' \
    | sed 's/^[[:space:]]*/  /' | cut -c1-160
  echo
done

echo "=== which GPU nodes have jobs on them right now ==="
pbsnodes -a 2>/dev/null | awk '
  /^[^ \t]/ { node=$1 }
  /resources_available.node_type/ { split($0,a,"="); t=a[2]; gsub(/ /,"",t) }
  /resources_available.ngpus/ { split($0,c,"="); g=c[2]+0 }
  /resources_assigned.ngpus/ { split($0,d,"="); ag=d[2]+0
                               if (g>0) printf "  %-10s type=%-8s gpus=%d assigned=%d\n", node, t, g, ag }
'

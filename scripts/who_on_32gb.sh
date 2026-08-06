#!/usr/bin/env bash
for n in gpu2005 gpu2006; do
  echo "=== $n ==="
  pbsnodes "$n" 2>&1 | grep -E 'state =|resources_assigned.ngpus|jobs =' | sed 's/^[[:space:]]*/  /' | cut -c1-220
done
echo
for j in $( (pbsnodes gpu2005; pbsnodes gpu2006) 2>/dev/null | grep -oE '[0-9]+\.sched01' | sort -u ); do
  echo "--- $j ---"
  qstat -f "$j" 2>/dev/null | sed -e ':a' -e '$!N' -e 's/\n\t//' -e 'ta' -e 'P' -e 'D' \
    | grep -Ei 'Job_Name|Job_Owner|job_state|Resource_List.walltime|resources_used.walltime|exec_vnode' \
    | sed 's/^[[:space:]]*/  /' | cut -c1-180
done

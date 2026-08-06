#!/usr/bin/env bash
# Show submit details for the queued Track A jobs so duplicates can be told apart.
for j in "$@"; do
  echo "=== $j ==="
  qstat -f "$j" 2>/dev/null | grep -Ei \
    'Job_Name|job_state|queue|ctime|qtime|Resource_List.walltime|Resource_List.ncpus|Variable_List' \
    | sed 's/^[[:space:]]*/  /' | cut -c1-400
  echo
done

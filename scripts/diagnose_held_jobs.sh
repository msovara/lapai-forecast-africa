#!/usr/bin/env bash
# Explain why the Track A jobs are sitting in H (held) state.
for j in "$@"; do
  echo "=== $j ==="
  qstat -f "$j" 2>&1 | tr -d '\n' | sed 's/    //g' | tr ',' '\n' \
    | grep -Ei 'job_state|Hold_Types|comment|substate|Error_Path|Output_Path|etime|stime' \
    | sed 's/^[[:space:]]*/  /'
  echo
done

echo "=== raw comment lines ==="
for j in "$@"; do
  echo "--- $j ---"
  qstat -f "$j" 2>&1 | grep -A3 -i 'comment'
done

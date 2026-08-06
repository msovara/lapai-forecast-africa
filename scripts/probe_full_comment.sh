#!/usr/bin/env bash
# PBS wraps long attribute values across lines; unwrap them to read the full comment.
for j in "$@"; do
  echo "=== $j : full comment ==="
  qstat -x -f "$j" 2>&1 \
    | sed -e ':a' -e '$!N' -e 's/\n\t//' -e 'ta' -e 'P' -e 'D' \
    | grep -Ei 'comment|Exit_status|exec_vnode|Variable_List' \
    | fold -w 200 \
    | sed 's/^/  /'
  echo
done

#!/usr/bin/env bash
for n in gpu2005 gpu2006; do
  echo "=== $n ==="
  pbsnodes "$n" 2>&1 | grep -E 'state =|node_type|resources_available.ngpus|resources_available.ncpus|jobs =|comment|last_used' | sed 's/^[[:space:]]*/  /' | cut -c1-180
done

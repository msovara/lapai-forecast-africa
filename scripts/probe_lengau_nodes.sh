#!/bin/bash
# Summarise Lengau PBS node availability (run on login node).
set -euo pipefail
TMP=$(mktemp)
pbsnodes -a >"$TMP" 2>/dev/null || { echo "pbsnodes failed"; exit 1; }

echo "=== Node state counts ==="
grep 'state =' "$TMP" | awk '{print $3}' | sort | uniq -c | sort -rn

echo
echo "=== Nodetype counts (all nodes in pbsnodes) ==="
grep 'resources_available.nodetype' "$TMP" | awk '{print $3}' | sort | uniq -c | sort -rn

echo
echo "=== chpclic / interactive CLI hosts ==="
grep -iE 'chpclic|lic[0-9]' "$TMP" || echo "(no hostname/nodetype match for chpclic)"

echo
echo "=== Free nodes (first 25) ==="
awk '/^[a-z]/ {n=$1; st=""} /^[[:space:]]+state = free/ {st="free"} /^[[:space:]]+resources_available.nodetype =/ {if(st=="free") print n, $3}' "$TMP" | head -25

echo
echo "=== Queues (enabled, with jobs) ==="
qstat -Q 2>/dev/null | awk 'NR==1 || ($4=="yes" && $7+$8>0) || ($4=="yes" && $3 ~ /normal|serial|gpu|bigmem|smp/)' | head -30

rm -f "$TMP"

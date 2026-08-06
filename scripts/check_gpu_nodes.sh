#!/usr/bin/env bash
# Report the state of every GPU node, grouped by node_type, so we can tell whether
# a job pinned to node_type=32GB has any healthy node to land on.
echo "=== GPU nodes by type and state ==="
pbsnodes -a 2>/dev/null | awk '
  /^[^ \t]/          { node=$1 }
  /resources_available.node_type/ { split($0,a,"="); type=a[2]; gsub(/ /,"",type) }
  /^[ \t]*state =/   { split($0,b,"="); st=b[2]; gsub(/ /,"",st) }
  /resources_available.ngpus/ { split($0,c,"="); ng=c[2]; gsub(/ /,"",ng)
                                if (ng+0 > 0) printf "%-22s %-10s gpus=%-3s %s\n", node, type, ng, st }
' | sort -k2,2 -k4,4

echo
echo "=== summary: node_type=32GB ==="
pbsnodes -a 2>/dev/null | awk '
  /^[^ \t]/ { node=$1; st=""; type="" }
  /^[ \t]*state =/ { split($0,b,"="); st=b[2]; gsub(/ /,"",st) }
  /resources_available.node_type/ { split($0,a,"="); type=a[2]; gsub(/ /,"",type)
                                    if (type=="32GB") print "  " node " state=" st }
'

echo
echo "=== all distinct GPU node types + states ==="
pbsnodes -a 2>/dev/null | grep -E 'resources_available.node_type|^[ \t]*state =' \
  | paste - - 2>/dev/null | sort | uniq -c | sort -rn | head -25

echo
echo "=== job output files if any ==="
ls -la /home/msovara/repos/lapai-forecast/lapai_trackA_full.* 2>&1
echo
echo "--- tail of .o / .e ---"
for f in /home/msovara/repos/lapai-forecast/lapai_trackA_full.o /home/msovara/repos/lapai-forecast/lapai_trackA_full.e; do
  if [ -f "$f" ]; then
    echo "--- $f ---"
    tail -40 "$f"
  fi
done

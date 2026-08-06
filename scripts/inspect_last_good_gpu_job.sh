#!/usr/bin/env bash
# Find artefacts from the last GPU job that actually worked (June smoke, job 7318569)
# so we can diff its submission environment against today's failing ones.
cd /home/msovara/repos/lapai-forecast || exit 1

echo "=== job output files in repo ==="
ls -la lapai_trackA*.o* lapai_trackA*.e* *.o7* 2>/dev/null | head -20
echo

echo "=== head of smoke job output (shows node + env that worked) ==="
for f in lapai_trackA.o lapai_trackA.o7318569 lapai_trackA.e; do
  if [ -f "$f" ]; then
    echo "--- $f ---"
    head -30 "$f"
    echo
  fi
done

echo "=== any PBS output files anywhere in home from June ==="
find /home/msovara -maxdepth 3 -name 'lapai_track*' -type f 2>/dev/null | head -10
echo

echo "=== smoke run dir (proves it trained) ==="
ls -la models/trackA_coarsen_runs/ 2>/dev/null
echo

echo "=== does history still have 7318569? ==="
qstat -x -f 7318569 2>&1 | sed -e ':a' -e '$!N' -e 's/\n\t//' -e 'ta' -e 'P' -e 'D' \
  | grep -Ei 'project|egroup|euser|exec_vnode|Exit_status|comment|Resource_List.select|group_list' \
  | sed 's/^[[:space:]]*/  /' | cut -c1-200

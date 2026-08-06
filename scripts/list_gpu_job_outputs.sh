#!/usr/bin/env bash
# List PBS job output files in $HOME (where -k oe puts them) newest last,
# and show the tail of the run the poster credits as the successful smoke (7318569).
cd /home/msovara || exit 1

echo "=== all lapai job outputs, oldest -> newest ==="
ls -lat --time-style=long-iso lapai_* 2>/dev/null | tail -30
echo

echo "=== today's files (if any) ==="
find /home/msovara -maxdepth 1 -name 'lapai_*' -newermt '-6 hours' 2>/dev/null
echo "(nothing above = no output produced today)"
echo

echo "=== successful smoke 7318569 ==="
if [ -f lapai_trackA.o7318569 ]; then
  echo "--- first 25 lines ---"
  head -25 lapai_trackA.o7318569
  echo
  echo "--- last 25 lines ---"
  tail -25 lapai_trackA.o7318569
else
  echo "  lapai_trackA.o7318569 not found; newest available instead:"
  newest=$(ls -t lapai_trackA.o* 2>/dev/null | head -1)
  echo "  --- $newest ---"
  head -25 "$newest" 2>/dev/null
fi

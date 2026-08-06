#!/usr/bin/env bash
# Upload already done via scp from laptop; this runs on Lengau.
cd /home/msovara/repos/lapai-forecast || exit 1
CR=$(printf '\r')
sed -i "s/${CR}\$//" pbs/gpu_probe.pbs pbs/trackA_full.pbs
echo "select line: $(grep -E '^#PBS -l select' pbs/trackA_full.pbs)"

echo "=== GPU probe ==="
JID=$(qsub -P ERTH0859 pbs/gpu_probe.pbs)
echo "probe: $JID"
J="${JID%%.*}"
for i in $(seq 1 24); do
  st=$(qstat -x -f "$J" 2>/dev/null | grep -E '^\s*job_state' | awk '{print $3}')
  echo "t=${i}0s state=${st:-?}"
  [ "$st" = "F" ] && break
  sleep 10
done
qstat -x -f "$J" 2>&1 | sed -e ':a' -e '$!N' -e 's/\n\t//' -e 'ta' -e 'P' -e 'D' \
  | grep -Ei 'job_state|Exit_status|exec_vnode|comment|project' | sed 's/^[[:space:]]*/  /' | cut -c1-200
echo "--- reports/gpu_probe_last.txt ---"
cat reports/gpu_probe_last.txt 2>/dev/null || echo MISSING

echo
echo "=== Track A dry-run (unpinned) ==="
JID2=$(qsub -v LAPAI_TRACKA_DRY=1 pbs/trackA_full.pbs)
echo "dry: $JID2"
J2="${JID2%%.*}"
for i in $(seq 1 36); do
  st=$(qstat -x -f "$J2" 2>/dev/null | grep -E '^\s*job_state' | awk '{print $3}')
  echo "t=${i}0s state=${st:-?}"
  case "$st" in F|H) break ;; esac
  sleep 10
done
qstat -x -f "$J2" 2>&1 | sed -e ':a' -e '$!N' -e 's/\n\t//' -e 'ta' -e 'P' -e 'D' \
  | grep -Ei 'job_state|Exit_status|exec_vnode|comment|project|Variable_List' | sed 's/^[[:space:]]*/  /' | cut -c1-220

echo "--- dry-run output ---"
for f in lapai_trackA_full.o "/home/msovara/lapai_trackA_full.o${J2}" \
         "/home/msovara/repos/lapai-forecast/lapai_trackA_full.o${J2}"; do
  [ -f "$f" ] && echo "== $f ==" && tail -100 "$f"
done
find /home/msovara -maxdepth 2 -name 'lapai_trackA_full*' -newermt '-30 min' 2>/dev/null

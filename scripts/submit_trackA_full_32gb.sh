#!/usr/bin/env bash
cd /home/msovara/repos/lapai-forecast || exit 1
CR=$(printf '\r')
sed -i "s/${CR}\$//" pbs/trackA_full.pbs

echo "=== 32GB GPU occupancy ==="
for n in gpu2005 gpu2006; do
  echo "--- $n ---"
  pbsnodes "$n" 2>&1 | grep -E 'state =|resources_assigned.ngpus|resources_assigned.ncpus|resources_available.ngpus|jobs =' | sed 's/^[[:space:]]*/  /' | cut -c1-200
done

# Prefer whichever has a free GPU; default gpu2005
for n in gpu2005 gpu2006; do
  ag=$(pbsnodes "$n" 2>&1 | grep 'resources_assigned.ngpus' | awk '{print $3}')
  av=$(pbsnodes "$n" 2>&1 | grep 'resources_available.ngpus' | awk '{print $3}')
  echo "$n assigned_gpus=${ag} available=${av}"
  if [ "${ag:-99}" -lt "${av:-0}" ]; then
    TARGET=$n
    break
  fi
done
TARGET=${TARGET:-gpu2005}
echo "TARGET=$TARGET"
sed -i "s/host=gpu200[56]/host=${TARGET}/" pbs/trackA_full.pbs
grep -E '^#PBS -l select' pbs/trackA_full.pbs

echo "=== submit full train (not dry) ==="
JID=$(qsub pbs/trackA_full.pbs)
echo "job: $JID"
sleep 5
qstat -u msovara
qstat -f "${JID%%.*}" 2>&1 | sed -e ':a' -e '$!N' -e 's/\n\t//' -e 'ta' -e 'P' -e 'D' \
  | grep -Ei 'job_state|comment|Resource_List.select|project|queue' | sed 's/^[[:space:]]*/  /' | cut -c1-200

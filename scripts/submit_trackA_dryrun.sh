#!/usr/bin/env bash
# Normalise the re-uploaded PBS script and submit the Track A dry run.
cd /home/msovara/repos/lapai-forecast || exit 1

CR=$(printf '\r')
sed -i "s/${CR}\$//" pbs/trackA_full.pbs
echo "CR remaining in pbs/trackA_full.pbs: $(grep -c "${CR}" pbs/trackA_full.pbs)"
echo "walltime line: $(grep -i 'walltime' pbs/trackA_full.pbs | grep -v '^#.*enforces')"
echo

echo "=== submitting dry run ==="
qsub -v LAPAI_TRACKA_DRY=1 pbs/trackA_full.pbs
echo "qsub exit: $?"
echo

qstat -u msovara

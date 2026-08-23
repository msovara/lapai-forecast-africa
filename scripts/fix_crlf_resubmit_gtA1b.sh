#!/usr/bin/env bash
set -euo pipefail
cd /home/msovara/repos/lapai-forecast || exit 1
perl -pi -e 's/\r\n/\n/g; s/\r/\n/g' pbs/inc_conda_lengau.sh scripts/submit_trackA_gt_coarsen_32gb.sh pbs/trackA_full.pbs
python -c '
from pathlib import Path
for p in ["pbs/inc_conda_lengau.sh", "scripts/submit_trackA_gt_coarsen_32gb.sh", "pbs/trackA_full.pbs"]:
    b = Path(p).read_bytes()
    print("%s CR_count=%d lines=%d" % (p, b.count(13), b.count(10)))
'
echo "=== qdel 7358691 ==="
qdel 7358691 2>&1 || true
sleep 2
bash scripts/submit_trackA_gt_coarsen_32gb.sh
echo "=== queue ==="
qstat -u msovara

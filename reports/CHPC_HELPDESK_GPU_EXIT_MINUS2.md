# CHPC helpdesk ticket — GPU jobs fail with Exit_status = -2 (no stdout)

**Suggested subject:** Lengau `gpu_1` jobs fail immediately with Exit_status=-2 on gpu2004 (no job output)

**To:** helpdesk@chpc.ac.za (or usual CHPC support channel)

**From / username:** msovara  
**Cluster:** Lengau  
**Date observed:** Sunday 2 August 2026  

---

## Summary

GPU jobs submitted to queue `gpu_1` are accepted by PBS, scheduled onto a GPU node, then fail immediately with **Exit_status = -2** and produce **no stdout/stderr**. The job script never appears to start (no shell output). This happens for both project codes **RCHPC** and **ERTH0859**, so it does not look like a depleted allocation on one project alone.

A minimal probe job (only `hostname` + `nvidia-smi`) fails the same way. Training scripts and datasets are therefore unlikely to be the cause.

## Example jobs

| Job ID | Project | Queue | Select | Node | Exit | Comment |
|--------|---------|-------|--------|------|------|---------|
| **7353072.sched01** | RCHPC | gpu_1 | `1:ncpus=4:ngpus=1` (no node_type) | **gpu2004** | **-2** | Job run … on (gpu2004:ncpus=4:ngpus=1) and failed |
| **7353073.sched01** | ERTH0859 | gpu_1 | same probe script | **gpu2004** | **-2** | Job run … on (gpu2004:ncpus=4:ngpus=1) and failed |
| 7353070.sched01 | RCHPC | gpu_1 | `1:ncpus=9:ngpus=1:node_type=32GB` | (held) | -3 | job held, too many failed attempts to run |
| 7353071.sched01 | RCHPC | gpu_1 | same as 7353070 | (held) | -3 | job held, too many failed attempts to run |

Notes on 7353070 / 7353071:
- PBS tried to start them many times (`run_count = 21`) then system-held them.
- They were pinned to `node_type=32GB` (gpu2005 / gpu2006).
- Those nodes report `state=free` but show `last_used_time` around **23 July 2026** (no successful jobs for ~10 days).
- Neighbouring GPU nodes **gpu2001** and **gpu2004** show PBS comment **`CUDA Update`**.

## Minimal probe used (7353072 / 7353073)

PBS directives (essentially):

```bash
#PBS -N lapai_gpu_probe
#PBS -P RCHPC          # or -P ERTH0859 for 7353073
#PBS -q gpu_1
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=00:05:00
#PBS -j oe
#PBS -k oe
```

Job body only prints hostname / node info and runs `nvidia-smi`.  
No conda activate, no training, no large I/O.

Expected output path if the script had started (with `-k oe`):  
`/home/msovara/lapai_gpu_probe.o<jobid>` or under  
`/home/msovara/repos/lapai-forecast/`.  
**No probe output files were created today.**

## What still works

- Login SSH to Lengau is fine.
- CPU/normal queue jobs still run (e.g. currently running job **7353059** under project **ERTH0859** on queue `normal`).
- The same style of `gpu_1` job **did succeed previously**:
  - Job **7318569** (Track A smoke train), **30 June 2026**
  - Host: **gpu2005**
  - Project: **RCHPC**
  - Produced full training logs under `/home/msovara/lapai_trackA.o7318569`

So the account/group membership for `gpu` appears historically valid; something has changed on the GPU nodes since late June / late July.

## Request

Please could you check:

1. Why jobs on **gpu2004** (and possibly the `node_type=32GB` nodes gpu2005/gpu2006) fail at MOM start with **Exit_status = -2**.
2. Whether the **CUDA Update** maintenance on those GPU nodes is incomplete / nodes are not ready for user jobs.
3. When `gpu_1` is expected to be usable again, or which GPU queue/node types we should use in the meantime.

Happy to provide the full probe script (`~/repos/lapai-forecast/pbs/gpu_probe.pbs`) or re-submit a diagnostic job if useful.

Thank you,  
Mthetho Sovara (`msovara`)

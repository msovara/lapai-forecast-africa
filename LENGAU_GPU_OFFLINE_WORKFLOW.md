# Lengau GPU training — offline prep, GPU-only cluster runs

**Context:** GPU nodes are now available without long queue waits, but **compute nodes have no internet**. All installs, downloads, and validation happen **offline** (DTN, login node prep, WSL/Linux pack, or lustre staging). Only **GPU- and memory-bound** work runs in PBS on `gpu_1`.

**Project path:** `/home/msovara/lustre/anemoi-weather-quest`  
**Dataset:** `data/jra3q_processed/jra3q_anemoi_format.zarr`  
**Working GPU smoke config:** job `7317611` (256 ch, 8 layers, scale `[1]`, batch 1, step ckpts every 5)

---

## Why checkpoints failed before

| Issue | Cause | Fix |
|-------|--------|-----|
| No `.ckpt` on CPU bigmem | 48 h walltime; ~86 s/val batch; epoch never finished | Move to **GPU**; save **every N steps**, not only end-of-epoch |
| GPU OOM (65 M, scale `[1,2,4]`) | Full model + activations > 32 GB V100 | Use **v4 smoke sizing** first; scale up only if memory allows |
| `cpu_offload=True` | Not implemented in Anemoi GNN mapper | Do not use |
| `num_workers=0` + prefetch | Lightning constraint | Keep `num_workers=2` |
| Script corruption (`tr -d "\r"`) | PowerShell SSH | Upload + `python3 scripts/fix_crlf_on_cluster.py` only |

---

## Split: offline vs cluster GPU

```
OFFLINE (DTN / login / WSL / lustre prep)
├── Build & conda-pack env (once) → upload to ~/conda-packs/
├── JRA-3Q Zarr on lustre (already done)
├── Upload PBS scripts + anemoi_train_common.sh (fix CRLF)
├── Config probe: python scripts/probe_zarr_read.py
├── Dry-run imports on login (no train): conda activate → import anemoi
└── Optional: graph/build cache if Anemoi supports pre-build on login

GPU PBS ONLY (gpu_1, memory-bound)
├── module load cuda + activate packed env
├── python -m anemoi.training train ...  (forward/backward + ckpt write)
└── logs → logs/gpu_train_<JOBID>.log
```

**Do not on GPU nodes:** `pip install`, `conda create`, CDS/ERA5 download, git clone, heavy Zarr preprocessing.

---

## Offline checklist (before every qsub)

```bash
# On Lengau login or DTN
cd /home/msovara/lustre/anemoi-weather-quest
python3 scripts/fix_crlf_on_cluster.py

# Env (pick one — do NOT use source .../bin/activate; use conda or direct python)
module load chpc/python/anaconda/3-2024.10.1
source /home/apps/chpc/bio/anaconda3-2024.10.1/etc/profile.d/conda.sh
conda activate /apps/chpc/chem/anaconda3-2021.11/envs/anemoi-training
# OR direct (works on login without conda init):
# /apps/chpc/chem/anaconda3-2021.11/envs/anemoi-training/bin/python scripts/probe_zarr_read.py ...

python scripts/probe_zarr_read.py data/jra3q_processed/jra3q_anemoi_format.zarr
/apps/chpc/chem/anaconda3-2021.11/envs/anemoi-training/bin/python -c "import anemoi.training, torch; print('cuda', torch.cuda.is_available())"

grep -E 'tain_|msovaa' train_jra3q_anemoi*.pbs anemoi_train_common.sh && echo BAD || echo scripts OK
```

Upload from Windows (PowerShell):

```powershell
scp lapai-forecast/train_jra3q_anemoi_gpu_smoke.pbs msovara@lengau.chpc.ac.za:/home/msovara/lustre/anemoi-weather-quest/
scp lapai-forecast/anemoi_train_common.sh msovara@lengau.chpc.ac.za:/home/msovara/lustre/anemoi-weather-quest/
ssh msovara@lengau.chpc.ac.za "cd /home/msovara/lustre/anemoi-weather-quest && python3 scripts/fix_crlf_on_cluster.py"
```

---

## Recommended training ladder (GPU only)

### Stage A — Smoke (proven, ~15 min)

```bash
qsub train_jra3q_anemoi_gpu_smoke.pbs
tail -f logs/gpu_smoke_<JOBID>.log
find models/jra3q_smoke_<JOBID> -name '*.ckpt'
```

Settings: `model.num_channels=256`, `processor.num_layers=8`, `scale_resolutions=[1]`, `batch_size=1`, `max_steps=50`, ckpt every **5 steps**.

### Stage B — Short GPU run with step checkpoints

Promote smoke settings; increase `max_steps` / `max_epochs` gradually:

```bash
diagnostics.checkpoint.every_n_train_steps.save_frequency=5
diagnostics.checkpoint.every_n_train_steps.num_models_saved=3
diagnostics.checkpoint.every_n_minutes.save_frequency=10
training.max_steps=500   # or max_epochs with small epoch if faster on GPU
```

Output: `models/jra3q_gpu_<JOBID>/checkpoint/.../last.ckpt`

### Stage C — Scale model (only if Stage B stable)

Increase one knob at a time; watch `nvidia-smi` in log:

1. `scale_resolutions=[1,2]`  
2. `model.num_channels=384`  
3. Full `scale_resolutions=[1,2,4]` + 512 ch (likely needs **gpu_2** or 80 GB node if available)

**Do not** jump straight to 65 M + `[1,2,4]` on 32 GB V100.

---

## PBS template principles (GPU job)

| Item | Value |
|------|--------|
| Queue | `gpu_1` (or `gpu_2` if multi-GPU) |
| Walltime | Start **2 h** for short runs; extend after timing one epoch |
| Log | `exec > >(tee -a logs/gpu_train_${JOB_NUM}.log) 2>&1` |
| Ckpt dir | `system.output.root=${WORK_DIR}/models/jra3q_gpu_${JOB_NUM}` on **lustre** |
| Seed | `ANEMOI_BASE_SEED=${JOB_NUM}` |
| CUDA | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Plots | `+diagnostics.plot.callbacks=[]` (avoid ERA5 plot errors) |

---

## Env strategy (offline)

Same pattern as `aifs-africa/docs/LINUX_BUILD_FOR_LENGAU.md`:

1. Build Linux env on **WSL or Linux VM** with network  
2. `conda-pack` → upload tarball once to Lengau  
3. Unpack on login: `~/conda-envs/lapai-anemoi`  
4. GPU PBS only `source ~/conda-envs/lapai-anemoi/bin/activate`

CHPC module env (`/apps/chpc/chem/.../anemoi-training`) is fine if already complete — no install on compute node.

---

## Resume from checkpoint (offline path + GPU job)

1. Identify last good ckpt on lustre (smoke job `7317611` path).  
2. Offline: verify ckpt file size and path.  
3. GPU PBS: add Hydra resume override (Anemoi/Lightning):

```bash
# Example — confirm exact key in your Anemoi 0.8.4 config
# checkpoint_path=.../last.ckpt
# or trainer.fit ckpt_path=... per anemoi.training CLI
```

Check `python -m anemoi.training train --help` on login for `ckpt_path` / resume flag for 0.8.4.

---

## What to report in NEOSS / WeEarth docs

- **Anemoi JRA-3Q:** GPU training infrastructure validated (step checkpoints written); full 65 M training is a **scale-up** step, not yet complete.  
- **Cluster use:** Offline prep + GPU-only training aligns with CHPC policy and new fast GPU access.  
- **Separate track** from ConvLSTM fire maps (Patience) — do not conflate in publications.

---

## Quick commands

```bash
ssh msovara@lengau.chpc.ac.za
qstat -u msovara
qsub /home/msovara/lustre/anemoi-weather-quest/train_jra3q_anemoi_gpu_smoke.pbs
tail -f /home/msovara/lustre/anemoi-weather-quest/logs/gpu_smoke_*.log
find /home/msovara/lustre/anemoi-weather-quest/models -name '*.ckpt' -mtime -7
```

---

*Last updated: June 2026*

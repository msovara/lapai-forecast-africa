# Lengau (CHPC) — LapAI-Forecast ops

This repo is wired for **Centre for High Performance Computing** [Lengau](https://www.chpc.ac.za/).

**First-time teacher inference on GPU:** after **`conda activate lapai-anemoi`**, use the *Minimal copy-paste path* in [`reports/RUNBOOK_BASELINE_LENGAU.md`](../reports/RUNBOOK_BASELINE_LENGAU.md).

## Submit PBS from the LapAI repo root

`PBS_O_WORKDIR` becomes the clone directory (`lapai-forecast/`).

```bash
cd ~/repos/lapai-forecast   # wherever you cloned github.com/msovara/lapai-forecast-africa
qsub pbs/trackA.pbs
```

Submit **from inside** `lapai-forecast/`, not from a parent repo unless you deliberately set PBS `-d`.

## Conda stacks

| Slot | Default env | Create from | Purpose |
|------|-------------|-------------|---------|
| **Track A** | **`lapai-anemoi`** (isolated) | **`scripts/setup_lengau_envs.sh`** defaults to [`environment-anemoi-nogrib.yml`](../environment-anemoi-nogrib.yml) **(CHPC‑safe)** | PyTorch + CUDA + full **Anemoi** pip stack; optional conda [`environment-anemoi.yml`](../environment-anemoi.yml) adds **eccodes/cfgrib** when `LAPAI_ENV_ANEMOI_YML=environment-anemoi.yml` and glibc is new enough (~2.28+) |
| **Track B / LoRA** | **`lapai-credit`** | [`environment-credit-lengau.yml`](../environment-credit-lengau.yml) | Student / LoRA / ONNX helpers (`pbs/student.pbs`, `pbs/lora.pbs`) |

**First-time environment creation belongs on compute node `chpclic1`**, not `login*`:
login-node conda solves often get **OOM-killed**; some batch queues have **no outbound HTTPS**, so conda/pip stalls. Mirror the allocation pattern used elsewhere on CHPC (see also project notes on **`chpclic1`**).

Interactive (recommended first run — project code **RCHPC**):

```bash
ssh msovara@lengau.chpc.ac.za   # or your login node path
qsub -I -P RCHPC -q normal -l select=1:ncpus=4:mem=48GB -l walltime=8:00:00 \
  -l place=scatter:excl -W x=FLAGS:ADVRES:chpclic1
hostname   # expect chpclic1
module purge
module load chpc/python/anaconda/3-2024.10.1
source /home/apps/chpc/bio/anaconda3-2024.10.1/etc/profile.d/conda.sh
cd ~/repos/lapai-forecast                        # wherever you cloned lapai-forecast-africa
bash scripts/setup_lengau_envs.sh
```

Batch option (still pinned to **`chpclic1`**) from repo root [`pbs/setup_env_chpclic1.pbs`](../pbs/setup_env_chpclic1.pbs): edit **`#PBS -P`** if needed, then **`qsub pbs/setup_env_chpclic1.pbs`**.

Legacy login-only attempt (often fails on RAM) or smp batch [`pbs/setup_env_lengau.pbs`](../pbs/setup_env_lengau.pbs):

```bash
module load chpc/python/anaconda/3-2024.10.1
source /home/apps/chpc/bio/anaconda3-2024.10.1/etc/profile.d/conda.sh
cd /path/to/lapai-forecast
bash scripts/setup_lengau_envs.sh
```

That creates/updates **`lapai-anemoi`** and **`lapai-credit`**, and runs **`pip install -e . --no-deps`** in **each** so `lapai_inference` resolves in Track A / Track B jobs.

Optional: heavy env prefixes on lustre (`export CONDA_ENVS_PATH=...`; see README).

**If conda fails on `eccodes` / `jasper` / `__glibc`:** some CHPC nodes ship **glibc &lt; 2.28** but recent conda-forge GRIB binaries target newer glibc.

1. Check: `ldd --version`.
2. **`bash scripts/setup_lengau_envs.sh`** uses **`environment-anemoi-nogrib.yml`** by default (already avoids conda **`eccodes`/`cfgrib`**).  
   If you overrode **`LAPAI_ENV_ANEMOI_YML=environment-anemoi.yml`**, unset it or reinstall after **`conda env remove -n lapai-anemoi --yes`**.
3. On a workstation with **`glibc` ≥ ~2.28**, optional full GRIB in conda:  
   `LAPAI_ENV_ANEMOI_YML=environment-anemoi.yml bash scripts/setup_lengau_envs.sh`

## Teacher inference (AIFS checkpoint on disk)

With **`lapai-anemoi`** active and **`anemoi-inference`** on `PATH`:

```bash
bash scripts/lapai_inference_gate.sh
# After editing LAPAI_INFER_TEMPLATE / YAML:
# python scripts/run_aifs_inference.py --dry-run
python scripts/run_aifs_inference.py   # inference on this node (typically GPU / interactive GPU)
qsub pbs/inference_aifs_teacher.pbs
# Custom template on the GPU job (must be a repo-relative path):
# qsub -v LAPAI_INFER_TEMPLATE=configs/inference_aifs_netcdf_example.yaml pbs/inference_aifs_teacher.pbs
# Dry-run on the batch node:
# qsub -v LAPAI_AIFS_INFER_DRY=1 pbs/inference_aifs_teacher.pbs
```

Starter YAML: [`configs/inference_aifs_minimal.yaml`](../configs/inference_aifs_minimal.yaml) (bundled **`dataset: test`**, **`printer`**). Persisting forecasts: start from [`configs/inference_aifs_netcdf_example.yaml`](../configs/inference_aifs_netcdf_example.yaml).

**If the gate or inference fails:** see [`reports/RUNBOOK_BASELINE_LENGAU.md`](../reports/RUNBOOK_BASELINE_LENGAU.md) section **7** (troubleshooting table).

### Site-wide fallback (no isolation)

If you prefer the CHPC **chem** env instead of **lapai‑anemoi**:

```bash
qsub -v LAPAI_CONDA_TRACK_A=/apps/chpc/chem/anaconda3-2021.11/envs/anemoi-training pbs/trackA.pbs
```

## Override conda targets

```bash
qsub -v LAPAI_CONDA_TRACK_A=my-anemoi-env pbs/trackA.pbs
qsub -v LAPAI_CONDA_TRACK_B=my-torch-env pbs/student.pbs
```

Source: [`pbs/inc_conda_lengau.sh`](../pbs/inc_conda_lengau.sh).

## GPU jobs

Batch scripts call `lapai_load_cuda_gpu` which tries:

`module load chpc/cuda/12.0/12.0`, then `chpc/cuda/11.8/11.8`.

## Create `lapai-credit` only (Track B)

If **`lapai-anemoi`** already exists:

```bash
conda env create -f environment-credit-lengau.yml
conda activate lapai-credit
pip install -e . --no-deps
```

## Troubleshooting conda updates

If **`lapai-anemoi`** was created from an older `environment-anemoi.yml` (conda-forge‑only sketch) and `conda env update` conflicts on PyTorch/NVIDIA channels, remove and recreate:

```bash
conda env remove -n lapai-anemoi --yes
conda env create -f environment-anemoi.yml
conda activate lapai-anemoi
pip install -e . --no-deps
```

## Sanity: personal env with peft + torch

[`scripts/check_peft_envs.sh`](../scripts/check_peft_envs.sh)

```bash
bash scripts/check_peft_envs.sh
```

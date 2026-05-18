# Lengau (CHPC) — LapAI-Forecast ops

This repo is wired for **Centre for High Performance Computing** [Lengau](https://www.chpc.ac.za/).

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
| **Track A** | **`lapai-anemoi`** (isolated) | [`environment-anemoi.yml`](../environment-anemoi.yml) | PyTorch + CUDA + full **Anemoi** pip stack + GRIB/xarray (`pbs/trackA.pbs`) |
| **Track B / LoRA** | **`lapai-credit`** | [`environment-credit-lengau.yml`](../environment-credit-lengau.yml) | Student / LoRA / ONNX helpers (`pbs/student.pbs`, `pbs/lora.pbs`) |

Recommended one-shot install on Lengau (interactive or batch [`pbs/setup_env_lengau.pbs`](../pbs/setup_env_lengau.pbs)):

```bash
module load chpc/python/anaconda/3-2024.10.1
source /home/apps/chpc/bio/anaconda3-2024.10.1/etc/profile.d/conda.sh
cd /path/to/lapai-forecast
bash scripts/setup_lengau_envs.sh
```

That creates/updates **`lapai-anemoi`** and **`lapai-credit`**, and runs **`pip install -e . --no-deps`** in **each** so `lapai_inference` resolves in Track A / Track B jobs.

Optional: heavy env prefixes on lustre (`export CONDA_ENVS_PATH=...`; see README).

## Teacher inference (AIFS checkpoint on disk)

With **`lapai-anemoi`** active and **`anemoi-inference`** on `PATH`:

```bash
python scripts/verify_inference_stack.py --strict
python scripts/run_aifs_inference.py --dry-run
python scripts/run_aifs_inference.py
qsub pbs/inference_aifs_teacher.pbs
# Dry-run on the batch node:
# qsub -v LAPAI_AIFS_INFER_DRY=1 pbs/inference_aifs_teacher.pbs
```

Starter YAML: [`configs/inference_aifs_minimal.yaml`](../configs/inference_aifs_minimal.yaml) (bundled **`dataset: test`**, **`printer`**). Persisting forecasts: start from [`configs/inference_aifs_netcdf_example.yaml`](../configs/inference_aifs_netcdf_example.yaml).

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

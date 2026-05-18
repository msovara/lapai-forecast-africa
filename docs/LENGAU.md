# Lengau (CHPC) — LapAI-Forecast ops

This repo is wired for **Centre for High Performance Computing** [Lengau](https://www.chpc.ac.za/).

## Submit PBS from the LapAI repo root

`PBS_O_WORKDIR` becomes the clone directory (`lapai-forecast/`).

```bash
cd ~/repos/lapai-forecast   # wherever you cloned github.com/msovara/lapai-forecast-africa
qsub pbs/trackA.pbs
```

Submit **from inside** `lapai-forecast/`, not from a parent repo unless you deliberately set PBS `-d`.

## Conda stacks (verified on login2)

| Slot | Env | Purpose |
|------|-----|---------|
| **Track A** (default) | `/apps/chpc/chem/anaconda3-2021.11/envs/anemoi-training` | **torch** (+cu wheels) + **`anemoi`** (`pbs/trackA.pbs`) |
| **Track B / LoRA** (must exist) | `lapai-credit` | Create with [`environment-credit-lengau.yml`](../environment-credit-lengau.yml); `pbs/student.pbs` / `pbs/lora.pbs` |

Personal environments under `$HOME/.conda/envs/` are **not** used unless you override (see below). A probe for `peft` + `torch` in personal envs found none until **`lapai-credit`** exists.

## Override conda env targets

```bash
qsub -v LAPAI_CONDA_TRACK_A=/path/to/other-anemoi pbs/trackA.pbs
qsub -v LAPAI_CONDA_TRACK_B=my-torch-env pbs/student.pbs
```

Source: [`pbs/inc_conda_lengau.sh`](../pbs/inc_conda_lengau.sh).

## GPU jobs

Batch scripts call `lapai_load_cuda_gpu` which tries:

`module load chpc/cuda/12.0/12.0`, then `chpc/cuda/11.8/11.8`.

## Create `lapai-credit` once (required for Track B PBS)

Interactive or [`pbs/setup_env_lengau.pbs`](../pbs/setup_env_lengau.pbs):

```bash
module load chpc/python/anaconda/3-2024.10.1
source /home/apps/chpc/bio/anaconda3-2024.10.1/etc/profile.d/conda.sh
cd /path/to/lapai-forecast
conda env create -f environment-credit-lengau.yml
conda activate lapai-credit
pip install -e . --no-deps
```

## Optional: dedicated `lapai-anemoi` for Track A

[`scripts/setup_lengau_envs.sh`](../scripts/setup_lengau_envs.sh) builds **`lapai-anemoi`**. PBS defaults to CHPC **`anemoi-training`** unless you prefer isolation:

```bash
qsub -v LAPAI_CONDA_TRACK_A=lapai-anemoi pbs/trackA.pbs
```

## Sanity: personal env with peft + torch

[`scripts/check_peft_envs.sh`](../scripts/check_peft_envs.sh)

```bash
bash scripts/check_peft_envs.sh
```

# Baseline & Week-1 runbook (Lengau)

Use this after `git pull` and a working **`lapai-anemoi`** env ([`docs/LENGAU.md`](../docs/LENGAU.md)).

## 1. Environments

```bash
module load chpc/python/anaconda/3-2024.10.1
source /home/apps/chpc/bio/anaconda3-2024.10.1/etc/profile.d/conda.sh
conda activate lapai-anemoi
pip install huggingface_hub   # once, for downloader
```

Optionally recreate isolated env:

```bash
bash scripts/setup_lengau_envs.sh
```

## 2. Download teacher checkpoint

From repo root:

```bash
python scripts/download_teacher_ckpt.py --local-dir models/teacher
# optional reproducible pin once you chose a HF revision:
# python scripts/download_teacher_ckpt.py --revision <sha_or_tag>
```

Copy the printed path into **`configs/teacher_aifs.yaml`** (`revision` + confirm `local_checkpoint_relative`).

## 3. Smoke: checkpoint file (+ optional `torch.load`)

```bash
python scripts/smoke_teacher_ckpt.py
```

Fails fast if weights are missing; with **torch** installed, prints shallow keys from the pickled checkpoint.

## 4. ERA5 → Zarr (when CDS / GRIB is ready)

- Place GRIB under `data/raw/lapai/era5/n320/` (gitignored).
- Wire [`recipes/recipe_era5_n320.yaml`](../recipes/recipe_era5_n320.yaml) and [`recipes/recipe_era5_n96.yaml`](../recipes/recipe_era5_n96.yaml) to your **`anemoi-datasets`** / ingest driver; uncomment/adjust **`regrid`** for N96 with your toolchain.

## 5. Smoke: Track A PBS

From repo root:

```bash
python training/train_trackA.py --step coarsen --suggest-invocation
qsub pbs/trackA.pbs
```

Expect the [`train_trackA.py`](../training/train_trackA.py) stub until Anemoi training is wired; env activation must succeed (**`lapai-anemoi`**).

## 6. Baseline forecast artefact (Phase 0)

### 6.a Teacher smoke inference (bundled **`dataset: test`**, **`printer`** output)

From repo root (`lapai-anemoi`, checkpoint on disk — see §2–3):

```bash
python scripts/run_aifs_inference.py --dry-run
python scripts/run_aifs_inference.py
```

Uses [`configs/inference_aifs_minimal.yaml`](../configs/inference_aifs_minimal.yaml); swap `input`/`output`/lead time per [Quickstart](https://anemoi-inference.readthedocs.io/en/latest/usage/quickstart.html).

When **`anemoi-inference`** is configured with real ICs (**CDS**/MARS/GRIB) for **`aifs_single_v1.0.ckpt`**:

1. Produce one rollout (e.g. 10-day) NetCDF/Zarr under `data/processed/lapai/`.
2. Record command line, **`revision`**, node type, CUDA module → append below.

### Runs log

| Date (UTC) | User | revision | Job ID | Output path | Notes |
| ---------- | ---- | --------- | ------ | ----------- | ----- |
|            |      |           |        |             |       |

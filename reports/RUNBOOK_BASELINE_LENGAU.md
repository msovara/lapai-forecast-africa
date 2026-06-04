# Baseline & Week-1 runbook (Lengau)

Use this after `git pull` and a working **`lapai-anemoi`** env ([`docs/LENGAU.md`](../docs/LENGAU.md)).

### Minimal copy-paste path (Lengau)

Run inside **`lapai-forecast/`** after §1 (`module load …` + **`conda activate lapai-anemoi`**):

```bash
git pull
python scripts/download_teacher_ckpt.py --local-dir models/teacher
bash scripts/lapai_inference_gate.sh
qsub -v LAPAI_AIFS_INFER_DRY=1 pbs/inference_aifs_teacher.pbs   # optional: check logs, no full forward pass
qsub pbs/inference_aifs_teacher.pbs
```

If anything errors, see **§7**. For a saved forecast file (NetCDF/GRIB), continue with checklist step **6** and **§6**.

**Challenge teacher (gated `C4E-Mvula/n320_gt6`)** instead of public AIFS — `huggingface-cli login` first, then point every step at its config:

```bash
python scripts/download_teacher_ckpt.py --config configs/teacher_n320_gt6.yaml --local-dir models/teacher_n320_gt6
LAPAI_TEACHER_CONFIG=configs/teacher_n320_gt6.yaml bash scripts/lapai_inference_gate.sh
qsub -v LAPAI_TEACHER_CONFIG=configs/teacher_n320_gt6.yaml pbs/inference_aifs_teacher.pbs
```

## 0. Same-day checklist (teacher inference on GPU)

1. **`git pull`** — repo root is `lapai-forecast/`.
2. **`conda activate lapai-anemoi`** (§1).
3. **§2 → §3** — download **`aifs-single-mse-1.0.ckpt`**, **`smoke_teacher_ckpt`**.
4. **`bash scripts/lapai_inference_gate.sh`** — strict verify + inference **`--dry-run`** (on failure the script mentions **§7**).
5. **GPU:** `qsub -v LAPAI_AIFS_INFER_DRY=1 pbs/inference_aifs_teacher.pbs` then `qsub pbs/inference_aifs_teacher.pbs` (see §6.a).
6. **Optional artefact:** set **`LAPAI_INFER_TEMPLATE`** to your edited NetCDF YAML, re-submit; log §6 runs table.

## 1. Environments

```bash
module load chpc/python/anaconda/3-2024.10.1
source /home/apps/chpc/bio/anaconda3-2024.10.1/etc/profile.d/conda.sh
conda activate lapai-anemoi
# huggingface_hub is in lapai-anemoi yaml pip list; pip install once if your env predates it
```

Optionally recreate isolated env — **prefer running this on interactive node `chpclic1`** (RAM + outbound network for conda/pip). Example:

```bash
qsub -I -P CHPC -q normal -l select=1:ncpus=4:mem=48GB -l walltime=8:00:00 \
  -l place=scatter:excl -W x=FLAGS:ADVRES:chpclic1
# then hostname should show chpclic1 — replace CHPC with your PBS project code if denied
# setup defaults to environment-anemoi-nogrib.yml (conda eccodes/cfgrib omitted for old glibc hosts)
bash scripts/setup_lengau_envs.sh
# workstation with glibc ~2.28+ and conda GRIB: LAPAI_ENV_ANEMOI_YML=environment-anemoi.yml bash scripts/setup_lengau_envs.sh
```

Or batch: edit **`pbs/setup_env_chpclic1.pbs`** (`-P`), then **`qsub pbs/setup_env_chpclic1.pbs`** from **`lapai-forecast/`**.

## 2. Download teacher checkpoint

From repo root (`configs/teacher_aifs.yaml` already pins **`revision`** + **`aifs-single-mse-1.0.ckpt`**):

```bash
python scripts/download_teacher_ckpt.py --local-dir models/teacher
# floating main instead of pinned commit:
# python scripts/download_teacher_ckpt.py --local-dir models/teacher --revision ""
```

After download, **`local_checkpoint_relative`** in **`configs/teacher_aifs.yaml`** should match the relative path inside `models/teacher/` (adjust only if you store weights elsewhere).

## 3. Smoke: checkpoint file (+ optional `torch.load`)

```bash
python scripts/smoke_teacher_ckpt.py
```

Fails fast if weights are missing; with **torch** installed, prints shallow keys from the pickled checkpoint. **stderr** includes a one-line **`configs/teacher_aifs.yaml`** summary (HF repo / filename / revision) when those keys parse.

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
python scripts/verify_inference_stack.py --strict
python scripts/run_aifs_inference.py --dry-run
python scripts/run_aifs_inference.py
```

Or on the cluster (GPU PBS job): `qsub pbs/inference_aifs_teacher.pbs` (`qsub -v LAPAI_AIFS_INFER_DRY=1 …` for dry-run only). Use `qsub -v LAPAI_INFER_TEMPLATE=configs/inference_aifs_netcdf_example.yaml …` after you customise that YAML (**`output:`** uncommented).

Uses [`configs/inference_aifs_minimal.yaml`](../configs/inference_aifs_minimal.yaml); swap `input`/`output`/lead time per [Quickstart](https://anemoi-inference.readthedocs.io/en/latest/usage/quickstart.html). For a skeleton with longer lead and file-output examples, see [`configs/inference_aifs_netcdf_example.yaml`](../configs/inference_aifs_netcdf_example.yaml) and [Saving Outputs](https://anemoi.readthedocs.io/projects/inference/en/latest/usage/advanced/saving.html).

When **`anemoi-inference`** is configured with real ICs (**CDS**/MARS/GRIB) for **`aifs-single-mse-1.0.ckpt`**:

1. Produce one rollout (e.g. 10-day) NetCDF/Zarr under `data/processed/lapai/`.
2. Record command line, **`revision`**, node type, CUDA module → append below.

### Runs log

| Date (UTC) | User | revision | Job ID | Output path | Notes |
| ---------- | ---- | --------- | ------ | ----------- | ----- |
| _example_ | you  | `f0bb02c077e…` (from teacher yaml, or HF tip if you floated) | 123456[].chpc.ac.za | `data/processed/lapai/…` | `printer` baseline or first NetCDF |

_Use a short SHA or full commit as in **`configs/teacher_aifs.yaml`**._

## 7. If `lapai_inference_gate.sh` fails

| Symptom | What to try |
|--------|-------------|
| **`teacher_ckpt_exists: MISSING`** | Run **`python scripts/download_teacher_ckpt.py --local-dir models/teacher`**. Paths must match **`local_checkpoint_relative`** in **`configs/teacher_aifs.yaml`**. |
| **`anemoi-inference_path: MISSING`** or **`NOT_INSTALLED`** | **`conda activate lapai-anemoi`**; refresh env from **`environment-anemoi.yml`**. PBS must **`source`** conda via **`pbs/inc_conda_lengau.sh`**. |
| **`torch: NOT IMPORTABLE`** | Same env — avoid mixing login-node system Python with LapAI conda for these scripts. |
| **`--dry-run` OK but real run fails** | Use **GPU**: **`pbs/inference_aifs_teacher.pbs`**. Inspect **`LAPAI`** template/checkpoint lines on **stderr**. |
| **YAML rejected by `anemoi-inference run`** | Align with [Quickstart](https://anemoi-inference.readthedocs.io/en/latest/usage/quickstart.html) and [Saving Outputs](https://anemoi.readthedocs.io/projects/inference/en/latest/usage/advanced/saving.html). |

## 8. Score forecasts vs ERA5 (Africa default)

Once you have a forecast file and matching ERA5 truth (NetCDF or Zarr), score with the
Mvua protocol in [`configs/eval.yaml`](../configs/eval.yaml). Needs the `data` extra
(`pip install -e '.[data]'` — xarray, zarr, pyyaml). Defaults to the **Africa** domain
(`lat -40..40`, `lon -20..55`); use `--domain global` for full-grid benchmark runs.

```bash
python -m evaluation.run_scorecard \
  --pred data/processed/lapai/forecast.nc \
  --truth data/processed/lapai/era5_truth.nc \
  --variables t2m,tp,u10,v10 --temporal 6h,daily \
  --out reports/scorecard_africa.json --markdown
```

Precipitation (`tp`) is accumulated when aggregating to daily; other variables are averaged.
For input-attribution / pruning diagnostics on a checkpoint, see
[`evaluation/attribution_shap.py`](../evaluation/attribution_shap.py) (`--demo` runs without one).

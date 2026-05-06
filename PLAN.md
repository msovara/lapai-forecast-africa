# LapAI-Forecast — Detailed Technical Implementation Plan

**Project:** LapAI-Forecast (Code for Earth, African Stream)
**Goal:** A compressed, laptop-deployable AI NWP model distilled from ECMWF AIFS, with regional adaptation for Africa.
**Status:** Planning (no code yet). This document is the contract for Phases 0–3.
**Project root:** `lapai-forecast/` — a dedicated, self-contained directory sibling to `neoss-csir-chpc/`, `mpas-lengau/`, `wrf-lengau/`, etc. **None** of the existing top-level `configs/`, `recipes/`, `training/`, `inference/`, `evaluation/`, `diagnostics/`, or `utils/` folders (which belong to the IOD / drought / JRA-3Q / T2M work) are touched, shared, or modified by this project.

---

## 0. Source plan reconciliation

The submitted plan had a scrambled "Frameworks" table. The intended mapping, used throughout this document, is:

| Framework        | Source     | Primary role                                                                                  |
| ---------------- | ---------- | --------------------------------------------------------------------------------------------- |
| **Anemoi**       | ECMWF      | Track A: structural compression, attention-head pruning, AIFS checkpoint management.          |
| **MILES-CREDIT** | NSF NCAR   | Track B: lightweight CNN student, InceptionNeXt blocks, weather-specific distillation losses. |
| **PyTorch + LoRA adapters** | — | Phase 3: regional fine-tuning over Africa.                                                    |
| **ONNX Runtime** | —          | Phase 4: laptop-deployable inference package.                                                 |

Targets (from the project brief — held fixed):

- **Hardware target:** Intel i7 CPU, 16 GB RAM, no discrete GPU.
- **Forecast spec:** 10-day global forecast at 1° resolution.
- **Skill envelope:** ≤ 15 % RMSE degradation vs. AIFS baseline globally; ≤ 20 % RMSE degradation on African extreme events.
- **Pruning skill loss budget:** < 3 % CRPS degradation per Track A pruning step.
- **LoRA cost:** ≤ 6 h on a single consumer GPU (e.g. RTX 4090 / 4070 Ti).
- **Compute budget:** 1,650 GPU-hours total on CHPC (500 / 1000 / 100 / 50 across the four activities).
- **Schedule:** 12 weeks.
- **Deliverable:** Open-source ONNX + Python inference package for African NMHS.

---

## 1. End-to-end architecture

```
                      ┌─────────────────────────────────────────┐
                      │  ECMWF AIFS (teacher, public checkpoint)│
                      └──────────────────┬──────────────────────┘
                                         │ (1) checkpoint mgmt
                                         ▼
   ERA5 N320 ──► ERA5 N96 ──► ┌─────────────────────────────┐
   (input)      (coarsened)   │ Track A: Anemoi compression │
                              │  - grid coarsening O96→O48  │
                              │  - attention-head pruning   │
                              │  - CRPS gated re-training   │
                              └──────────────┬──────────────┘
                                             │ teacher_pruned.ckpt
                                             ▼
         ┌────────────────────────────────────────────────────────┐
         │  Track B: MILES-CREDIT student (10–15 M params CNN)    │
         │   InceptionNeXt blocks (3×3 / 1×11 / 11×1 branches)    │
         │   Distillation: MAE + feature (L10,L14) + spectral     │
         └─────────────────────────┬──────────────────────────────┘
                                   │ student_global.ckpt
                                   ▼
              ┌─────────────────────────────────────────┐
              │ Phase 3: LoRA over Africa (ERA5+ENACTS) │
              └─────────────────────────────────────────┘
                                   │ student_africa_loraR.ckpt
                                   ▼
              ┌─────────────────────────────────────────┐
              │ Phase 4: ONNX export + Python package   │
              │   torch → onnx (opset 17) → ORT-CPU     │
              └─────────────────────────────────────────┘
```

---

## 2. Directory layout (fully self-contained under `lapai-forecast/`)

Everything lives below — nothing leaks into the existing top-level project folders. The internal layout mirrors a standard Anemoi project so existing Anemoi/Hydra conventions still work, but inside this dedicated tree.

```
tiny-media-analysis/
├── neoss-csir-chpc/                     (unrelated — left alone)
├── mpas-lengau/                         (unrelated — left alone)
├── wrf-lengau/                          (unrelated — left alone)
├── configs/  recipes/  training/  ...   (existing IOD/JRA/T2M work — left alone)
│
└── lapai-forecast/                      ← **all LapAI work lives here**
    ├── PLAN.md                          ← this document
    ├── README.md                        ← short user-facing intro (Week 1)
    ├── pyproject.toml                   ← installable `lapai_inference` package
    ├── requirements.txt                 ← pinned env (Anemoi + MILES-CREDIT + LoRA + ONNX)
    ├── environment-anemoi.yml           ← conda env for Track A
    ├── environment-credit.yml           ← conda env for Tracks B / 3 / 4
    ├── .gitignore                       ← scoped to this subtree
    │
    ├── recipes/
    │   ├── recipe_era5_n320.yaml        ← Phase 0: full-resolution teacher inputs
    │   └── recipe_era5_n96.yaml         ← Phase 1 A1: coarsened student inputs
    │
    ├── configs/
    │   ├── teacher_aifs.yaml            ← Phase 0: AIFS checkpoint loader config
    │   ├── trackA_coarsen.yaml          ← Phase 1 A1: O96 → O48 graph + grid
    │   ├── trackA_prune.yaml            ← Phase 1 A2: head-pruning schedule
    │   ├── student_global.yaml          ← Phase 2: InceptionNeXt CNN training
    │   ├── student_distill.yaml         ← Phase 2: 3-component loss config
    │   ├── lora_africa.yaml             ← Phase 3: LoRA adapters config
    │   └── eval.yaml                    ← Phase 4: evaluation suite config
    │
    ├── diagnostics/
    │   └── plot/
    │       └── lapai.yaml               ← LapAI-specific diagnostics callbacks
    │
    ├── training/
    │   ├── train_trackA.py              ← drives Anemoi compression + pruning
    │   ├── train_student.py             ← drives MILES-CREDIT student distillation
    │   └── train_lora.py                ← drives LoRA fine-tune
    │
    ├── inference/
    │   ├── run_global.py                ← 10-day rollouts for evaluation
    │   └── run_laptop.py                ← ONNX-RT CPU rollout (deployment demo)
    │
    ├── evaluation/
    │   ├── eval_skill.py                ← RMSE / ACC / CRPS vs. AIFS baseline
    │   └── eval_africa_extremes.py      ← extreme-event skill over Africa
    │
    ├── utils/
    │   ├── grid_coarsen.py              ← N320 → N96, O96 → O48 helpers
    │   ├── losses_distillation.py       ← MAE + feature + spectral losses
    │   ├── lora_adapters.py             ← rank-r adapter injection
    │   └── onnx_export.py               ← torch → ONNX with shape verification
    │
    ├── lapai_inference/                  ← Python package source (Phase 4)
    │   ├── __init__.py
    │   ├── model.py
    │   ├── preprocess.py
    │   ├── postprocess.py
    │   └── cli.py
    │
    ├── containers/
    │   ├── Dockerfile                   ← Ubuntu 22.04 + ORT-CPU
    │   └── Singularity.def              ← derived from Docker image
    │
    ├── pbs/
    │   ├── trackA.pbs                   ← CHPC PBS for Anemoi compression
    │   ├── student.pbs                  ← CHPC PBS for MILES-CREDIT training
    │   └── lora.pbs                     ← CHPC PBS for LoRA fine-tune
    │
    ├── data/                            ← (gitignored) raw + processed datasets
    │   ├── raw/
    │   └── processed/
    │       ├── era5_n320.zarr
    │       ├── era5_n96.zarr
    │       ├── aifs_targets.zarr
    │       └── enacts.zarr
    │
    ├── models/                          ← (gitignored) checkpoints
    │   ├── teacher_coarsened.ckpt
    │   ├── teacher_pruned.ckpt
    │   ├── student_global.ckpt
    │   ├── student_africa_lora_r{4,8,16,32}.safetensors
    │   ├── student_africa_lora.safetensors
    │   └── lapai_forecast.onnx
    │
    ├── logs/                            ← (gitignored) training/eval logs
    │
    └── reports/
        ├── TRACKA_REPORT.md
        ├── LORA_REPORT.md
        └── FINAL_REPORT.md
```

**Hard isolation rules:**

1. No file in `lapai-forecast/` imports from the repo's other top-level folders.
2. The two conda envs (`environment-anemoi.yml`, `environment-credit.yml`) live inside `lapai-forecast/` and are independent of the existing `anemoi-training.yml` at the repo root, which continues to serve the JRA-3Q/T2M pipeline.
3. PBS jobs `cd` into `lapai-forecast/` before launching anything.
4. The `.gitignore` inside `lapai-forecast/` ignores `data/`, `models/`, `logs/`, and `*.onnx` regardless of the repo-root `.gitignore`.

---

## 3. Phase 0 — Environment, data, and baseline (Weeks 1–2)

### 3.1 Environments

Two conda environments, both defined under `lapai-forecast/` and kept independent so Anemoi and MILES-CREDIT do not fight over PyTorch / CUDA pins:

- `lapai-anemoi` (file: `lapai-forecast/environment-anemoi.yml`) — used for Track A and AIFS checkpoint handling.
- `lapai-credit` (file: `lapai-forecast/environment-credit.yml`) — clean env with PyTorch ≥ 2.3 + CUDA 12.x, MILES-CREDIT, and `peft` for LoRA, used for Tracks B/3 and ONNX export.

`lapai-forecast/requirements.txt` (to be created) will pin: `miles-credit`, `peft`, `onnx`, `onnxruntime`, `einops`, `torch-harmonics`, `xarray`, `zarr`, `cdsapi`, `cfgrib`, `eccodes`, plus the `anemoi-*` package set. All versions to be locked in Week 1 from the latest stable releases.

### 3.2 Data inventory

| Dataset                      | Use                            | Format        | Approx. size  | Source                       |
| ---------------------------- | ------------------------------ | ------------- | ------------- | ---------------------------- |
| ERA5 N320 (0.25°)            | Teacher input + ground truth   | GRIB → Zarr   | ~12 TB / decade | CDS / Anemoi recipes         |
| ERA5 N96 (~1°)               | Student input                  | Zarr (regridded) | ~600 GB / decade | derived from N320          |
| AIFS public checkpoint(s)    | Teacher logits + features      | `.ckpt`       | ~3–10 GB       | ECMWF AIFS releases          |
| AIFS rolled-out forecasts    | Distillation targets / cache   | NetCDF/Zarr   | ~2 TB total    | generated locally on CHPC    |
| ENACTS (Africa)              | Phase-3 fine-tune targets      | NetCDF        | ~50 GB         | IRI / national NMHSs (per-country licence) |
| Climatology (1991–2020)      | Anomaly correlation, extremes  | NetCDF        | ~5 GB          | derived from ERA5            |

Data conventions:

- **Years:** train 1979–2018, val 2019–2020, test 2021–2022.
- **Frequency:** 6 h.
- **Variables (start):** `2t, 10u, 10v, msl, sp, tp` at surface; `t, u, v, q, z` at pressure levels {1000, 850, 700, 500, 250, 100, 50} hPa. (Final list locked at end of Week 2.)
- **Storage:** under `lapai-forecast/data/processed/{era5_n320.zarr, era5_n96.zarr, aifs_targets.zarr, enacts.zarr}`.

### 3.3 Recipes (added in Week 1)

`lapai-forecast/recipes/recipe_era5_n320.yaml` — full-resolution teacher input dataset.
`lapai-forecast/recipes/recipe_era5_n96.yaml` — coarsened student input dataset (N320 → N96 via Anemoi `regrid` block).

Both follow the standard Anemoi recipe schema and add a `regrid:` section for the N96 variant.

### 3.4 Baseline reproduction

Deliverable for end of Week 2:

1. AIFS public checkpoint loaded inside `lapai-anemoi` env using `anemoi-inference`.
2. A 10-day global rollout for one initialisation date (e.g. 2022-06-01 00 Z) saved to `lapai-forecast/data/processed/aifs_baseline_20220601.nc`.
3. `lapai-forecast/evaluation/eval_skill.py` produces RMSE / ACC for `2t, msl, z500, t850` against ERA5 — **this is the AIFS baseline number** that all subsequent experiments are measured against.

---

## 4. Phase 1 — Track A: Structural compression with Anemoi (Weeks 3–5)

Two steps, each with its own Hydra config under `lapai-forecast/configs/` and explicit acceptance gate.

### 4.1 Step A1 — Grid coarsening (Week 3)

Config: `lapai-forecast/configs/trackA_coarsen.yaml`.

| Parameter                | Value / decision                              |
| ------------------------ | --------------------------------------------- |
| Input grid               | N320 (≈ 0.25°)                                |
| Coarsened input grid     | N96 (≈ 1°), via `recipe_era5_n96.yaml`        |
| Processor graph          | O96 → **O48** (Anemoi `graph: multi_scale`, processor.nodes set to O48) |
| Encoder/decoder          | Inherited from AIFS topology                  |
| Re-training schedule     | Short fine-tune (5–10 epochs) of the AIFS processor on the new graph to recover skill |
| Acceptance gate          | < 5 % RMSE degradation on `2t / msl / z500 / t850` vs. the Phase 0 AIFS baseline |

Driver: `python lapai-forecast/training/train_trackA.py --step coarsen --config lapai-forecast/configs/trackA_coarsen.yaml`.

### 4.2 Step A2 — Attention-head pruning (Weeks 4–5)

Config: `lapai-forecast/configs/trackA_prune.yaml`.

Procedure (iterative, gated):

1. Score each attention head by its contribution to validation CRPS (importance via gradient × activation).
2. Prune the lowest-importance K % of heads (start K = 10 %, expand if budget allows).
3. Fine-tune for 2–5 epochs.
4. Recompute CRPS for `{2t, msl, z500, t850, tp}`.
5. **Gate:** accept the pruning step iff CRPS degradation < 3 % per the project brief; otherwise revert and reduce K.

| Hyperparameter           | Value / decision                              |
| ------------------------ | --------------------------------------------- |
| Pruning rounds           | 3 (10 %, 20 %, 30 % cumulative — review at each round) |
| Importance metric        | mean(|grad × activation|) over a stratified val mini-batch |
| Recovery fine-tune       | 2 epochs at lr = 5 × 10⁻⁵                     |
| Acceptance metric        | per-variable CRPS, area-weighted                |
| Output                   | `lapai-forecast/models/teacher_pruned.ckpt`    |

Driver: `python lapai-forecast/training/train_trackA.py --step prune --config lapai-forecast/configs/trackA_prune.yaml`.

### 4.3 Track A deliverables (end of Week 5)

- `lapai-forecast/models/teacher_coarsened.ckpt` (post-A1).
- `lapai-forecast/models/teacher_pruned.ckpt` (post-A2).
- `lapai-forecast/reports/TRACKA_REPORT.md` summarising what passed/failed gates, with plots from `diagnostics/plot/lapai.yaml`.

---

## 5. Phase 2 — Track B: Student model with MILES-CREDIT (Weeks 6–9)

### 5.1 Architecture

Config: `lapai-forecast/configs/student_global.yaml`.

A MILES-CREDIT InceptionNeXt CNN with three parallel depthwise branches per block, fused by 1×1 conv:

| Component                | Spec                                                         |
| ------------------------ | ------------------------------------------------------------ |
| Block type               | InceptionNeXt                                                |
| Local branch             | 3 × 3 depthwise (mesoscale)                                  |
| Zonal branch             | 1 × 11 depthwise (east–west waves; periodic padding in lon)  |
| Meridional branch        | 11 × 1 depthwise (north–south gradients)                     |
| Stages                   | 4 (downsample-by-2 between stages)                           |
| Channels per stage       | [96, 192, 384, 384] (target ≈ 12 M params; tuned to land in [10, 15] M) |
| Activation               | GELU                                                         |
| Norm                     | LayerNorm (channels_last)                                    |
| Input tensor             | `(B, V, 181, 360)` for 1° global grid (V = number of channel-stacked variables × pressure levels) |
| Output                   | Same shape, 6-h step prediction; rolled out 40 × for 10-day forecast |
| Lon padding              | Cyclic                                                       |
| Lat padding              | Pole reflection                                              |

Parameter-budget check is performed in unit tests before training launches.

### 5.2 Distillation losses

Config: `lapai-forecast/configs/student_distill.yaml`. Total loss

\[
\mathcal{L} = \lambda_A \mathcal{L}_A + \lambda_B \mathcal{L}_B + \lambda_C \mathcal{L}_C
\]

with starting weights `λ_A = 1.0, λ_B = 0.5, λ_C = 0.25` (re-tuned at end of Week 7).

| Component | Definition                                                   | Target       |
| --------- | ------------------------------------------------------------ | ------------ |
| **A — Area-weighted MAE** | `mean( w(lat) · |ŷ − y_ERA5| )` with `w(lat) = cos(lat)` | ERA5 ground truth |
| **B — Feature distillation** | MSE between student stage-2/stage-3 features and AIFS processor outputs at **layers 10 and 14**, projected via a learned 1×1 conv to match channel counts | Pruned teacher (`teacher_pruned.ckpt`) |
| **C — Spectral distillation** | MSE on the log power spectrum of selected fields (`z500, t850, tp`) computed with `torch-harmonics` SH transform — preserves fine-scale energy | ERA5 ground truth |

Loss helpers live in `lapai-forecast/utils/losses_distillation.py`.

### 5.3 Training schedule

| Stage                 | Duration   | Notes                                                                   |
| --------------------- | ---------- | ----------------------------------------------------------------------- |
| Pre-training (ERA5 only) | Weeks 6–7 | Loss = `L_A` only, 6-h next-step + 1-step rollout regularisation.       |
| Distillation (full loss) | Weeks 8–9 | All three components active. Multi-step rollout: 1 → 4 → 12 → 40 steps. |
| Final fine-tune          | end Week 9 | Lower LR (1 × 10⁻⁵), full 40-step rollout, cos schedule.                |

| Hyperparameter        | Value                                  |
| --------------------- | -------------------------------------- |
| Optimiser             | AdamW (β = (0.9, 0.95), wd = 0.05)     |
| LR schedule           | Linear warmup 1 k steps, cosine decay  |
| Peak LR               | 5 × 10⁻⁴ (pre-train) / 1 × 10⁻⁴ (distill) / 1 × 10⁻⁵ (final) |
| Batch size            | 16 per GPU × 4 GPUs (effective 64)     |
| Mixed precision       | bf16                                   |
| Gradient clipping     | 1.0                                    |
| Rollout curriculum    | 1 → 4 → 12 → 40 steps                  |
| Checkpoint cadence    | every epoch + best on val ACC(z500)    |

Driver: `python lapai-forecast/training/train_student.py --config lapai-forecast/configs/student_global.yaml`.

### 5.4 Track B deliverables (end of Week 9)

- `lapai-forecast/models/student_global.ckpt`.
- Skill table vs. AIFS baseline for `{2t, msl, z500, t850, tp}` at lead times `{24, 72, 120, 240}` h.
- Pass/fail check against the **15 % global RMSE budget**.

---

## 6. Phase 3 — Regional adaptation: LoRA over Africa (Weeks 10–11)

### 6.1 Adapter design

Config: `lapai-forecast/configs/lora_africa.yaml`. Helper: `lapai-forecast/utils/lora_adapters.py`.

| Item                  | Decision                                                                 |
| --------------------- | ------------------------------------------------------------------------ |
| Where injected        | The 1×1 fusion convs of every InceptionNeXt block + final stage 3×3 conv |
| Adapter form          | Conv with low-rank decomposition `W + (B · A)` where `A: r×Cin`, `B: Cout×r` |
| Ranks tested          | `r ∈ {4, 8, 16, 32}` — full sweep per the brief                          |
| Frozen weights        | All non-adapter parameters frozen                                        |
| Domain                | Africa box: lat ∈ [−40, 38], lon ∈ [−20, 55]                             |
| Training data         | ERA5 (Africa subset) + ENACTS where licensed; weighted 70 / 30           |
| Epochs                | 3–5 per rank                                                             |
| Wall-clock target     | ≤ 6 h on a single consumer GPU (RTX 4090 ref.)                           |

### 6.2 Sensitivity analysis

For each `r ∈ {4, 8, 16, 32}`:

1. Fine-tune student.
2. Score on `lapai-forecast/evaluation/eval_africa_extremes.py` (heavy-precip POD/FAR, heatwave bias, drought ACC).
3. Record adapter file size (KB) and consumer-GPU wall-clock (min).

Selection criterion: smallest `r` that achieves **≤ 20 % RMSE degradation** for African extremes (per brief). Output `lapai-forecast/reports/LORA_REPORT.md`.

### 6.3 Phase 3 deliverables (end of Week 11)

- 4 adapter files: `lapai-forecast/models/student_africa_lora_r{4,8,16,32}.safetensors`.
- One **selected** adapter promoted to `lapai-forecast/models/student_africa_lora.safetensors`.
- Skill report covering Africa-specific extremes.

---

## 7. Phase 4 — Evaluation, ONNX export, packaging (Week 12)

### 7.1 Evaluation suite

Config: `lapai-forecast/configs/eval.yaml`. Drivers: `lapai-forecast/evaluation/eval_skill.py`, `lapai-forecast/evaluation/eval_africa_extremes.py`.

Reported metrics:

- **Global:** area-weighted RMSE, ACC, bias, CRPS for `{2t, msl, z500, t850, t2m, 10u, 10v, tp}` at `{24, 72, 120, 240}` h lead.
- **Africa:** the same, plus extreme-event scores (POD, FAR, ETS for tp > 95th-pctl; heatwave detection on 2t).
- **Spectral integrity:** zonal power spectrum vs. ERA5 at z500 and t850 — sanity check for blurring.
- **Compute:** wall-clock per 10-day rollout on the i7/16 GB target machine.

### 7.2 ONNX export

Helper: `lapai-forecast/utils/onnx_export.py`. Inference driver: `lapai-forecast/inference/run_laptop.py`.

Steps:

1. Merge selected LoRA adapter into student weights (W ← W + B·A) so the deployed model has no adapter at runtime.
2. Trace with `torch.onnx.export(..., opset_version=17, dynamic_axes={'time': {0: 'T'}})`.
3. Validate parity: max-abs error ≤ 1 × 10⁻⁴ vs. PyTorch on a 5-step rollout.
4. Apply `onnxruntime.transformers.optimizer` for CPU optimisations.
5. Smoke-test on i7/16 GB: 10-day rollout must complete in < 15 min.

Output: `lapai-forecast/models/lapai_forecast.onnx` (target size: ≤ 100 MB after quantisation; FP32 first, then INT8 dynamic quantisation as a stretch goal).

### 7.3 Packaging

The `lapai-forecast/lapai_inference/` Python package exposes:

```python
from lapai_inference import LapAIForecast

model = LapAIForecast.from_pretrained("lapai-forecast/models/lapai_forecast.onnx")
forecast = model.forecast(initial_state, steps=40)   # 10 days × 4 = 40 × 6h
forecast.to_netcdf("forecast.nc")
```

with a CLI `lapai-forecast --init 2026-04-29T00 --hours 240 --out forecast.nc`.

Distribution:

- Python wheel built from `lapai-forecast/pyproject.toml` → PyPI test index.
- `lapai-forecast/containers/Dockerfile` (Ubuntu 22.04 + ORT-CPU) → publishable to GHCR.
- `lapai-forecast/containers/Singularity.def` derived from the Docker image for HPC-restricted NMHS sites.

### 7.4 Phase 4 deliverables (end of Week 12)

- `lapai_forecast.onnx` + adapter.
- Public `lapai_inference` package + Docker/Singularity images.
- `lapai-forecast/reports/FINAL_REPORT.md` summarising skill, footprint, deployment results.

---

## 8. CHPC compute plan

Aligns with the brief's 1,650 GPU-h budget. A single A100 / H100 partition is assumed.

| Activity                  | GPU-h | Wall-clock @ 4 GPUs | PBS template                          |
| ------------------------- | ----- | ------------------- | ------------------------------------- |
| Track A (compression)     | 500   | ~5.2 days           | `lapai-forecast/pbs/trackA.pbs`       |
| Track B (student)         | 1,000 | ~10.4 days          | `lapai-forecast/pbs/student.pbs`      |
| Phase 3 (LoRA × 4 ranks)  | 100   | ~1.0 day            | `lapai-forecast/pbs/lora.pbs`         |
| Evaluation & validation   | 50    | ~0.5 day            | reused `pbs/lora.pbs` template        |
| **Total**                 | **1,650** | **~17 days wall-clock** |                                  |

Buffer: the 12-week schedule has ≈ 4 weeks of slack on top of the ~17 wall-clock days, intentionally absorbed by data prep, gating decisions, and re-runs after failed gates.

PBS script outlines (to be written in Week 1):

- All scripts `cd "$PBS_O_WORKDIR/lapai-forecast"` before launching.
- 96 GB RAM / node, 8 CPUs / GPU, NVMe scratch for Zarr caches.
- `module load chpc/cuda/12.x`, conda env activation, `OMP_NUM_THREADS=8`.
- All checkpoints written to `lapai-forecast/models/`; logs to `lapai-forecast/logs/<run-name>/`.

---

## 9. Risks and mitigations

| #  | Risk                                                                        | Likelihood | Impact | Mitigation                                                                                         |
| -- | --------------------------------------------------------------------------- | ---------- | ------ | -------------------------------------------------------------------------------------------------- |
| R1 | AIFS internal layer indices (10, 14) differ across released checkpoints     | Medium     | High   | Pin a specific AIFS release in Week 1; freeze layer mapping in `configs/teacher_aifs.yaml`.        |
| R2 | Pruning gate (< 3 % CRPS) cannot be met at 30 % cumulative                  | Medium     | Medium | Reduce K per round; fall back to 20 % cumulative; document trade-off.                              |
| R3 | Spectral loss destabilises training                                         | Medium     | Medium | Start with `λ_C = 0` for the first 1 k steps; warm-up to 0.25.                                     |
| R4 | 15 % global / 20 % Africa skill budget breached                             | Medium     | High   | Add multi-step rollout supervision; revisit channel widths up to 15 M params; revisit lon padding. |
| R5 | i7/16 GB target cannot run 10-day rollout in < 15 min                       | Low        | High   | Fall back to 12-h step (rollout × 20 instead of × 40); INT8 quantise; reduce vertical channels.    |
| R6 | ENACTS data licence blocks redistribution                                   | High       | Low    | Treat ENACTS as private fine-tune signal only; ship only the resulting adapter, not the data.      |
| R7 | Anemoi / MILES-CREDIT API drift during 12 weeks                             | Medium     | Medium | Pin both at known-good commits in `lapai-forecast/requirements.txt` Week 1; re-eval at Week 6.     |
| R8 | CHPC queue contention                                                        | Medium     | Medium | Use checkpoint-and-resume aggressively; prefer many short jobs over one long job for Track B.      |
| R9 | Accidental coupling with the existing JRA-3Q / IOD / T2M scaffolding        | Low        | Medium | Hard isolation rules in §2; no imports across the boundary; separate conda envs and PBS scripts.   |

---

## 10. Open decisions (to close in Week 1)

1. **AIFS checkpoint release** to pin (latest stable vs. a specific frozen release).
2. **Variable list V** — final set of surface + pressure-level fields (drives student input channel count and parameter budget).
3. **ENACTS access** — confirm which African countries' data is available under what licence.
4. **Africa box** — keep the proposed `[−40, 38] × [−20, 55]` or extend to include Mascarenes / Madagascar offshore.
5. **Quantisation scope** — INT8 dynamic only, or also weight-only INT4? (Affects accuracy vs. RAM trade-off.)
6. **CI** — add a lightweight GitHub Actions matrix for the `lapai_inference` package now or only at Week 12?

These are the only items that should block Week 2 from starting on time.

---

## 11. Acceptance criteria (definition of done)

The project ships when **all** of the following are true:

- [ ] `lapai_forecast.onnx` runs a 10-day, 1° global forecast on i7/16 GB in < 15 min.
- [ ] Global RMSE for `{2t, msl, z500, t850}` at 5-day lead within **15 %** of AIFS baseline.
- [ ] African extreme-event RMSE within **20 %** of AIFS baseline.
- [ ] LoRA fine-tune reproducible on a single RTX 4090 in **≤ 6 h** from `student_global.ckpt`.
- [ ] `lapai_inference` package installable via `pip install lapai-inference` on a clean Windows / Linux laptop.
- [ ] Docker and Singularity images published.
- [ ] `FINAL_REPORT.md` and per-phase reports committed under `lapai-forecast/reports/`.

---

## 12. Week-by-week milestones

| Week | Phase    | Concrete deliverables                                                                 |
| ---- | -------- | ------------------------------------------------------------------------------------- |
| 1    | 0        | `lapai-forecast/` directory tree created; pinned `requirements.txt`; both conda envs created; AIFS checkpoint downloaded; data plan signed off; PBS templates drafted. |
| 2    | 0        | Recipes `recipe_era5_n320.yaml` and `recipe_era5_n96.yaml` produce Zarr stores; AIFS baseline rollout + skill table archived. |
| 3    | 1 (A1)   | Coarsened processor (O96 → O48) trained; `teacher_coarsened.ckpt`; A1 gate passed.    |
| 4    | 1 (A2)   | First pruning round (10 %); CRPS gate passed; report 1.                                |
| 5    | 1 (A2)   | Rounds 2–3 (20 %, 30 %); final `teacher_pruned.ckpt`; `TRACKA_REPORT.md`.              |
| 6    | 2        | Student architecture implemented; param-budget unit tests green; ERA5 pre-training started. |
| 7    | 2        | Pre-training complete (loss `L_A`); intermediate skill snapshot.                       |
| 8    | 2        | Distillation training with `L_A + L_B + L_C`; rollout curriculum 1 → 12.               |
| 9    | 2        | Final fine-tune at full 40-step rollout; `student_global.ckpt`; 15 % gate evaluated.   |
| 10   | 3        | LoRA implementation tested; runs for `r ∈ {4, 8}` complete on consumer GPU.            |
| 11   | 3        | Runs for `r ∈ {16, 32}`; sensitivity report; selected adapter promoted.                |
| 12   | 4        | Adapter merged; ONNX export + parity test; laptop benchmark; `lapai_inference` package and containers published; `FINAL_REPORT.md`. |

---

## 13. What is *not* in scope

To keep scope realistic for 12 weeks and 1,650 GPU-h:

- Ensemble forecasting (only deterministic outputs are produced; CRPS is computed with a single forecast against ERA5).
- Data assimilation — initial conditions are taken straight from ERA5 / operational analyses.
- Tropical cyclone tracking, gust diagnostics, and other downstream products beyond direct model variables.
- Re-training AIFS from scratch — we only fine-tune and prune existing public checkpoints.
- Mobile / browser deployment — laptop CPU is the only deployment target.
- **Any modification of the existing JRA-3Q / IOD / drought / T2M / MPAS / WRF / ICON code** in this repo. Those projects continue under their own folders, untouched.

---

## 14. Next concrete actions (when you say "go")

1. Resolve the six open decisions in §10.
2. Inside `lapai-forecast/`, create `requirements.txt`, `environment-anemoi.yml`, `environment-credit.yml`, `recipes/recipe_era5_n320.yaml`, `recipes/recipe_era5_n96.yaml`, and the three PBS templates under `pbs/`.
3. Stand up `lapai-anemoi` and `lapai-credit` conda envs on CHPC.
4. Reproduce the AIFS baseline rollout (Phase 0 §3.4) and write the result to `lapai-forecast/data/processed/aifs_baseline_20220601.nc`.

These four items are the entire content of Week 1 and unblock every subsequent phase.

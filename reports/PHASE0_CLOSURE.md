# Phase 0 closure — baseline teacher + Africa scorecard

## Current status (2026-06-29)

| Item | Status |
|------|--------|
| Lengau GPU + lustre Python | OK (V100, job 7318486) |
| Teacher checkpoint | OK (`~/models/teacher_n320_gt6/inference.ckpt`) |
| Phase 0 forecasts | **Done** — 5 × Jan 2023 weekly inits, ~59 MB each |
| Scorecard JSON | **Done** — `reports/PHASE0_BASELINE_SCORECARD.json` (GCS truth, laptop score-only) |
| Offline workflow | CDS IC cache + earthkit regrid cache (laptop populate → rsync); scoring on laptop |

**Phase 0 is closed.** Short-lead teacher baseline (example 20230101 +24h): t2m RMSE 2.2 K, ACC 0.93; tp/u10/v10 scored for all inits and leads 24–240 h.

### Long-lead t2m note (investigated 2026-06-29)

Scoring alignment is **correct** (forecast `valid_time` matches init + lead for all leads). Fields **evolve** between 6-hour steps (~1.6–2.3 K mean step-to-step change). RMSE growth (+24h → +240h: 2.2 K → 35 K) and negative ACC at +120h+ reflect **teacher rollout skill loss on the Africa box**, not a NetCDF or scorecard bug. Use **+24–72 h** for Track A compression gates until longer leads are validated against Mvula protocol.

### Lessons learned (Lengau offline)

| Blocker | Fix |
|---------|-----|
| No outbound internet on GPU/login nodes | CDS IC cache + earthkit regrid matrices on laptop, rsync to Lengau |
| ecCodes missing on cluster | `LD_LIBRARY_PATH` → `grib_env` in PBS |
| CRLF in PBS/scripts | `sed -i 's/\r$//'` before `qsub` |
| GCS scoring on cluster | Run `run_phase0_closure.py --score-only` on laptop with `gcsfs` + ADC |
| Duplicate fields at all lead times | `snapshot_forecast_state()` when saving runner output |
| tp GCS Zarr `(time, step)` layout | `_select_tp_at_valid_time()` in `gcs_era5_truth.py` |

### CDS pipeline vs aifs-africa (regrid map)

| Step | Module | Function | Direction |
|------|--------|----------|-----------|
| CDS IC download | `utils/cds_ic.py` | `ensure_cds_grib_cached`, `build_cds_input_state` | ERA5 GRIB 0.25° (cached under `~/.cache/aifs-africa/era5`, same MD5 keys as aifs-africa) |
| IC regrid to model | `utils/regrid.py` | `regrid_to_n320` | 0.25° → N320 |
| GPU inference | `scripts/run_phase0_forecast.py` | Anemoi runner | N320 state |
| Eval NetCDF | `utils/eval_forecast_io.py` | `regrid_state_fields_to_africa` | via `utils/regrid.regrid_n320_to_latlon025` (N320 → 0.25°), then Africa crop |

Canonical public regrid API: **`utils/regrid.py`**. Lower-level helpers remain in `utils/cds_ic.py` (input) and `utils/n320_forecast_io.py` (plotting / legacy `_regrid_*` names).

### Track A extension — IC regridding (design decision)

Phase 0 intentionally stops at **N320 ICs**. Track A adds a coarsened **O96** student whose inference grid differs from that pipeline.

| Approach | Where | When |
| -------- | ----- | ---- |
| **Mario (canonical benchmark)** | O96 interpolation in the open-data / IC builder (`get_open_data()` / `utils/cds_ic.py`), via `earthkit-regrid` | Oxford / online; **2024–2025 verification** |
| **Implemented first (Lengau offline)** | N320 CDS ICs → post-fetch `utils/grid_bridge.py` in `run_phase0_forecast.py` when IC grid ≠ checkpoint grid | Track A smoke / offline CHPC |

We did **not** implement Mario’s hook first because (1) Phase 0 locked in N320 before Track A existed, (2) Lengau offline caches were built for N320 + N320→0.25° earthkit matrices only, and (3) the bridge was the smallest change to prove train → forecast → score → gate on Lengau. Coarsened **training** uses native O96 Zarr; only **forecast ICs** used the bridge.

**Going forward:** O96-at-fetch is the production definition; the grid bridge remains an **offline fallback** on Lengau. Full rationale and code map: [`PLAN.md` §4.1 — Design decision: IC regridding](../PLAN.md).

---

## Next steps (do in order)

### Step 1 — Windows / laptop: CDS API + populate IC cache

1. Ensure `~/.cdsapirc` exists (Climate Data Store account — meeting action item #3).
2. From `lapai-forecast/`:

```powershell
conda activate lapai-anemoi
python scripts/populate_phase0_ic_cache.py --dry-run
python scripts/populate_phase0_ic_cache.py
```

This downloads ERA5 for all five Jan 2023 inits (+ t−6 h) into `~/.cache/aifs-africa/era5/` (20 GRIB files: 2 times × 2 datasets × 5 inits).

Optional: reuse existing partial cache from prior `aifs-africa` work — same hash keys.

### Step 2 — Sync cache + updated scripts to Lengau

```powershell
# Updated lapai-forecast scripts (after git pull or scp)
scp -r lapai-forecast/scripts lapai-forecast/utils lapai-forecast/configs msovara@lengau.chpc.ac.za:~/repos/lapai-forecast/

# CDS GRIB cache (large; rsync preferred)
scp -r $env:USERPROFILE\.cache\aifs-africa\era5 msovara@lengau.chpc.ac.za:~/.cache/aifs-africa/
```

On Lengau, fix CRLF if needed:

```bash
sed -i 's/\r$//' pbs/phase0_baseline_lengau.pbs scripts/*.py
```

### Step 3 — Lengau: resubmit GPU closure job

```bash
cd ~/repos/lapai-forecast
qsub pbs/phase0_baseline_lengau.pbs
tail -f ~/lapai_phase0.o<JOBID>
```

The PBS script sets `LAPAI_CDS_OFFLINE=1` — inference reads cache only, no CDS download on cluster.

### Step 4 — Scoring (GCS vs local truth)

| Where | Command | Needs |
|-------|---------|--------|
| Lengau (if GCS works) | default closure job | `gcloud auth application-default login` on login node |
| Windows (recommended if cluster has no GCS) | `python scripts/run_phase0_closure.py --score-only` | GCS ADC locally |
| Fully offline | `--score-only --truth local` | `download_era5_eval_truth.py --years 2023 --convert` first |

### Step 5 — Phase 0 closed when

- `data/processed/phase0/forecasts/202301{01,08,15,22,29}_00Z.nc` exist (t2m, tp, u10, v10)
- `reports/PHASE0_BASELINE_SCORECARD.json` written
- Update `PLAN.md` → proceed to Track A

---

Phase 0 is **closed** when all items below are true:

1. C4E teacher checkpoint runs on Lengau GPU (`models/teacher_n320_gt6/inference.ckpt`)
2. Five Jan 2023 weekly inits are forecast with **t2m, tp, u10, v10** on the eval Africa grid
3. `reports/PHASE0_BASELINE_SCORECARD.json` exists (Mvua protocol, leads 24–240 h)

That JSON is the **teacher skill ceiling** for Track A/B compression — do not start pruning or distillation until it exists.

---

## Prerequisites

| Item | Where |
|------|--------|
| Lengau GPU + lustre `lapai-anemoi` Python | [`docs/LENGAU.md`](../docs/LENGAU.md) |
| Hugging Face access to `C4E-Mvula/n320_gt6` | `huggingface-cli login` |
| **CDS API key** (`~/.cdsapirc`) | Populate IC cache on laptop |
| **CDS GRIB cache on Lengau** | `~/.cache/aifs-africa/era5/` (rsync from laptop) |
| GCS ADC (for `--truth gcs`) | `gcloud auth application-default login` |
| Optional local ERA5 truth | `python scripts/download_era5_eval_truth.py --years 2023 --convert` |

---

## One-command closure (Lengau)

From **`lapai-forecast/`** repo root:

```bash
module load chpc/python/anaconda/3-2024.10.1
source /home/apps/chpc/bio/anaconda3-2024.10.1/etc/profile.d/conda.sh
conda activate lapai-anemoi
qsub pbs/phase0_baseline_lengau.pbs
```

When the job finishes:

```bash
cat reports/PHASE0_BASELINE_SCORECARD.json
cat reports/PHASE0_MANIFEST.json
```

Forecasts land in `data/processed/phase0/forecasts/YYYYMMDD_00Z.nc`.

---

## Step-by-step (interactive)

### 1. Gate + checkpoint

```bash
python scripts/download_teacher_ckpt.py --config configs/teacher_n320_gt6.yaml --local-dir models/teacher_n320_gt6
LAPAI_TEACHER_CONFIG=configs/teacher_n320_gt6.yaml bash scripts/lapai_inference_gate.sh
```

### 2. Populate CDS cache (laptop) + single init smoke (GPU)

```bash
# Laptop — download IC GRIBs (internet required)
python scripts/populate_phase0_ic_cache.py --dry-run
python scripts/populate_phase0_ic_cache.py

# Lengau — dry-run then full forecast (CDS cache must exist)
python scripts/run_phase0_forecast.py --init 20230101 --dry-run
python scripts/run_phase0_forecast.py --init 20230101 --cds-offline
```

### 3. Full closure (all inits + scorecard)

```bash
python scripts/run_phase0_closure.py --dry-run
python scripts/run_phase0_closure.py
```

Score only (forecasts already present):

```bash
python scripts/run_phase0_closure.py --score-only
```

Use **local ERA5 truth** instead of GCS:

```bash
python scripts/download_era5_eval_truth.py --years 2023 --convert
python scripts/run_phase0_closure.py --score-only --truth local
```

---

## Score existing aifs-africa outputs (partial Phase 0)

If you already have regridded NetCDF from `aifs-africa` (may only contain `2t`, `tp`):

```bash
python scripts/run_phase0_closure.py --score-only \
  --forecast-dir /path/to/aifs-africa/output \
  --inits 20230101
```

Missing variables appear as `"error": "missing in forecast file"` in the scorecard — rerun with `run_phase0_forecast.py` for full protocol compliance.

---

## Configuration

Edit [`configs/phase0_baseline.yaml`](../configs/phase0_baseline.yaml):

- `inits` — init dates (default: five Mondays in Jan 2023)
- `ic_source` — `cds` (default, offline cache on Lengau) or `opendata`
- `cds_cache_dir` — GRIB cache root (default `~/.cache/aifs-africa/era5`)
- `lead_time_hours` — inference rollout (default 240 h = 40 × 6 h steps)
- `leads_hours` — scoring leads (must fit inside rollout)
- `truth.source` — `gcs` (team bucket) or `local` (CDS Zarr)

---

## Deliverables checklist

| File | Purpose |
|------|---------|
| `data/processed/phase0/forecasts/*.nc` | Eval-grid teacher forecasts |
| `reports/PHASE0_BASELINE_SCORECARD.json` | Multi-init, multi-lead Africa metrics |
| `reports/PHASE0_MANIFEST.json` | Paths + inits index |

After closure, update `PLAN.md` status and proceed to **Track A** (§Phase 1 in [`PLAN.md`](../PLAN.md)).

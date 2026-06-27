# Phase 0 closure — baseline teacher + Africa scorecard

Phase 0 is **closed** when all items below are true:

1. C4E teacher checkpoint runs on Lengau GPU (`models/teacher_n320_gt6/inference.ckpt`)
2. Five Jan 2023 weekly inits are forecast with **t2m, tp, u10, v10** on the eval Africa grid
3. `reports/PHASE0_BASELINE_SCORECARD.json` exists (Mvua protocol, leads 24–240 h)

That JSON is the **teacher skill ceiling** for Track A/B compression — do not start pruning or distillation until it exists.

---

## Prerequisites

| Item | Where |
|------|--------|
| Lengau GPU + `lapai-anemoi` | [`docs/LENGAU.md`](../docs/LENGAU.md) |
| Hugging Face access to `C4E-Mvula/n320_gt6` | `huggingface-cli login` |
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

### 2. Single init smoke (GPU)

```bash
python scripts/run_phase0_forecast.py --init 20230101 --dry-run
python scripts/run_phase0_forecast.py --init 20230101
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

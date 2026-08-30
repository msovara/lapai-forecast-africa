# Day 0 diagnosis + locked v6 criteria (path start)

**Date:** 2026-08-29  
**Status:** **CLOSED — REJECT (2026-08-30).** Keep frozen `student_global_stable_v5.ckpt` / tag `trackb-v5-c4e`. Do **not** promote `v5_t2mRMSE` → v6.

## Locked v6 promotion criteria

Promote `student_global_stable_v5_t2mRMSE.ckpt` → `student_global_stable_v6.ckpt` only if:

| Metric | v5 baseline | Accept as v6 |
|--------|-------------|--------------|
| +6 h Africa t2m RMSE | 1.382 | **≤ 1.28** (~≥7% better) |
| +6 h ACC | 0.965 | **≥ 0.96** |
| +24 h RMSE | 3.231 | **≤ 3.55** (not >10% worse) |

Stretch: +6 h RMSE ≤ 1.25.  
Reject: +6 h RMSE > 1.38, ACC collapse, or training NaNs.  
Keep `trackb-v5-c4e` / v5 freeze untouched regardless.

## Smoke gate result (20 inits × leads 6/12/24)

Source: `TRACKB_V6_T2M_GATE.json` / `TRACKB_T2M_T2MRMSE_SMOKE.*`

| Lead | Smoke RMSE | Smoke ACC | Gate |
|-----:|-----------:|----------:|------|
| +6 h | **1.377** | 0.967 | **FAIL** RMSE (need ≤1.28); ACC pass |
| +12 h | 7.704 | 0.459 | Diagnostic only (still cold-biased) |
| +24 h | 3.343 | 0.833 | Pass ≤3.55 |

**Verdict:** REJECT — continue-train was essentially flat vs v5 full (+6 h 1.377 vs 1.382). Paper/report numbers remain on v5.

## Day 0 findings (from TRACKB_T2M_EXPANDED, n=61)

| Lead | RMSE | Bias (K) | ACC |
|-----:|-----:|---------:|----:|
| +6 h | 1.382 | −0.12 | 0.965 |
| +12 h | **7.800** | **−5.33** | 0.446 |
| +18 h | 4.377 | −1.12 | 0.749 |
| +24 h | 3.231 | +1.30 | 0.835 |

**+12 h is a systematic cold bias:** bias ∈ [−5.76, −4.91] K on **61/61** inits, **all seasons**. Not a random fail — likely **diurnal / lead-conditioned** error when AF step is 06Z→12Z (00Z inits). Continue-train may help +6 h; **do not expect +12 h to become K1-like** from this branch alone.

Spatial bias (+6 h): near-zero mean; residual noise at coasts/orography.  
Spatial bias (+24 h): Sahara cold / E–S Africa warm dipole — regional systematic error.

## Path (completed)

1. Continue-train → `v5_t2mRMSE` — **done**.  
2. Smoke AF eval — **done**.  
3. Accept → v6 — **rejected**; keep v5.

See also `TRACKB_T2M_RMSE_REDUCTION_PLAN.md`.

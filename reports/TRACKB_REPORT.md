# Track B report — student MVP gate + held-out Jan-2023

> **State closure (2026-08-26):** **Case A** — Cout=3 (`tp`/`msl`/`2t`) is intentional; free-run is not possible. See [`TRACKB_STATE_CLOSURE.md`](TRACKB_STATE_CLOSURE.md).

**Date:** 2026-08-25  
**MVP gate:** **FAIL 2/3** (≤15% RMSE degradation vs K1 on cache) — **accepted soft-fail for MVP**.  
**Decision:** Do **not** run another tp-reweight training round. Promote `student_global_stable_v3.ckpt` as the Track B MVP student and proceed to **held-out Jan-2023** skill vs K1.

## Soft-fail wording (accepted)

Stable v3 closes the cache MVP with **msl and 2t beating K1 on-cache**, while **tp remains ~21% worse than K1** (bar ≤15%). Further tp-only reweight/continue-train is deferred: on-cache tp has been stuck ~21% across v1→v3, and the next evidence that matters for Track B is **off-cache Jan-2023 skill**, not another distillation-cache gate. Documented trade-off: **accept tp soft-fail for MVP; 2/3 cache gate; move to held-out eval.**

## Gate outcome (stable v3 — accepted MVP)

MVP protocol `trackB_cache_mvp` scored **`models/student_global_stable_v3.ckpt`** (resume from v2, 80 more epochs, channel-weighted L_A, `teacher_k1_cache_t256.zarr` T=256, 6h one-step, channels `tp/msl/2t`) against K1 `teacher_pred` and ERA5 targets in the same store.

| var | student RMSE vs ERA5 | teacher RMSE | degradation vs teacher | v2 (T=128) | v1 (T=32) |
|-----|----------------------|--------------|------------------------|------------|-----------|
| tp  | 0.00255              | 0.00211      | **21.2%** (fail ≤15%)  | 21.1%      | 22.5%     |
| msl | 207                  | 250          | **−17.2%** (pass)      | −6.6%      | 81.7%     |
| 2t  | 2.50                 | 2.90         | **−13.8%** (pass)      | 30.2%      | 86.2%     |

### Levers applied (v3)

1. **Channel-weighted L_A** — `channel_weights=[3.0, 0.4, 3.0]` for `(tp, msl, 2t)` so tp/2t dominate the loss; msl already passed under v2.
2. **Continue from `student_global_stable_v2.ckpt`** — 80 epochs, `steps_per_epoch=64`, β/γ still off, prior stabilizations kept (channel-norm, soft tp, grad clip).
3. **Larger cache** — built `teacher_k1_cache_t256.zarr` (T=256, stride 4 from `era5_n96_2020_2021.zarr`, ~85 min on GPU1).

Train: L_A ~0.34 → ~0.27 (finite; weighted scale). Log: `/local/Mthetho/logs/trackB_student_stable_v3_train.log`. Gate: `/local/Mthetho/logs/trackB_stable_v3_gate.log`.

**Pass/fail:** **FAIL** overall on the strict 3/3 bar, but **accepted as MVP soft-fail** (tp only). **msl and 2t both beat the teacher on-cache.**

## Prior: stable v2

| var | degradation | |
|-----|-------------|---|
| tp  | 21.1% fail  | |
| msl | −6.6% pass  | |
| 2t  | 30.2% fail  | |

Levers: β/γ=0, 80 epochs, T=128 cache. See logs `trackB_student_stable_v2_train.log` / `trackB_stable_v2_gate.log`.

**Artifacts kept:** `models/student_global_stable_v3.ckpt` (MVP), `…_v2.ckpt`, `…_stable.ckpt` (v1), `models/student_global.ckpt` (catastrophic) — do not promote broken ckpt.

**MVP artifacts:** `reports/TRACKB_STUDENT_SCORECARD.json`, `reports/TRACKB_GATE.json`, runner `evaluation/trackB_gate.py`.

---

## Held-out Jan-2023 (next step — executed)

Runner: `evaluation/trackB_held_out_jan2023.py`  
Results: `reports/TRACKB_HELD_OUT_JAN2023.json`

### Protocol actually run

1. **K1 multi-lead (free-running):** score existing `trackA_prune_k1_a1bplus` Jan-2023 weekly NetCDFs vs GCS ERA5 Africa truth at leads **{6, 24, 72, 120, 240}** h for **`t2m, tp, u10, v10`**.
2. **Student 6h held-out (analysis-forced):** CDS ERA5 IC cache (`~/.cache/aifs-africa/era5`) → 65-ch 1° state → `student_global_stable_v3.ckpt` → score **`tp` / `t2m`** on Africa vs GCS and vs K1 +6h from the same forecast files (Cassava GPU1).

### Explicit non-goals / gaps (do not claim PLAN compliance)

- No student free-running multi-day rollout (student heads are `tp/msl/2t` only).
- No **z500 / t850** in this table (neither student heads nor Track A forecast NCs).
- **msl** not scored held-out (no GCS truth stem; K1 forecast NCs lack msl).
- Multi-lead columns are **K1-only**; student column is **+6h only**.

### Key skill numbers

**K1 free-running Africa RMSE (mean over 5 Jan-2023 weekly inits) vs GCS ERA5:**

| lead | t2m | tp | u10 | v10 |
|------|-----|----|-----|-----|
| 6 h  | 1.21 | 0.00132 | 1.19 | 1.27 |
| 24 h | 1.47 | 0.00051 | 1.96 | 2.13 |
| 72 h | 2.44 | 0.00081* | 3.15 | 3.24 |
| 120 h | 3.09 | 0.00099 | 3.13 | 3.38 |
| 240 h | 3.46 | 0.00115* | 4.68 | 4.14 |

\*tp n=4 at 72 h / 240 h (one init missing finite tp truth).

**Student v3 held-out +6h (CDS IC → Africa score vs GCS; same 5 inits):**

| var | student RMSE | K1 RMSE | degradation vs K1 | notes |
|-----|--------------|---------|-------------------|-------|
| tp  | 0.000389 | 0.001322 | **−70.0%** | low RMSE but ACC≈0 and POD₁ₘₘ=0 → dry-bias artifact, not skill |
| t2m | 3.26 | 1.21 | **+168%** | large held-out gap vs on-cache (−13.8%) |

Student–teacher field RMSE (Africa, no ERA5): tp 0.00151, t2m 2.88.

### Artifacts

- `reports/TRACKB_HELD_OUT_JAN2023.json`
- Runner: `evaluation/trackB_held_out_jan2023.py`
- Cassava student forecasts (not in git): `data/processed/trackB_student_heldout/forecasts/`
- Merge/rescore helper: `scripts/merge_trackB_held_out_student.py`
- Cassava launchers: `scripts/run_trackB_held_out_*cassava.sh`

---

## Generalization v4 (2026-08-26) — held-out t2m fix

**Goal:** improve **held-out** Jan-2023 Africa t2m / honest tp skill (not cache-gate 3/3).  
**Ckpt:** `models/student_global_stable_v4.ckpt` (resume v3; β/γ still off).

### Root cause / mask fix

First v4 train crashed on `africa_hw_mask`: in-place `&=` on a `(H,1)` lat mask cannot expand to `(H,W)`. Fixed in `evaluation/masks.py` to build `(lat_ok & lon_ok)` via broadcasting → shape `(181,360)`, `mask_frac≈0.0945`.

### Levers (v4)

1. Full teacher cache `teacher_k1_cache_full2020_2021.zarr` (T=730; already built — not rebuilt).
2. Africa-weighted L_A (`africa_mix=0.5`) after mask fix.
3. Input channel z-score (`normalize_inputs=true`) for train/held-out parity.
4. Precip-aware term (`precip_log1p_weight=2.0`) + `tp_mode=softplus`.
5. Channel weights `[4.0, 0.3, 3.0]` for `(tp, msl, 2t)`.

### Cache MVP gate (optional regression; full cache, global)

| var | degradation vs K1 | |
|-----|-------------------|---|
| tp  | **+22.1%** fail (≤15%) | soft-fail unchanged |
| msl | **−1.1%** pass | |
| 2t  | **−58.4%** pass | |

Africa-domain cache: tp +16.1%, msl −65.6%, 2t −83.4%.

### Held-out Jan-2023 +6h Africa (before → after)

| var | v3 deg vs K1 | v4 deg vs K1 | v3 RMSE | v4 RMSE | K1 RMSE | notes |
|-----|--------------|--------------|---------|---------|---------|-------|
| t2m | **+168%** | **+23.9%** | 3.26 | 1.50 | 1.21 | ACC≈0.97; large generalization win |
| tp  | −70% RMSE* | −70% RMSE* | 0.000389 | 0.000389 | 0.00132 | still ACC≈0, POD₁ₘₘ=0 → dry-bias, not skill |

\*Identical tp RMSE/ACC/POD to v3: softplus + precip log1p did **not** escape all-dry collapse on held-out CDS ICs.

Student–teacher field RMSE (Africa): t2m **2.88 → 1.34**; tp unchanged 0.00151.

### Artifacts (v4)

- Reports: `TRACKB_GATE_V4.json`, `TRACKB_HELD_OUT_JAN2023_V4.json`, `TRACKB_HELD_OUT_JAN2023_V4_MERGED.json`, scorecards `TRACKB_STUDENT_SCORECARD_V4*.json`
- Config: `configs/student_global_v4.yaml`
- Train/eval: `scripts/run_trackB_stable_v4.sh`, `scripts/run_trackB_v4_post_eval.sh`, `scripts/merge_trackB_held_out_v4.py`
- Cassava forecasts (not in git): `data/processed/trackB_student_heldout_v4/forecasts/`

---

## Focused tp recovery v5 (2026-08-26) — ONE attempt; tp out-of-scope

**Decision:** **Declare precipitation (`tp`) out-of-scope for this Cout=3 student head.** Stop further tp-only chase. Keep `student_global_stable_v4.ckpt` / `v5` for **t2m (+ msl on-cache)**; do not claim precip skill.

**Ckpt:** `models/student_global_stable_v5.ckpt` (resume v4; β/γ still off; full cache reused, not rebuilt).

### Levers tried (v5 vs v4)

Kept from v4 (t2m worked): Africa mix `0.5`, input z-score, soft physicality, channel-norm L_A.

Added / strengthened for tp:

| lever | v4 | v5 |
|-------|----|----|
| `tp_mode` | softplus (×1000) | **softplus_soft (×50)** |
| `precip_log1p_weight` | 2.0 | **6.0** |
| `precip_wet_boost` | 4 (default) | **12** |
| under-pred on wet | — | **4.0** |
| soft POD hinge (≥1 mm) | — | **2.0** |
| dry-collapse mean match | — | **1.0** |
| `tp_logit_boost` on resume | — | **+0.0002** |
| channel weights (tp,msl,2t) | [4, 0.3, 3] | **[6, 0.25, 2.5]** |
| epochs | 100 | 80 |

Train log: L_tp stuck **~1.30** all 80 epochs (v4 L_tp stuck ~0.50). Held-out student `tp` field is **identically zero** (min=max=mean=0).

### Held-out Jan-2023 +6h Africa (ACC/POD primary; RMSE secondary)

| var | metric | v4 | v5 | K1 | notes |
|-----|--------|----|----|-----|-------|
| tp | ACC | ≈0 | **−0.003** | ~0.52 | still no spatial skill |
| tp | POD₁ₘₘ | **0.0** | **0.0** | ~0.77 | all-dry |
| tp | RMSE | 0.000389 | 0.000389 | 0.00132 | identical dry-bias artifact |
| t2m | deg vs K1 | +23.9% | **+16.4%** | — | still OK / slightly better |
| t2m | ACC | ~0.97 | **0.977** | ~0.983 | kept |

### Cache gate (regression only)

Global: tp +22.1% fail; msl −18.1% pass; 2t −63.5% pass (same soft-fail pattern as v4).

### Multi-lead / free-run progress (secondary)

- **Implemented:** analysis-forced multi-lead path via `--student_leads` (e.g. `6,24`): each lead = ERA5 IC at `init+(lead−6)h` + one +6h step. Aggregates include ACC/POD by lead.
- **Ran:** +6h and **+24h** on Cassava GPU1 with **public ARCO ERA5 ICs** (no live CDS).
- **CDS bypass (option 2):** `utils/era5_ondisk_ic.py` builds the same 65-ch 1° student state from `gs://gcp-public-data-arco-era5/...` (anon). `--ic_source auto` tries CDS GRIB cache then ARCO; `--ic_source arco` forces ARCO. Local npy cache: `data/cache/student_ic_arco/`.
- **+24h unblock:** 18Z CDS GRIBs were MISS on Cassava (no `.cdsapirc`); ARCO supplies 18Z pressure-level fields without CDS.
- **Free-run:** **not supported** — student Cout=`tp/msl/2t` cannot update 65-ch state. Do **not** claim PLAN multi-day free-run or z500/t850 compliance.

### Held-out analysis-forced multi-lead (v5, ARCO IC, ARCO surface truth)

Protocol: analysis-forced only. IC at `init+(L−6)h` → one +6h step. Surface truth for student/K1 comparison: public ARCO ERA5 (code4earth ADC unavailable on Cassava for this run). Teacher multi-lead table retained from prior code4earth scoring.

| lead | var | student RMSE | K1 RMSE | deg vs K1 | student ACC | K1 ACC |
|------|-----|--------------|---------|-----------|-------------|--------|
| 6 h  | t2m | **1.415** | 1.214 | **+16.6%** | **0.977** | 0.983 |
| 6 h  | tp  | 0.000389 | 0.001322 | −70.0%* | ≈0 | 0.518 |
| 24 h | t2m | **2.913** | 1.475 | **+97.5%** | **0.904** | 0.973 |
| 24 h | tp  | 0.000358 | 0.000507 | −26.8%* | ≈0 | 0.415 |

\*tp still all-dry (POD₁ₘₘ=0) — out-of-scope; do not read as precip skill.

### Artifacts (v5)

- Reports: `TRACKB_GATE_V5.json`, `TRACKB_HELD_OUT_JAN2023_V5.json`, `TRACKB_HELD_OUT_JAN2023.json` (canonical ARCO multi-lead), `TRACKB_HELD_OUT_JAN2023_V5_L24.json` (prior CDS-fail), scorecards `TRACKB_STUDENT_SCORECARD_V5*.json`
- IC path: `utils/era5_ondisk_ic.py`; `--ic_source {auto,cds,arco}` on `evaluation/trackB_held_out_jan2023.py`
- Launch/rescore: `scripts/run_trackB_held_out_arco_l6_24_cassava.sh`, `scripts/rescore_trackB_held_out_arco.py`
- Config: `configs/student_global_v5.yaml`
- Train/eval: `scripts/run_trackB_stable_v5.sh`, `run_trackB_v5_post_eval.sh`, `run_trackB_v5_pipeline.sh`, `run_trackB_v5_multilead24.sh`, `merge_trackB_held_out_v5.py`
- Cassava forecasts (not in git): `data/processed/trackB_student_heldout_v5/forecasts/`
- Cassava IC npy cache (not in git): `data/cache/student_ic_arco/`


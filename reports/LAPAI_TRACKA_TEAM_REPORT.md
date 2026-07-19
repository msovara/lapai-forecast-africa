# LapAI-Forecast — Track A progress report

**Author:** Mthetho Sovara (CHPC Lengau)  
**Project:** LapAI-Forecast · Code for Earth, African Stream  
**Date:** July 2026  
**Status:** Phase 0 closed · Track A smoke pipeline end-to-end · A1 gate not passed (expected)

---

## 1. Executive summary

We have **closed Phase 0** (N320 AIFS teacher baseline on Lengau) and **validated the full Track A coarsening pipeline end-to-end**: train coarsened O96 student → offline forecast → GCS scorecard → A1 acceptance gate.

The **A1 gate did not pass** on the current **smoke configuration** (50-step fine-tune, reduced model size, partial training Zarr). That outcome was **expected** and does **not** indicate a broken pipeline — it reflects insufficient training and model capacity, not a fundamental architecture failure.

**Key messages for discussion:**

1. **Infrastructure is ready** on Lengau for coarsened student work (offline CDS, grid bridge, lustre Python stack, PBS jobs).
2. **Scalar verification** (domain RMSE/ACC vs ERA5 on GCS) is automated; **spatial map comparison** (ERA5 vs teacher vs coarsened) is now available in the Streamlit status dashboard.
3. **Initial-condition handling** for the coarsened benchmark should align with **Mario’s O96-at-fetch design** for 2024–2025 production; our current **post-fetch grid bridge** is an offline Lengau workaround, not the long-term definition.
4. **2024–2025 verification** should be split: **Oxford (Sh)** for N320 (+ optional native O96 AIFS) production runs; **Lengau** for coarsened O96 student train/infer and paired scoring.

---



## 2. Project context

**Goal:** Compress the C4E `n320_gt6` AIFS teacher to a deployable O96 student while preserving forecast skill over Africa, measured against a Phase 0 teacher baseline and ERA5 truth (Mvula evaluation protocol).

**Track A Step A1 (current focus):**


| Item               | Target                                                             |
| ------------------ | ------------------------------------------------------------------ |
| Input grid         | N320 teacher → **O96** coarsened student                           |
| Training           | Warm-start from teacher; native **O96 ERA5 Zarr**                  |
| Verification inits | Jan 2023 weekly (pipeline shakedown); **2024–2025** for production |
| Gate variables     | t2m, u10, v10 at **+24 h** and **+48 h**                           |
| Gate criterion     | ≤ **5%** RMSE degradation vs Phase 0 teacher (Africa domain)       |
| Truth              | ERA5 on GCS (`gs://code4earth/era5`)                               |


**Compute:** CHPC Lengau V100 GPUs (offline — no outbound internet on compute nodes).

---



## 3. What is complete



### 3.1 Phase 0 — N320 teacher baseline (closed 2026-06-29)


| Deliverable             | Location                                                  |
| ----------------------- | --------------------------------------------------------- |
| Teacher checkpoint      | `models/teacher_n320_gt6/inference.ckpt` (C4E `n320_gt6`) |
| Five Jan 2023 forecasts | `data/processed/phase0/forecasts/*_00Z.nc`                |
| Baseline scorecard      | `reports/PHASE0_BASELINE_SCORECARD.json`                  |
| Closure documentation   | `reports/PHASE0_CLOSURE.md`                               |


**Example teacher skill (20230101, +24 h, Africa, t2m):** RMSE **2.23 K**, ACC **0.93**.

**Offline workflow proven:** CDS GRIB cache + earthkit N320 regrid matrices populated on laptop → rsync to Lengau → GPU inference with `--cds-offline` → scoring on laptop with GCS.

### 3.2 Track A — coarsened O96 smoke pipeline (closed machinery)


| Step             | Result                                                                                                                            | Lengau job (example) |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| Environment      | Unified lustre stack: anemoi-training 0.14, anemoi-models 0.16, anemoi-inference 0.11; offline fixes for sklearn (GLIBC), trimesh | —                    |
| Coarsen training | 50-step smoke; **finite loss** (~0.13); O96 checkpoint saved                                                                      | 7318569              |
| Forecast         | All **5** Jan 2023 inits; N320→O96 IC bridge; eval NetCDF ~59 MB each                                                             | 7318585              |
| Scorecard        | `reports/TRACKA_COARSEN_SCORECARD.json` (GCS truth, laptop)                                                                       | —                    |
| A1 gate          | `reports/TRACKA_A1_GATE.json` — **FAILED**                                                                                        | —                    |


**Example coarsened skill (20230101, +24 h, Africa, t2m):** RMSE **3.52 K**, ACC **0.88** (vs teacher 2.23 K / 0.93).

### 3.3 Documentation and team artefacts


| Artefact                                  | Purpose                                                 |
| ----------------------------------------- | ------------------------------------------------------- |
| `PLAN.md` §4.1                            | IC regridding design decision (Mario vs bridge)         |
| `reports/PHASE0_CLOSURE.md`               | Phase 0 + Track A IC extension                          |
| `reports/LAPAI_STATUS_DECK.pdf` / `.pptx` | Slide deck for meetings                                 |
| `streamlit_status.py`                     | Live dashboard: metrics, spatial maps, gate tables      |
| Git `main` (lapai-forecast)               | Track A PBS, grid bridge, warm-start converter, configs |


---



## 4. Pipeline architecture



### 4.1 Phase 0 (N320 teacher)

```
CDS ERA5 cache (0.25° GRIB)
    → earthkit regrid → N320 ICs
    → N320 AIFS teacher rollout
    → regrid to Africa 0.25° eval grid
    → score vs GCS ERA5
```



### 4.2 Track A today (coarsened O96 smoke)

```
O96 ERA5 Zarr → coarsen training (warm-start) → O96 checkpoint

CDS ERA5 cache → N320 ICs → offline grid bridge → O96 ICs
    → O96 coarsened rollout
    → regrid to Africa 0.25° eval grid
    → score vs GCS ERA5 → A1 gate vs Phase 0
```

**Training** uses native **O96 Zarr**. The **IC workaround** applies only at **forecast** time.

### 4.3 Target architecture (2024–2025 benchmark)

```
CDS ERA5 cache → earthkit O96 ICs at fetch (Mario’s pattern)
    → O96 coarsened rollout
    → same eval grid and scorecard
```

Lengau remains offline: O96 earthkit matrices must be **pre-cached on laptop/login** and rsync’d (same pattern as N320 Phase 0). The post-fetch bridge stays as **fallback only**.

---



## 5. Initial conditions — discussion point for Mario / Sh

**Question from Sh:** Are we interpolating ICs to O96 inside `get_open_data()` as Mario suggested?

**Answer today:** **Not exactly.** The coarsened model **does** receive O96 ICs before rollout, but via a **post-fetch offline bridge** (`utils/grid_bridge.py`), not inside the IC builder.


|             | Current (Lengau)                                                                          | Target (production)               |
| ----------- | ----------------------------------------------------------------------------------------- | --------------------------------- |
| IC fetch    | N320 CDS path (Phase 0)                                                                   | Model-grid-aware IC builder       |
| O96 regrid  | scipy k-NN IDW after fetch                                                                | `earthkit-regrid` at fetch        |
| Why first?  | Phase 0 was N320-first; O96 earthkit cache not pre-staged; fastest path to green pipeline | Canonical coarsened benchmark     |
| Offline OK? | Yes                                                                                       | Yes, with pre-staged O96 matrices |


**Agreed direction:** Adopt Mario’s O96-at-fetch for **2024–2025 verification**; keep bridge as offline fallback on Lengau.

---



## 6. A1 gate results (smoke run)

**Gate definition:** t2m, u10, v10 at +24 h and +48 h; pass if RMSE degradation vs Phase 0 teacher ≤ 5%.

**Outcome:** **FAILED** — **15 / 30** checks passed.

### 6.1 Main pattern


| Variable | +24 h                                               | +48 h            |
| -------- | --------------------------------------------------- | ---------------- |
| **t2m**  | **Failed all 5 inits** (~37–58% worse than teacher) | Mixed            |
| **u10**  | Mixed (often **better** than teacher at +24 h)      | Several failures |
| **v10**  | Mixed                                               | Several failures |


**t2m +24 h degradation (% vs teacher), all inits:**


| Init     | Degradation |
| -------- | ----------- |
| 20230101 | +57.8%      |
| 20230108 | +37.7%      |
| 20230115 | +48.0%      |
| 20230122 | +36.1%      |
| 20230129 | +40.5%      |
| **Mean** | **~44%**    |




### 6.2 Why failure is expected (not alarming)


| Factor          | Smoke configuration                                                          |
| --------------- | ---------------------------------------------------------------------------- |
| Training length | **50 steps** only                                                            |
| Model capacity  | OOM-reduced: 256 channels, 8 layers                                          |
| Training data   | Partial smoke Zarr (90/360 timesteps missing; restricted to NaN-free window) |
| Purpose         | Prove **machinery**, not production skill                                    |




### 6.3 Interpretation

- The **pipeline** (train → forecast → score → gate) is **correct and repeatable**.
- The **coarsened smoke model** is not yet a fair test of A1 skill — a **longer fine-tune on a complete Zarr** with full model capacity is required before treating gate failure as a scientific conclusion.
- **Short-lead gates (+24–48 h)** remain appropriate; long-lead teacher RMSE growth on the Africa box is a separate known issue (see `PHASE0_CLOSURE.md`).

---



## 7. Spatial verification (new)

Domain-mean scorecards do not show **where** errors occur. We added **spatial map panels** comparing:

1. **ERA5** (GCS truth)
2. **AIFS teacher** (Phase 0 NetCDF)
3. **Coarsened O96** (Track A NetCDF)

Plus difference fields: Teacher − ERA5, Coarsened − ERA5, Coarsened − Teacher.

**Access:** `streamlit run streamlit_status.py` → **Spatial maps** tab.  
**Data:** `data/processed/phase0/forecasts/` and `data/processed/trackA/forecasts/` (Africa 0.25° grid).  
**ERA5:** Requires GCS ADC locally; can compare teacher vs coarsened without ERA5 if offline.

**Known gap:** `tp` is **NaN** in coarsened forecast NetCDF — precipitation not yet in spatial or gate comparisons for Track A.

---



## 8. Proposed 2024–2025 verification split

Based on discussion with **Sh (Oxford)**:


| Owner                | Responsibility                                                                                                                             |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **Oxford (Sh)**      | N320 AIFS production forecasts 2024–2025 (~5–9 days GPU for 2 years); optional native O96 AIFS; CDS pipeline; Mario-style O96 ICs at fetch |
| **Lengau (Mthetho)** | Coarsened O96 **student** training + inference; paired scoring vs Phase 0 and Oxford outputs; offline cache prep                           |
| **Shared**           | Same init calendar, leads, variables, scorecard JSON format; joint comparison N320 vs O96 reference vs coarsened O96                       |


**Open clarification for meeting:** Does “O96 in parallel” mean **coarsened student only** from Lengau, or also **native O96 AIFS** from Lengau? We recommend: **Oxford = native N320 + O96 AIFS; Lengau = coarsened student.**

---



## 9. Blockers cleared (Lengau)


| Blocker                      | Resolution                                                   |
| ---------------------------- | ------------------------------------------------------------ |
| GLIBC / sklearn on GPU nodes | Offline wheel (`scripts/install_sklearn_fix.sh`)             |
| Train vs infer version skew  | Unified lustre env (anemoi 0.14 / 0.16 / 0.11)               |
| Warm-start checkpoint format | `scripts/convert_inference_to_warmstart_ckpt.py`             |
| `trimesh` missing offline    | Pre-downloaded wheel                                         |
| NaN training loss            | Incomplete smoke Zarr diagnosed; clean date window in config |
| V100 fp16 overflow           | `training.precision="32"` for smoke                          |
| No GPU internet              | CDS + earthkit cache prep on laptop; score on laptop (GCS)   |
| O96 IC grid mismatch         | Offline N320→O96 grid bridge                                 |


---



## 10. Next steps (prioritised)



### Near term (Lengau)

1. **Rebuild training Zarr** — full 2018 Q1 or 2020–2021 subset; fix missing timesteps.
2. **Longer coarsening fine-tune** — drop OOM overrides where GPU memory allows.
3. **Re-run** forecast → score → gate; target A1 pass on t2m/u10/v10 @ +24/+48 h.
4. **Implement Mario IC path** — `_regrid_to_o96` in `cds_ic.py`; extend `populate_earthkit_regrid_cache.py` for O96 matrices.
5. **Fix** `tp` **in forecast NetCDF** for full Mvula protocol.



### Medium term (team)

1. **Lock 2024–2025 init calendar** and file naming with Oxford.
2. **Paired verification campaign** — three streams scored with common scorecard.
3. **Track A report** with diagnostics plots (`diagnostics/plot/lapai.yaml`) after real gate attempt.

---



## 11. Questions for colleagues

1. **Init calendar:** Weekly vs bi-weekly through 2024–2025? Same dates for all three model streams?
2. **O96 reference:** Native O96 AIFS from Oxford only, or also required from Lengau?
3. **IC benchmark:** Confirm adoption of Mario’s O96-at-fetch as canonical; bridge as Lengau fallback only?
4. **Gate thresholds:** Is 5% RMSE @ +24/+48 h still the right A1 bar, or adjust for coarsened smoke vs production?
5. **Spatial diagnostics:** Which variables/leads should be standard in team reports (t2m +24 h minimum)?
6. **Meeting:** Sh suggested debrief + slides for Mario — use `LAPAI_STATUS_DECK.pdf` and Streamlit dashboard?

---



## 12. References and artefacts


| Resource          | Path / link                                        |
| ----------------- | -------------------------------------------------- |
| Repo              | `lapai-forecast` (GitHub: `lapai-forecast-africa`) |
| Phase 0 scorecard | `reports/PHASE0_BASELINE_SCORECARD.json`           |
| Track A scorecard | `reports/TRACKA_COARSEN_SCORECARD.json`            |
| A1 gate           | `reports/TRACKA_A1_GATE.json`                      |
| Slide deck        | `reports/LAPAI_STATUS_DECK.pdf`                    |
| Dashboard         | `streamlit run streamlit_status.py`                |
| Design doc (IC)   | `PLAN.md` §4.1, `reports/PHASE0_CLOSURE.md`        |


**Lengau jobs (reference):** training 7318569 · forecast 7318585

---



## Appendix A — Plain-language summary

We got the **coarsened weather model working all the way through** on Lengau: train it, run forecasts offline, score against ERA5, and run the acceptance test. The test **failed**, but only because we used a **tiny practice run** (50 training steps, smaller model, incomplete data) — the **plumbing works**.

We are **aligning with Sh** on who runs which GPU jobs for **2024–2025**: Oxford for full N320 (and likely native O96) AIFS; Lengau for the **coarsened student**. We should use **Mario’s approach** for O96 initial conditions in production; our current shortcut was to unblock offline Lengau.

---


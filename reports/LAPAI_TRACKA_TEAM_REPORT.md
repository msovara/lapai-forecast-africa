# LapAI-Forecast — Track A progress report

**Author:** Mthetho Sovara (CHPC Lengau)  
**Project:** LapAI-Forecast · Code for Earth, African Stream  
**Date:** August 2026  
**Status:** Phase 0 closed · **Track A Step A1 PASSED** (30/30) · next: A2 pruning

---

## 1. Executive summary

We have **closed Phase 0** (N320 AIFS teacher baseline on Lengau) and **passed Track A Step A1** (grid coarsening): train coarsened O96 student → offline forecast → GCS scorecard → acceptance gate vs Phase 0.

**A1 gate: PASSED 30/30** (2026-08-06). After a 2000-step full fine-tune left t2m +24h cold-biased, a **+15 000-step extend** from `teacher_coarsened.ckpt` recovered skill. On all five Jan 2023 inits, coarsened **t2m +24h RMSE is ~25–33% better** than the Phase 0 teacher (gate allows ≤5% worse).

**Key messages for discussion:**

1. **A1 is closed** — artefact `reports/TRACKA_A1_GATE.json`; checkpoint via `models/teacher_coarsened.ckpt` → extend `inference-last.ckpt`.
2. **Infrastructure is ready** on Lengau for coarsened student work (offline CDS, grid bridge, lustre Python stack, PBS jobs).
3. **Scalar verification** (domain RMSE/ACC vs ERA5 on GCS) is automated; **spatial map comparison** is in the Streamlit status dashboard.
4. **Initial-condition handling** for production should still align with **Mario’s O96-at-fetch design**; our post-fetch grid bridge remains the Lengau offline fallback.
5. **Next technical step:** Track A **A2** (attention-head pruning) — CRPS gate includes **`tp`**.

---



## 2. Project context

**Goal:** Compress the C4E `n320_gt6` AIFS teacher to a deployable O96 student while preserving forecast skill over Africa, measured against a Phase 0 teacher baseline and ERA5 truth (Mvula evaluation protocol).

**Track A Step A1 (closed 2026-08-06):**


| Item               | Target                                                             |
| ------------------ | ------------------------------------------------------------------ |
| Input grid         | N320 teacher → **O96** coarsened student                           |
| Training           | Warm-start from teacher; native **O96 ERA5 Zarr** (2020–2021)      |
| Verification inits | Jan 2023 weekly (A1 gate); **2024–2025** for production            |
| Gate variables     | t2m, u10, v10 at **+24 h** and **+48 h**                           |
| Gate criterion     | ≤ **5%** RMSE degradation vs Phase 0 teacher (Africa domain)       |
| Truth              | ERA5 on GCS (`gs://code4earth/era5`)                               |
| Outcome            | **PASSED 30/30**                                                   |


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

### 3.2 Track A — coarsened O96 smoke pipeline (machinery proven)


| Step             | Result                                                                                                                            | Lengau job (example) |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| Environment      | Unified lustre stack: anemoi-training 0.14, anemoi-models 0.16, anemoi-inference 0.11; offline fixes for sklearn (GLIBC), trimesh | —                    |
| Coarsen training | 50-step smoke; **finite loss** (~0.13); O96 checkpoint saved                                                                      | 7318569              |
| Forecast         | All **5** Jan 2023 inits; N320→O96 IC bridge; eval NetCDF ~59 MB each                                                             | 7318585              |
| Scorecard        | Early smoke scorecard (superseded by full/extend)                                                                                 | —                    |
| A1 gate (smoke)  | **FAILED** (expected — under-trained)                                                                                             | —                    |


**Smoke skill (20230101, +24 h, Africa, t2m):** RMSE **3.52 K** (vs teacher 2.23 K) — not a fair A1 attempt.

### 3.2b Track A — full A1 (passed 2026-08-06)


| Step | Result | Lengau job |
| ---- | ------ | ---------- |
| Full Zarr | `era5_n96_2020_2021.zarr` (2924 × 6h, O96, on lustre) | — |
| Full fine-tune | 2000 steps, ~8.5 min, epoch loss ~0.070 | **7353532** (gpu2005) |
| Gate after 2k | **25/30** — all five **t2m +24h** failed (cold bias ~−1.5 K) | — |
| Extend fine-tune | +15 000 steps from coarsened ckpt, epoch loss ~0.042 | **7357693** (gpu2005) |
| Forecast | 5 Jan 2023 inits, Exit 0 | **7357781** / **7357782** |
| Scorecard | `reports/TRACKA_COARSEN_SCORECARD.json` | laptop / GCS |
| **A1 gate** | **`reports/TRACKA_A1_GATE.json` — PASSED 30/30** | — |


**Production A1 skill (20230101, +24 h, Africa, t2m):** RMSE **1.67 K** (teacher 2.23 K) — **~25% better** than Phase 0.

Checkpoint: `models/teacher_coarsened.ckpt` →  
`models/trackA_coarsen_full_extend_runs/checkpoint/b828d680-6d0e-4665-a5da-b9d0c8d56649/inference-last.ckpt`  
Configs: `trackA_coarsen_full.yaml`, `trackA_coarsen_full_extend.yaml`. Operator notes: `reports/TRACKA_FULL_TRAIN_CHECKLIST.md`.

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



## 6. A1 gate results

**Gate definition:** t2m, u10, v10 at +24 h and +48 h; pass if RMSE degradation vs Phase 0 teacher ≤ 5%.

### 6.0 Production A1 (extend 15k) — PASSED 30/30

| Variable | +24 h (mean deg vs Phase 0) | +48 h (mean deg) |
| -------- | --------------------------- | ---------------- |
| **t2m**  | **−29%** (all pass; better than teacher) | **−69%** |
| **u10**  | **−38%** | **−20%** |
| **v10**  | **−47%** | **−35%** |

**t2m +24 h (Africa RMSE, K):**

| Init     | Phase 0 | Coarsened (15k) | Degradation |
| -------- | ------- | --------------- | ----------- |
| 20230101 | 2.234   | 1.667           | −25.4%      |
| 20230108 | 2.312   | 1.675           | −27.6%      |
| 20230115 | 2.170   | 1.561           | −28.1%      |
| 20230122 | 2.343   | 1.593           | −32.0%      |
| 20230129 | 2.499   | 1.664           | −33.4%      |

**Path to pass:** 2000-step full train → t2m +24h still failed (cold bias). Extend +15 000 steps fixed skill without score hacks. Constant mean debias alone cleared only 2/5 inits.

### 6.1 Smoke run (historical) — FAILED 15/30

| Variable | +24 h                                               | +48 h            |
| -------- | --------------------------------------------------- | ---------------- |
| **t2m**  | **Failed all 5 inits** (~37–58% worse than teacher) | Mixed            |
| **u10**  | Mixed                                               | Several failures |
| **v10**  | Mixed                                               | Several failures |

### 6.2 Why smoke failure was expected

| Factor          | Smoke configuration                 |
| --------------- | ----------------------------------- |
| Training length | **50 steps** only                   |
| Model capacity  | OOM-reduced: 256 channels, 8 layers |
| Training data   | Partial smoke Zarr                  |
| Purpose         | Prove **machinery**, not skill      |

### 6.3 Interpretation

- The **pipeline** (train → forecast → score → gate) is **correct and repeatable**.
- **A1 is closed** on the full 2020–2021 Zarr + extend fine-tune.
- **`tp`** is on the scorecard for reporting; it is **not** an A1 gate variable — planned for **A2**.
- **Short-lead gates (+24–48 h)** remain appropriate; long-lead teacher issues are separate (`PHASE0_CLOSURE.md`).

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

**Known gap (updated):** `tp` is present in forecast NetCDFs and scored on the scorecard; it is **not** in the A1 gate. Prefer 6h-accumulated ERA5 truth before treating precip RMSE as primary.

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

### Near term

1. ~~Rebuild training Zarr / longer fine-tune / re-run A1 gate~~ — **done** (A1 30/30, 2026-08-06).
2. **Start A2** — attention-head pruning (`configs/trackA_prune.yaml`); CRPS gate includes `tp`.
3. **Implement Mario IC path** — `_regrid_to_o96` in `cds_ic.py`; extend earthkit O96 matrix cache for Lengau offline.
4. **Harden `tp` scoring** — sum GCS hourly precip to 6h to match forecast accumulation.

### Medium term (team)

1. **Lock 2024–2025 init calendar** and file naming with Oxford.
2. **Paired verification campaign** — three streams scored with common scorecard.
3. **Track A report** (`TRACKA_REPORT.md`) + diagnostics after A2.

---



## 11. Questions for colleagues

1. **Init calendar:** Weekly vs bi-weekly through 2024–2025? Same dates for all three model streams?
2. **O96 reference:** Native O96 AIFS from Oxford only, or also required from Lengau?
3. **IC benchmark:** Confirm adoption of Mario’s O96-at-fetch as canonical; bridge as Lengau fallback only?
4. **Gate thresholds:** A1 5% bar is met (with margin). Keep the same for A2 CRPS (&lt;3%), or adjust?
5. **Spatial diagnostics:** Which variables/leads should be standard in team reports (t2m +24 h minimum)?
6. **Meeting:** Sh suggested debrief + slides for Mario — use `LAPAI_STATUS_DECK.pdf` and Streamlit dashboard?

---



## 12. References and artefacts


| Resource          | Path / link                                        |
| ----------------- | -------------------------------------------------- |
| Repo              | `lapai-forecast` (GitHub: `lapai-forecast-africa`) |
| Phase 0 scorecard | `reports/PHASE0_BASELINE_SCORECARD.json`           |
| Track A scorecard | `reports/TRACKA_COARSEN_SCORECARD.json`            |
| A1 gate           | `reports/TRACKA_A1_GATE.json` (**passed 30/30**)   |
| Team note (A1)    | `reports/TEAM_NOTE_A1_PASSED.md`                   |
| Operator checklist| `reports/TRACKA_FULL_TRAIN_CHECKLIST.md`           |
| Slide deck        | `reports/LAPAI_STATUS_DECK.pdf`                    |
| Dashboard         | `streamlit run streamlit_status.py`                |
| Design doc (IC)   | `PLAN.md` §4.1, `reports/PHASE0_CLOSURE.md`        |


**Lengau jobs (reference):** training 7318569 · forecast 7318585

---



## Appendix A — Plain-language summary

We got the **coarsened weather model working all the way through** on Lengau: train it, run forecasts offline, score against ERA5, and run the acceptance test. The test **failed**, but only because we used a **tiny practice run** (50 training steps, smaller model, incomplete data) — the **plumbing works**.

We are **aligning with Sh** on who runs which GPU jobs for **2024–2025**: Oxford for full N320 (and likely native O96) AIFS; Lengau for the **coarsened student**. We should use **Mario’s approach** for O96 initial conditions in production; our current shortcut was to unblock offline Lengau.

---


# Mvula / LapAI — Internal technical report  
### Compressed AIFS-derived student for African short-range t2m: protocol, skill, and laptop accessibility

**Status:** Internal draft (Dueben-style evaluation report) — upgrade path to preprint noted in §10  
**Date:** 2026-08-30  
**Programme:** ECMWF Code for Earth 2026 — African Stream (Mvula)  
**Repo:** https://github.com/msovara/lapai-forecast-africa  
**Freeze tag:** `trackb-v5-c4e`  
**Primary artefact:** `models/student_global_stable_v5.ckpt`  
**Authors (draft):** Mthetho Vuyo Sovara (with team context: Chimwemwe Chanda et al.)  
**Mentors (programme):** Shruti Nath, Rendani Mbuvha, Mario Santa Cruz  

**Control inputs:** `TRACKB_T2M_EXPANDED.*`, `MVULA_LAPTOP_BENCHMARK.*`, `TRACKB_STATE_CLOSURE.md`, `FINAL_REPORT.md`, `TRACKB_V6_T2M_PATH.md`

---

## 0. How to use this document

| Audience | Use |
|----------|-----|
| Team / mentors | Single source for what was measured and what is claimable |
| Future paper | Lift §§1–8 almost verbatim; replace “internal” framing in §10 |
| SAWS / partners | Use §1 + §6 + §8 only (short brief) |

**Claim boundary (read first):** This report evaluates an **analysis-forced**, **partial-state** student for African **2 m temperature**. It does **not** claim free-running 10-day forecasts, precipitation skill, coastal hazard products, or an operational NMHS system.

---

## 1. Executive summary

Mvula asks whether advanced AI weather models can be **compressed** enough to support African short-range temperature evaluation on **ordinary laptop hardware**, without pretending the result is a full dynamical NWP emulator.

**Main findings (v5 freeze)**

1. **Compression:** Distilled CNN student is **8.8 MiB / 2.17 M params**, ~**6×** smaller on disk than the accepted K1 pruned GraphTransformer teacher (52.8 MiB).  
2. **Accessibility:** On a consumer Windows laptop (**i7-11800H**, ~32 GB RAM), CPU-only inference runs in **~2.53 s** per +6 h step; peak process RSS ~**1.19 GiB**; **no GPU required**.  
3. **Skill (analysis-forced African t2m, n=61 inits, 2023 seasons):**  
   - **+6 h:** RMSE **1.382 K**, ACC **0.965**, bias ≈ **−0.12 K** (~**+16.6%** RMSE vs K1 on n=3 shared inits).  
   - **+24 h:** RMSE **3.231 K**, ACC **0.835**, bias ≈ **+1.30 K** (~**+103%** vs K1).  
   - RMSE growth +6→+24 h ≈ **+134%**, persistent across DJF/MAM/JJA/SON.  
4. **Failure mode:** **+12 h** is pathological (RMSE **7.80 K**, bias **−5.33 K** on **61/61** inits) — systematic cold bias, not random noise.  
5. **Architecture limit (Case A):** Outputs are Cout=3 (`tp`,`msl`,`2t`) with **no** 3→65 decoder ⇒ **free-run impossible**. Precipitation on this head collapsed and is **out-of-scope**.

**One-sentence takeaway:** Mvula v5 is a successful **compression + short-range AF t2m + laptop inference** demonstration — not a finished 10-day free-running laptop AIFS.

---

## 2. Motivation (Dueben-style problem framing)

Modern AI NWP systems (AIFS / GraphCast-class) deliver strong forecast skill but typically assume:

- large GPU memory,
- complex software stacks,
- institutional compute budgets.

For many African research groups and meteorological services, the barrier is not only “is there a model?” but **can we run, verify, and iterate locally?**

Mvula therefore treats **computational accessibility** as a first-class evaluation axis alongside RMSE/ACC — in the spirit of ECMWF AI weather scorecards that report skill *and* cost/complexity honestly (Dueben et al. programme culture).

---

## 3. Methods

### 3.1 Models

| Component | Description |
|-----------|-------------|
| **Teacher (K1)** | Track A pruned Anemoi GraphTransformer (1/16 attention heads + recovery), accepted distillation teacher |
| **Student (v5)** | InceptionNeXt-style CNN; **Cin=65** (t/u/v/q/z × 13 levels, 1°) → **Cout=3** (`tp`, `msl`, `2t`) |
| **Training** | Distillation cache vs K1 + ERA5 targets (`teacher_k1_cache_full2020_2021.zarr`); Africa-weighted loss; β/γ feature/spectral terms off in late recipes |

### 3.2 Case A protocol (critical)

Because Cout=3 cannot update the next 65-channel atmospheric state:

- **Free-running multi-step rollout is not evaluated** (and is not technically supported).  
- **Analysis-forced (AF)** scoring is used: for lead \(L \in \{6,12,18,24\}\) h,

  1. Build ERA5 IC at time \(t_0 + (L-6)\) h,  
  2. Run **one** student +6 h step,  
  3. Score at valid time \(t_0 + L\).

Multi-lead tables are therefore **re-IC’d one-step skill**, not autonomous day-ahead rollouts. See `TRACKB_STATE_CLOSURE.md`.

### 3.3 Domain, truth, baselines

| Item | Choice |
|------|--------|
| Domain | Africa box (lat [−40, 40], lon [−20, 55] in packaged campaign) |
| Truth / ICs | Public ARCO ERA5 |
| Metrics | Cosine-latitude **RMSE**, anomaly **ACC**, mean **bias** (student − ERA5) |
| Teacher compare | Same metrics vs ERA5; degradation % vs K1 where K1 NetCDFs exist (**n=3** Jan-2023 weekly) |
| Inits | **61** × 00Z dates in 2023 across DJF/MAM/JJA/SON (`step_days=5`) |

### 3.4 Laptop benchmark protocol

- Device: CPU only; OMP/MKL/torch threads = 4  
- Timing: synthetic IC forward passes (IC fetch/build **excluded**)  
- Hardware: 11th Gen Intel Core i7-11800H, ~32 GB RAM (Windows)

---

## 4. Results — compression & compute

**Table 1. Footprint and laptop inference (v5).**

| Metric | Student v5 | K1 teacher |
|--------|------------|------------|
| Disk size | 8.8 MiB | 52.8 MiB (~6.0×) |
| Parameters | 2.169 M | (Anemoi GT) |
| Mean +6 h step (CPU) | 2.53 s | — |
| AF 4-lead infer-only (est.) | ~10.1 s | — |
| Peak RSS | ~1.19 GiB | — |
| GPU required | No | Typically yes |
| Free-run 10-day | Not supported | Supported |

**Interpretation:** The student head is laptop-plausible for short AF packages. End-to-end operational cost is dominated by **IC construction**, not the forward pass.

---

## 5. Results — African AF t2m skill (v5)

### 5.1 Lead-time scorecard

**Table 2. Africa t2m, all inits (n=61).**

| Lead | RMSE (K) | ACC | Bias (K) | K1 RMSE | Deg. vs K1 |
|-----:|---------:|----:|---------:|--------:|-----------:|
| +6 h | **1.382** | **0.965** | −0.115 | 1.216 | +16.6% |
| +12 h | **7.800** | 0.446 | **−5.330** | 1.698 | +360% |
| +18 h | 4.377 | 0.749 | −1.115 | 1.468 | +204% |
| +24 h | **3.231** | 0.835 | +1.299 | 1.467 | +103% |

**Table 3. Seasonal +6 h / +24 h (student vs ERA5).**

| Season | +6 RMSE | +6 ACC | +6 bias | +24 RMSE | +24 ACC | +24 bias |
|--------|--------:|-------:|--------:|---------:|--------:|---------:|
| DJF | 1.409 | 0.978 | +0.002 | 2.973 | 0.903 | +1.086 |
| MAM | 1.376 | 0.957 | −0.056 | 3.371 | 0.780 | +1.483 |
| JJA | 1.327 | 0.977 | −0.169 | 3.280 | 0.878 | +1.213 |
| SON | 1.415 | 0.949 | −0.246 | 3.318 | 0.777 | +1.428 |

**Findings**

- **+6 h is the scientific success region:** low bias, high ACC, stable across seasons (non-DJF mean RMSE ≈ 1.37).  
- **Skill collapses by midday lead (+12 h):** universal cold bias (−5.3 K) — treat as a **process/diurnal failure mode**, not a small RMSE gap.  
- **+24 h** partially recovers vs +12 h but remains far from K1; warm bias appears regionally.

### 5.2 Spatial structure (existing figures)

Already packaged under `reports/figures/`:

- `trackb_t2m_v5_rmse_L006h.png`, `…_bias_L006h.png`  
- `trackb_t2m_v5_rmse_L024h.png`, `…_bias_L024h.png`  
- **`mvula_fig02_t2m_lead_curves.png`** — RMSE/ACC vs lead (±1 std across inits; K1 overlay)  
- **`mvula_fig03_t2m_init_lead_heatmap.png`** — init × lead RMSE scorecard (shows +12 h ridge)  

Regenerate Fig. 2–3: `python scripts/plot_mvula_report_fig02_fig03.py`  

**Qualitative read**

- **+6 h bias:** near-zero continental mean; residuals at coasts / orography.  
- **+24 h bias:** Sahara cold / eastern–southern Africa warm dipole — systematic regional error.

---

## 6. Limitations (must stay in any upgrade to a paper)

1. **AF ≠ free-run.** Do not plot these leads as a 24 h autonomous forecast curve without stating re-IC.  
2. **tp failed** (dry collapse) — out-of-scope for this head.  
3. **K1 comparison n=3** for degradation % — student-vs-ERA5 is the statistically broader result.  
4. **Laptop timing excludes IC I/O.**  
5. **No extremes gate, no station verification, no SAWS ops pilot yet.**  
6. **v6-lite continue-train** (`v5_t2mRMSE`) trained; **smoke eval blocked** on Cassava by missing NetCDF/h5netcdf deps when reading teacher files — gate pending (§9).

---

## 7. Figure plan for this internal report → paper

Produce / assemble these as a fixed figure set (Dueben scorecard DNA + science maps).

| ID | Figure | Caption (draft) | Status |
|----|--------|-----------------|--------|
| **Fig. 1** | Pipeline schematic | *Mvula pathway: K1 teacher → distilled Cout=3 student → analysis-forced African t2m evaluation. Free-run is architecturally excluded (Case A).* | To draw |
| **Fig. 2** | Lead curves | *Cosine-latitude RMSE and ACC for African t2m versus lead time for student v5 (n=61). K1 shown where available (n=3). Error bars: ±1 std across inits.* | **Done** — `figures/mvula_fig02_t2m_lead_curves.png` |
| **Fig. 3** | Scorecard heatmap | *Init × lead RMSE (K) for African t2m (student v5). Rows ordered by season. Highlights the +12 h cold-bias ridge.* | **Done** — `figures/mvula_fig03_t2m_init_lead_heatmap.png` |
| **Fig. 4** | Spatial +6 h | *Mean bias and RMSE at +6 h (n=61). Near-unbiased large-scale field with coastal/orographic residuals.* | **Exists** (bias/rmse maps) |
| **Fig. 5** | Spatial +24 h | *Mean bias and RMSE at +24 h (n=61). Regional warm/cold dipole emerges.* | **Exists** |
| **Fig. 6** | +12 h pathology | *Histogram of +12 h bias across 61 inits (all cold) plus seasonal boxplots.* | **Done** — `figures/mvula_fig06_t2m_plus12h_pathology.png` |
| **Fig. 7** | Seasonal small multiples | *+6 h and +24 h RMSE/ACC by DJF/MAM/JJA/SON.* | Table ready → figure |
| **Fig. 8** | Compute panel | *Disk size, CPU step time, peak RSS on measured i7 laptop (student vs K1 size).* | **Done** — `figures/mvula_fig08_compute_panel.png` |

**Optional Fig. 9 (only if v6 accepted):** v5 vs v6 lead curves + ΔRMSE maps at +6 h.

---

## 8. Implications (careful)

**For research accessibility:** A ~9 MiB student that runs on CPU in seconds enables local AF t2m experimentation without continuous large-GPU dependence.

**For SAWS / NMHS interest:** Configure only as an **experimental +6 h t2m guidance / research tool** fed by analysis ICs — not as rainfall, storm, or 10-day production NWP.

**For finance / coastal conferences:** Position as **enabling climate-information infrastructure**, not as a coastal-economy forecast product.

---

## 9. Ongoing: v6-lite path

| Step | Status |
|------|--------|
| Locked accept: +6 h RMSE ≤ 1.28, ACC ≥ 0.96, +24 h RMSE ≤ 3.55 | Set |
| Continue-train → `student_global_stable_v5_t2mRMSE.ckpt` | **Done** (25 epochs) |
| Smoke AF eval (20 inits × 6/12/24) | **Done** — see `TRACKB_T2M_T2MRMSE_SMOKE.*` |
| Promote to `student_global_stable_v6.ckpt` | **REJECT** — keep v5 freeze (`TRACKB_V6_T2M_GATE.json`) |

**Smoke gate (2026-08-30):** +6 h RMSE **1.377** (need ≤1.28), ACC **0.967** (pass), +24 h RMSE **3.343** (pass ≤3.55). +6 h RMSE failed the bar (≈ flat vs v5 full 1.382). **All report/paper numbers stay on v5.**

---

## 10. Upgrade path: internal report → paper

| Stage | Output | When |
|-------|--------|------|
| **A. This document** | Internal Dueben-style tech report | Now |
| **B. Figure pack** | Figs 1–8 as PDF/PNG set + captions | 3–7 days |
| **C. Methods freeze** | Cite tag `trackb-v5-c4e` + JSON hashes | With B |
| **D. Preprint draft** | ~4–6 figures, 3–4k words, GMD/AIES-style | After B (+ optional v6) |
| **E. External verify** | Optional SAWS stations / persistence baseline | Stretch |

**Working paper title (when upgrading):**  
*Making AIFS-derived weather AI accessible for African short-range temperature forecasting: compression, analysis-forced verification, and laptop-scale inference*

**Suggested abstract spine (for later):** accessibility barrier → Case A student → AF protocol → +6 h skill / +12 h failure → laptop compute → honest limits.

---

## 11. Reproducibility pointers

```text
git checkout trackb-v5-c4e
# results (no re-run): reports/TRACKB_T2M_EXPANDED.md|.json
# figures: reports/figures/trackb_t2m_v5_{rmse,bias}_L{006,024}h.png
# laptop: reports/MVULA_LAPTOP_BENCHMARK.md
# re-run AF eval: scripts/run_trackB_t2m_expanded_cassava.sh
# laptop bench: scripts/bench_mvula_laptop_v5.py
```

Machine-readable aggregates: `reports/TRACKB_T2M_EXPANDED.json` (`student_results`, n=244 rows).

---

## 12. Decision log (team)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary metric | AF African t2m | Only defensible skill story on Cout=3 |
| Free-run | Out of scope | Case A |
| tp | Out of scope | Dry collapse |
| Report before paper | Yes | Lock figures/claims on v5 first |
| v6 promotion | Gate-only | Avoid diluting freeze narrative |

---

*End of internal report draft.* Figs 2–3, 6, and 8 generated. v6-lite smoke gate **REJECT** — paper numbers remain on frozen v5 (`trackb-v5-c4e`).

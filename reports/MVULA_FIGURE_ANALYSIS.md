# Mvula v5 — comprehensive figure analysis & interpretation

**Purpose:** Deep annotations for paper captions, mentor talks, posters, and Q&A.  
**Companion (short talking points):** [`MVULA_FIGURE_ANNOTATION_GUIDE.md`](MVULA_FIGURE_ANNOTATION_GUIDE.md)  
**Primary numbers:** [`TRACKB_T2M_EXPANDED.md`](TRACKB_T2M_EXPANDED.md) · XAI: [`MVULA_XAI_ATTRIBUTION.md`](MVULA_XAI_ATTRIBUTION.md)

---

## 0. Shared experimental frame (read first)

Every skill/spatial figure below (Figs. 2–8, 10–11d) is governed by the same contract. Misreading the protocol is the main way these plots get over-claimed.

| Item | Definition |
|------|------------|
| **Variable** | 2 m temperature (`2t` / t2m) over the Africa eval box |
| **Protocol** | **Analysis-forced (AF)** only: IC = ERA5 at `init + (L − 6) h`, then **one** student +6 h step → valid at lead **L** |
| **Leads** | L ∈ {6, 12, 18, 24} h |
| **Inits** | **n = 61** × 00Z, 2023, balanced across DJF / MAM / JJA / SON |
| **Truth / IC** | Public **ARCO ERA5** |
| **Metrics** | Cosine-latitude-weighted **RMSE**, **ACC**, **bias** (student − ERA5) |
| **Baselines** | AF persistence (`T(valid)=T(IC)`); **K1** teacher where available (**n = 3** overlapping inits) |
| **Architecture** | Case A: **Cin=65 → Cout=3** (`tp`, `msl`, `2t`); **no** free-run / 10-day rollout |

**What AF implies for interpretation**

- You are **not** watching error accumulate in a free-running forecast. Each lead is a **fresh IC + one step**.  
- Therefore a spike at +12 h is a **lead-/IC-conditioned** failure (often diurnal), not “chaos after 12 h of integration.”  
- For typical **00Z** inits: +6 h uses ~**00Z** IC; +12 h uses ~**06Z** IC; +18 h ~**12Z**; +24 h ~**18Z**. That mapping is central to Fig. 6 / Fig. 11.

**Claim boundary (all figures)**

| Supported | Not supported |
|-----------|----------------|
| Compression + laptop CPU inference | Free-run / Cout=65 / 10-day NWP |
| Useful **+6 h** African AF t2m | Operational day-ahead without caveats |
| Documented **+12 h** cold-bias pathology | tp skill; u/v wind outputs |
| Partial +24 h recovery vs +12 h | “Mvula matches K1 through 24 h” (K1 n=3; large gap remains) |

---

## Cross-cutting synthesis

### What the figure suite collectively shows

1. **Accessibility is real** (Fig. 9; Fig. 1D): ~**8.8 MiB**, ~**2.5 s**/+6 h step on a consumer CPU, ~**1.2 GiB** RSS.  
2. **Short-range skill is real** (Figs. 2, 7, 8): +6 h RMSE **≈1.38 °C**, ACC **≈0.97**, **+41.8%** vs AF persistence.  
3. **Failure is systematic, not anecdotal** (Figs. 3, 6): +12 h RMSE **≈7.80 °C**, mean bias **−5.33 °C**, **61/61** inits cold, all seasons.  
4. **Recovery is incomplete** (Figs. 2, 7, 8): +24 h RMSE **≈3.23 °C** (still ~**+134%** vs +6 h; ~**+103%** vs K1 RMSE on overlapping inits).  
5. **Errors have geography** (Figs. 4, 5, 10): mild topographic/arid structure at +6 h; strong NW-cool / SE–Arabia-warm dipole by +24 h.  
6. **The network is a near-surface thermodynamic mapper** (Fig. 11): sensitivity dominated by **t** (esp. **t1000**), then **z**, **q**; **u/v** weak for African-mean 2t.

### Scientific narrative arc

```
Compression succeeds (Fig. 9)
        ↓
+6 h AF t2m is useful (Figs. 2, 8, 4)
        ↓
+12 h exposes a coherent diurnal/lead pathology (Figs. 3, 6, 8, 11d)
        ↓
+18/+24 h partially heal but leave organised spatial bias (Figs. 2, 5, 10)
        ↓
Honest product: edge short-range African t2m demonstrator — not ops NWP
```

---

## Figure 1 — Pipeline overview

**File:** `figures/mvula_fig01_pipeline_publication.png`

### What it shows

Four-stage flowchart: **LEARN** (K1 teacher) → **COMPRESS** (Mvula v5 student) → **VERIFY** (AF African t2m) → **ACCESS** (laptop CPU). Subtitle states Case A (**Cout=3**; free-run not supported) and the driving question: *Can an AIFS-derived model be compressed while retaining useful African short-range t2m skill?*

### Panel-by-panel interpretation

| Block | Content | Interpretation |
|-------|---------|----------------|
| **A LEARN** | K1 pruned Anemoi GraphTransformer; ~52.8 MiB; full-state capable | Teacher = compressed AIFS-lineage GraphTransformer, not GraphCast |
| **B COMPRESS** | InceptionNeXt CNN; 65→3 (`tp`,`msl`,`2t`); 8.8 MiB; 2.17 M params; Africa-weighted mix | Distillation MVP; **no** 3→65 decoder → free-run impossible by design |
| **C VERIFY** | AF protocol; n=61; RMSE/ACC/bias; persistence + K1 | Headline: +6 h beats persistence; +12/+18 failure regime |
| **D ACCESS** | ~2.53 s/step CPU; ~1.2 GiB RSS; no GPU; “not ops NWP” | Accessibility is a first-class result, not an afterthought |

### How to use it

- Open talks / paper intro.  
- Anchor **architecture honesty** before showing skill plots.  
- Separates Mvula from Chimwemwe’s GraphCast scorecards (different pathway).

### Limitations / misreads

- Does not itself prove skill — that is Figs. 2–8.  
- “AIFS-derived” ≠ “AIFS emulator.”  
- Cout=3 includes tp/msl, but **claimed** verification here is t2m.

**Caption (paper-ready):** *Overview of the Mvula compressed student pathway: distillation of an AIFS-derived GraphTransformer teacher (K1) into an analysis-forced African 2 m temperature student (Case A: Cout = 3; free-run not supported).*

---

## Figure 2 — Lead-time RMSE and ACC

**File:** `figures/mvula_fig02_t2m_lead_curves.png`

### What it shows

Two curves vs lead {6,12,18,24} h: (left) African t2m RMSE; (right) ACC. Student v5 (n=61, ±1 σ shading) vs K1 teacher (n=3).

### Quantitative reading

| Lead | Student RMSE | ACC | Bias | Nature of point |
|-----:|-------------:|----:|-----:|-----------------|
| +6 | 1.38 °C | 0.97 | −0.12 | Near-K1 skill regime |
| +12 | 7.80 °C | 0.45 | −5.33 | Collapse / pathology |
| +18 | 4.38 °C | 0.75 | −1.12 | Partial recovery |
| +24 | 3.23 °C | 0.84 | +1.30 | Incomplete recovery; warm mean bias |

K1 stays roughly flat (~1.2–1.7 °C RMSE; ACC ~0.96–0.98) on its small sample.

### Interpretation

1. **Non-monotonic skill** — If this were free-run chaos growth, error would usually rise smoothly. The **peak at +12 h** then drop at +18/+24 h is the signature of **AF lead/IC conditioning** (different analysis times), consistent with Figs. 6 and 8.  
2. **+6 h validates the distillation hypothesis** for short-range African t2m: large compression need not destroy near-term skill.  
3. **+12 h is the scientific negative result** that must stay in the narrative (draft paper §4.3).  
4. **+24 h is not “fixed”** — RMSE still ≈2.3× +6 h; ACC 0.84 vs 0.97; vs K1 still roughly doubled error on overlapping inits.

### Communication

- **Mentor:** “Lead-conditioned skill, not a smooth 24 h forecast.”  
- **Avoid:** “Skillful through 24 h” without the +12 h clause.

**Caveats:** K1 n=3 → degradation % is directional, not a powered test. Shaded ±1 σ is across inits, not model uncertainty.

**Caption:** *Analysis-forced African t2m RMSE and ACC versus lead for Mvula student v5 (n = 61) and K1 teacher (n = 3). Error bars/bands show ±1 standard deviation across initialisations.*

---

## Figure 3 — Init × lead RMSE heatmap

**File:** `figures/mvula_fig03_t2m_init_lead_heatmap.png`

### What it shows

Matrix: rows = 61 initialisation dates (season-blocked DJF→SON); columns = leads +6/+12/+18/+24; colour = RMSE (°C), ~0 (blue) to ~8 (red).

### Interpretation

1. **Vertical red band at +12 h** — Failure is **campaign-wide**, not a seasonal subset or a few storm cases.  
2. **Column structure** — +6 h cool blues (~1–2 °C); +18 h pale (~4–5 °C); +24 h mid blues (~3 °C). Confirms Fig. 2’s non-monotonic pattern **per init**.  
3. **Season labels on the y-axis** — Same red +12 h column through DJF/MAM/JJA/SON → not a single-hemisphere winter artefact (aligns with Fig. 7).

### Why this figure matters

Heatmaps are the best antidote to “maybe the mean was skewed by outliers.” Here the mean **is** the story: essentially every row fails at +12 h.

**Caption:** *Init × lead African t2m RMSE scorecard for student v5 (n = 61). The bright column at +12 h corresponds to a systematic cold-bias failure mode (campaign-mean RMSE ≈ 7.8 °C).*

---

## Figure 4 — Spatial bias & RMSE at +6 h

**File:** `figures/mvula_fig04_t2m_spatial_plus6h.png`

### What it shows

Africa maps (≈40°N–40°S, 20°W–50°E+): (a) mean bias; (b) RMSE; mean over 61 inits at +6 h. Bias scale ≈±3 °C; RMSE ≈0–3.5 °C.

### Spatial interpretation

| Pattern | Where | Likely reading |
|---------|-------|----------------|
| Cold bias | Highlands (Ethiopia, Atlas, Rift) | Orography / lapse-rate / unresolved terrain — common t2m issue |
| Warm bias | Parts of Sahara, southern arid belts, Arabia | Arid / high diurnal-range regimes harder under AF |
| Low RMSE | Congo Basin / much of West–Central Africa | More moderate diurnal amplitude; errors look “easiest” |
| High RMSE | Horn, southern tip, Red Sea–Arabia | Co-located with larger \|bias\| → **bias-dominated** RMSE |

### Interpretation

At +6 h the student is **not spatially perfect**, but errors are **moderate and structured**. Domain-mean skill (Fig. 2) is not hiding a continent-sized disaster; residual issues look like **terrain and arid-zone** difficulty.

**Caption:** *Spatial mean bias and RMSE of African t2m at +6 h for student v5 (61 initialisations, 2023).*

---

## Figure 5 — Spatial bias & RMSE at +24 h

**File:** `figures/mvula_fig05_t2m_spatial_plus24h.png`

### What it shows

Same layout as Fig. 4 at +24 h, with **wider** colour ranges (bias ≈±8 °C; RMSE up to ~8 °C).

### Spatial interpretation

| Pattern | Where | Reading |
|---------|-------|---------|
| Cold bias | NW Africa / Sahara / western margins | Strengthened cool lobe |
| Warm bias | Horn–East Africa, Southern Africa, Arabian Peninsula | Strong warm lobe; Arabia often at colour-bar ceiling |
| Low RMSE | Central Africa | Remains relatively forgiving |
| High RMSE | Aligns with warm-bias regions | Systematic bias drives error magnitude |

### Interpretation

1. **Organised dipole** — Not random speckles; a coherent NW-cool / SE–Arabia-warm structure.  
2. **Lead dependence** — Comparing Fig. 4→5 (or Fig. 10 shared scales) shows **growth and reorganisation** of bias under AF 18Z-class ICs for 00Z inits, not mere amplification of the +6 h map.  
3. **Regional honesty** — Any “Africa-mean +24 h ACC = 0.84” claim should be paired with “large regional biases remain.”

**Caption:** *Same as Figure 4 at +24 h. Bias magnitude and RMSE increase, with a coherent north-west cool / south-east–Arabia warm pattern.*

---

## Figure 10 — Publication spatial quad

**File:** `figures/mvula_fig10_t2m_spatial_6h_24h_quad.png`

### What it shows

2×2 with **shared colour scales**: (a–b) mean bias +6/+24 h; (c–d) RMSE +6/+24 h.

### Interpretation

Same science as Figs. 4–5, but **fair visual comparison** (shared scales prevent +6 h looking artificially “noisy” or +24 h looking artificially calm). Prefer **Fig. 10** for talks/paper; keep 4–5 for supplements if needed.

**Caption:** *Four-panel spatial summary of African t2m verification for student v5: mean bias at +6 h and +24 h (a–b) and RMSE at +6 h and +24 h (c–d), using shared colour scales.*

---

## Figure 6 — +12 h cold-bias pathology

**File:** `figures/mvula_fig06_t2m_plus12h_pathology.png`

### What it shows

Left: histogram of +12 h bias across 61 inits (all negative; mean −5.33 °C). Right: seasonal boxplots (DJF/MAM/JJA/SON), all tightly clustered near −5 to −6 °C.

### Interpretation

1. **Pathology definition** — Tight, universal cold bias ⇒ **systematic**, not heavy-tailed noise.  
2. **Seasonally invariant** — SON slightly less cold, but still ~−5 °C; rules out “one bad season.”  
3. **Protocol link** — Under AF, +12 h for 00Z inits is tied to **06Z** ERA5 IC. The failure aligns with a **morning/diurnal** analysis time, matching the draft-paper discussion.  
4. **Ties to Fig. 8** — Persistence also struggles at midday-ish leads, but the student is **worse** than persistence at +12/+18 h → the network actively hurts relative to “copy the IC.”

### Scientific importance

This is the figure that makes the negative result **publishable and trustworthy**: quantified, complete sample coverage, seasonal confirmation.

**Caption:** *+12 h African t2m cold-bias pathology for student v5. All 61 initialisations are cold (bias ∈ [−5.76, −4.91] °C); mean RMSE = 7.80 °C. The pattern is consistent with a diurnal / lead-conditioned analysis-forced failure rather than random scatter.*

---

## Figure 7 — Seasonal skill at +6 h vs +24 h

**File:** `figures/mvula_fig07_t2m_seasonal.png`

### What it shows

Grouped bars by season: RMSE and ACC at +6 h vs +24 h (±1 σ across inits).

### Quantitative reading

| Season | n | +6 RMSE | +24 RMSE | Growth | +6 ACC | +24 ACC |
|--------|--:|--------:|---------:|-------:|-------:|--------:|
| DJF | 16 | 1.41 | 2.97 | +111% | 0.978 | 0.903 |
| MAM | 15 | 1.38 | 3.37 | +145% | 0.957 | 0.780 |
| JJA | 15 | 1.33 | 3.28 | +147% | 0.977 | 0.878 |
| SON | 15 | 1.42 | 3.32 | +134% | 0.949 | 0.777 |

### Interpretation

1. **+6 h is seasonally robust** — RMSE stays ~1.33–1.42 °C; ACC ≥0.95. The headline +6 h result is not a DJF fluke.  
2. **+24 h degradation is universal** — All seasons >+100% RMSE growth.  
3. **Transition seasons hardest at +24 h** — MAM/SON ACC ≈0.78, worse than DJF/JJA. Consistent with more variable shoulder-season regimes.  
4. **Does not show +12 h** — Pair with Figs. 3/6 when discussing pathology; Fig. 7 is specifically about **endpoint** (+6 vs +24) seasonal structure.

**Caption:** *Seasonal breakdown of analysis-forced African t2m RMSE and ACC for student v5 at +6 h and +24 h. RMSE increases by more than 100% from +6 h to +24 h in every season.*

---

## Figure 8 — Baselines vs persistence

**File:** `figures/mvula_fig08_t2m_baselines.png`

### What it shows

Left: RMSE of AF persistence, Mvula v5, and K1 vs lead. Right: relative skill  
\(1 - \mathrm{RMSE}_\mathrm{stu}/\mathrm{RMSE}_\mathrm{pers}\) in percent (positive = beats persistence).

### Quantitative reading

| Lead | Skill vs persistence | Regime |
|-----:|---------------------:|--------|
| +6 h | **+41.8%** | Useful extraction of state |
| +12 h | **−18.5%** | Failure (worse than copy-IC) |
| +18 h | **−18.6%** | Failure |
| +24 h | **+16.7%** | Partial recovery |

### Interpretation

1. **Persistence is the right null** under AF — because the IC is already analysis, “skill” means *improving on the analysis time’s temperature field toward the valid time*.  
2. **+6 h success is meaningful** — Beating persistence by ~42% is strong evidence of learned dynamics/adjustment, not memorisation of IC.  
3. **+12/+18 h are scientifically damning** — Below persistence means the student **adds error**. That is stronger than “high RMSE alone.”  
4. **K1 remains a ceiling** — Low flat RMSE; student never approaches it after +6 h on this plot.  
5. **Non-monotonic again** — Same story as Fig. 2, now in “utility vs null model” language preferred by verification audiences.

**Caption:** *Student skill relative to analysis-forced persistence. Positive relative skill indicates the student extracts useful information beyond copying the analysis initial condition.*

---

## Figure 9 — Laptop compute / accessibility

**File:** `figures/mvula_fig09_compute_panel.png`

### What it shows

Three bars: checkpoint size (8.8 vs 52.8 MiB); CPU +6 h step time (2.53 s, 4 threads); peak RSS (~1.16–1.2 GiB). Hardware: i7-11800H laptop; 2.17 M params; GPU not required.

### Interpretation

1. **Answers the Code for Earth accessibility question** independently of perfect 24 h skill.  
2. **~6× on-disk shrink** vs accepted K1 teacher — compression is measured, not rhetorical.  
3. **Seconds-scale inference** on CPU makes interactive demos / edge experiments plausible.  
4. **Memory ~1 GiB** — fits ordinary laptops; not a hidden “needs 64 GB” story.

### Caveats (always state)

- Timing **excludes** ERA5 IC fetch/build (often the real ops bottleneck).  
- Does not include training cost (Cassava/CHPC).  
- Does not imply NMHS operational readiness.

**Caption:** *Laptop compute and accessibility panel for Mvula student v5 on an Intel Core i7-11800H CPU (GPU not required). Timing excludes ERA5 initial-condition fetch and build.*

---

## Figure 11 — Explainability (XAI)

**File:** `figures/mvula_fig11_t2m_xai_attribution.png`  
**Detail:** [`MVULA_XAI_ATTRIBUTION.md`](MVULA_XAI_ATTRIBUTION.md)

### What it shows

(a) Variable-group sensitivity (sum over 13 levels): **t ≫ z > q ≫ u, v**.  
(b) Top channels — **t1000** dominates; then z925, q1000, etc.  
(c) Spatial mean |∂2t/∂x| over Africa (longitude-corrected crop).  
(d) Packaged AF RMSE/bias bars + text note on +12 h cold bias / 06Z IC.

### Interpretation

1. **Near-surface thermodynamics dominate African-mean 2t** — Physically plausible for a surface temperature head; supports “thermodynamic mapper” language.  
2. **Winds are weak drivers of this scalar** — Aligns with Sam’s u/v question: winds matter as **inputs to the 65-ch state**, but the head’s African-mean 2t sensitivity is not wind-led; **u/v are not outputs**.  
3. **Spatial |grad|** — Higher coastal / Sahel / Horn sensitivity vs darker Congo — attribution geography, not the same as RMSE geography (don’t conflate Fig. 11c with Fig. 4b).  
4. **Panel (d) binds XAI to the pathology** — Explainability without the skill collapse would be decorative; together they argue the failure is **diagnosable**.

### Method limits (must say)

- Saliency on **climate mean + noise IC**, not matched ARCO 00Z vs 06Z contrast yet.  
- Gradient magnitude ≠ causal proof.  
- Africa-mean target can under-weight regional wind-driven effects.

**Caption:** *Explainability for Mvula v5: input-group and channel saliency for African-mean 2 t, spatial sensitivity, and packaged analysis-forced skill highlighting the +12 h cold-bias failure. Saliency uses a climate-initial-condition probe and should be read as model sensitivity, not full physical causation.*

---

## Recommended figure packages

### Minimal talk (7 slides of figures)

1 → 9 → 2 → 8 → 3 or 6 → 10 → 11 (optional)

### Paper main body

1, 2, 6, 8, 9, 10, 11 — seasonal (7) and heatmap (3) as supplement or main if space allows.

### Mentor “honesty pack”

2 + 6 + 8 + 9 — skill, pathology, persistence bar, accessibility.

---

## Integrated limitations (apply to the whole suite)

1. **AF ≠ free-run** — Do not extrapolate these curves to multi-day rollout.  
2. **K1 n=3** — Teacher gap is indicative.  
3. **t2m-only claim** — tp out of scope; msl produced but not the freeze headline; no u/v outputs.  
4. **2023 inits** — No multi-year student scorecard in this freeze.  
5. **XAI probe IC** — Not yet lead-conditioned ARCO attribution.  
6. **Domain** — Africa box metrics; global maps are not the claim.

---

## Closing interpretation (one paragraph)

Taken together, the Mvula v5 figures show a **successful compression-and-accessibility result** coexisting with a **diagnosable short-range verification failure**. The student is small enough and fast enough to run African analysis-forced t2m experiments on a laptop; at +6 h it is skillful and clearly better than persistence; at +12 h it fails on every initialisation with a ~5 °C cold bias consistent with diurnal/lead-conditioned AF initial conditions; by +24 h skill partially returns but remains far from the teacher and carries organised regional biases. Explainability indicates a near-surface thermodynamic pathway for African-mean 2t. The scientific position is therefore: **edge-scale African short-range temperature demonstration with honest failure modes** — not a replacement for AIFS, GraphCast, or operational NWP.

---

*Expanded 2026-09-14 from packaged v5 artefacts under `reports/figures/`.*

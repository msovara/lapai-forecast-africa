# Mvula v5 — figure annotation & communication guide

**Audience:** mentors, team, talks, paper captions  
**Deep analysis:** [`MVULA_FIGURE_ANALYSIS.md`](MVULA_FIGURE_ANALYSIS.md)  
**Presentation (Mario framing):** [`MVULA_PRESENTATION_SLIDES.md`](MVULA_PRESENTATION_SLIDES.md) — headline **00Z one-step only**; multi-hour = IC-hour backup  
**Protocol:** analysis-forced **one-step** African **t2m** (not autoregressive lead skill); n=61 × 00Z inits; K1 n=3.  
**Claim boundary:** scoped edge t2m — not free-run, not 10-day, not precip skill.

---

## One-slide story (use this order)

1. **Fig. 9** — We made it small and laptop-runnable (~9 MiB, ~2.5 s/step CPU).  
2. **Fig. 2 + Fig. 8** — At **+6 h** it is useful and beats persistence; at **+12 h** it fails systematically.  
3. **Fig. 3 + Fig. 6** — Failure is **every init, every season** (cold bias ≈ −5.3 °C), not a bad case.  
4. **Fig. 10 / 4–5** — Spatial maps show moderate +6 h errors; organised regional bias by +24 h.  
5. **Fig. 11** — Network listens mainly to **near-surface T / moisture / height**; +12 h is diurnal/lead-conditioned under AF.  
6. **Close:** success = compression + honest short-range African t2m; next science = stabilize through 24 h.

**Do not say:** “Mvula works for 24 h forecasting” without the +12 h caveat.  
**Do say:** “Strong +6 h AF t2m on Africa; documented +12 h cold-bias pathology; partial recovery by +24 h.”

---

## Lead-time skill table (Fig. 2 numbers)

| Lead | Student RMSE | ACC | Bias | vs K1 RMSE | Soundbite |
|-----:|-------------:|----:|-----:|-----------:|-----------|
| +6 h | **1.38 °C** | **0.97** | −0.12 | +17% | Competitive short-range |
| +12 h | **7.80 °C** | **0.45** | **−5.33** | +360% | Systematic collapse |
| +18 h | 4.38 °C | 0.75 | −1.12 | +204% | Partial recovery |
| +24 h | **3.23 °C** | **0.84** | +1.30 | +103% | Better than +12, still ≫ K1 |

---

## Figure-by-figure annotations

### Fig. 1 — Pipeline (`mvula_fig01_pipeline.png` / `_publication.png`)

| | |
|--|--|
| **Shows** | Train 2020–21 → freeze Mvula v5 → AF one-step verify on 2023; columns = **00/06/12/18Z IC** |
| **Say** | “Architecture story: we distill a small student from a compressed AIFS-lineage teacher; every score is one +6 h step from a fresh analysis.” |
| **Don’t** | Call the four columns AR lead times, or imply GraphCast is in this pipeline. |

### Fig. 2 — Lead curves (`mvula_fig02_t2m_lead_curves.png`)

| | |
|--|--|
| **Shows** | RMSE ↑ and ACC ↓ vs lead; student (n=61) vs K1 (n=3) |
| **Read** | +6 h near teacher; **sharp spike/dip at +12 h**; partial recovery +18/+24 h |
| **Say** | “Skill is lead-conditioned, not smoothly degrading.” |
| **Caveat** | K1 n=3 — gap is real, sample unequal. |

### Fig. 3 — Init × lead heatmap (`mvula_fig03_t2m_init_lead_heatmap.png`)

| | |
|--|--|
| **Shows** | Every init’s RMSE at 6/12/18/24 h |
| **Read** | Solid **red column at +12 h** across DJF–SON |
| **Say** | “Not a few bad days — campaign-wide failure mode.” |

### Fig. 4 / 5 — Spatial +6 h / +24 h (`mvula_fig04_…`, `mvula_fig05_…`)

| | |
|--|--|
| **Shows** | Mean bias (left) and RMSE (right) over Africa |
| **+6 h** | Mild structured errors; RMSE mostly ~1–2 °C |
| **+24 h** | Stronger organised bias (cool NW Sahara / warm E–S Africa & Arabia); RMSE up |
| **Say** | “Errors are geographic, not uniform noise — useful for regional honesty.” |

### Fig. 10 — Quad spatial (`mvula_fig10_t2m_spatial_6h_24h_quad.png`)

| | |
|--|--|
| **Shows** | Same as 4–5 with **shared colour scales** (publication layout) |
| **Use** | Default spatial figure for talks/paper |
| **Say** | “(a–b) bias +6/+24; (c–d) RMSE +6/+24 — degradation is spatially organised.” |

### Fig. 6 — +12 h pathology (`mvula_fig06_t2m_plus12h_pathology.png`)

| | |
|--|--|
| **Shows** | Bias histogram + seasonal boxes |
| **Read** | **61/61 cold**; bias ∈ [−5.76, −4.91] °C; mean −5.33 °C; all seasons |
| **Say** | “Pathology, not variance. AF +12 h for 00Z inits uses 06Z IC → diurnal/lead-conditioned.” |

### Fig. 7 — Seasonal (`mvula_fig07_t2m_seasonal.png`)

| | |
|--|--|
| **Shows** | +6 vs +24 RMSE/ACC by DJF/MAM/JJA/SON |
| **Read** | +6 h strong all year (~1.3–1.4 °C); +24 h RMSE ≈ double+; ACC drop worst MAM/SON |
| **Say** | “+24 h degradation is cross-season, not a winter-only artefact.” |

### Fig. 8 — vs persistence (`mvula_fig08_t2m_baselines.png`)

| | |
|--|--|
| **Shows** | Persistence vs Mvula vs K1; relative skill % |
| **Read** | +6 h **+42%** vs persistence; +12/+18 h **worse** than persistence (−19%); +24 h recovers **+17%** |
| **Say** | “If you can’t beat persistence, you’re not extracting useful state — +12/+18 fail that test.” |

### Fig. 9 — Laptop compute (`mvula_fig09_compute_panel.png`)

| | |
|--|--|
| **Shows** | 8.8 vs 52.8 MiB; 2.53 s/+6 h step; ~1.2 GiB RSS |
| **Say** | “This is the accessibility claim: commodity CPU, no GPU required.” |
| **Caveat** | Timing **excludes** ERA5 IC fetch/build. |

### Fig. 11 — XAI (`mvula_fig11_t2m_xai_attribution.png`)

| | |
|--|--|
| **Shows** | Group sensitivity (t≫z>q≫u,v); top channel **t1000**; spatial |grad|; skill bars |
| **Say** | “Explainability: near-surface thermodynamics dominate African 2t; winds matter less for this head.” |
| **Don’t** | Call it causal physics proof — it’s gradient saliency on a climate+noise IC, not ARCO diurnal contrast yet. |

---

## Communication snippets

### Mentor / close-out (30 s)

> Mvula v5 is an ~9 MiB student that runs African analysis-forced t2m on a laptop CPU in ~2.5 s per step. At +6 h it is strong (RMSE ~1.4 °C, ACC ~0.97) and beats persistence. At +12 h it fails systematically with a ~5 °C cold bias on all 61 inits — we document that as a diurnal/lead-conditioned pathology, not a success. Partial recovery by +24 h still leaves a large gap to the teacher. We claim compression and short-range African t2m honesty, not 10-day free-run NWP.

### LinkedIn / public (careful)

> We compressed an AIFS-lineage weather model to ~9 MB so African short-range temperature experiments can run on a normal laptop. Results are open and claim-bounded: useful at +6 h under analysis forcing; a documented failure mode at +12 h; not a 10-day forecast system.

### Q&A ready answers

| Question | Answer |
|----------|--------|
| Does it work at 24 h? | Partially — better than +12 h, still ~2× RMSE vs +6 h and ≫ K1; don’t sell as operational day-ahead without Mvula-2 work. |
| Why +12 h? | Systematic cold bias under AF; for 00Z inits the IC is 06Z — points to diurnal/lead conditioning. Persistence also wins there. |
| What about wind/MSLP? | Student head has msl+tp+2t; freeze claim is **t2m**. u/v are inputs, not outputs. |
| vs GraphCast? | Different role: GraphCast = Africa reference bench; Mvula = edge compression demo. |

---

## Suggested talk/paper figure set (minimal)

| Slot | Figure | Why |
|------|--------|-----|
| Method | Fig. 1 | Pipeline |
| Headline skill | Fig. 2 | Lead curves |
| Failure evidence | Fig. 3 or 6 | Systematic +12 h |
| Spatial | Fig. 10 | Shared scales |
| Utility bar | Fig. 8 | vs persistence |
| Edge claim | Fig. 9 | Size/CPU |
| Optional depth | Fig. 11 | Explainability |

Full narrative: [`DRAFT_PAPER_MVULA_V5.md`](DRAFT_PAPER_MVULA_V5.md) · numbers: [`TRACKB_T2M_EXPANDED.md`](TRACKB_T2M_EXPANDED.md) · XAI: [`MVULA_XAI_ATTRIBUTION.md`](MVULA_XAI_ATTRIBUTION.md).

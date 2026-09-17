# Mvula v5 — presentation slide outline (Mario framing)

**Audience:** Code for Earth / mentors  
**Rule:** Headline = **analysis-forced one-step from 00Z IC** only. Multi-hour panels = **IC-hour sensitivity backup**, never “AR lead time.”

---

## Core deck (use these)

| # | Slide title | Figure / content |
|---|-------------|----------------|
| 1 | Title | Mvula: compressing AIFS-lineage weather AI for African edge use |
| 2 | What we built | **Fig. 1** (pipeline) — Case A Cout=3; AF one-step; no free-run |
| 3 | Accessibility | **Fig. 9** — 8.8 MiB · ~2.5 s/step CPU · ~1.2 GiB RSS |
| 4 | **Headline skill** | **Fig. 2 primary** `mvula_fig02_primary_00z_onestep.png` — RMSE≈1.38 °C, ACC≈0.97, bias≈−0.12; +41.8% vs persistence |
| 5 | Spatial (00Z only) | **Fig. 4** — one-step from **00Z** analysis IC |
| 6 | What the network uses | **Fig. 11 (a–c only)** — t / z / q; top channel t1000 *(skip or crop panel d on this slide)* |
| 7 | Claim boundary | One-step AF African t2m · not AR · not 10-day · not precip · Cout=3 |
| 8 | Thanks / next | Optional Mvula-2: broader IC hours or true AR+forcings if full state |

**Spoken line for slide 4:**  
“These are one-step forecasts from 00Z analysis — the model’s native +6 h step — not a multi-step rollout.”

---

## Backup only (if asked about “other leads”)

| # | Slide title | Figure |
|---|-------------|--------|
| B1 | IC-hour sensitivity (still one-step) | **Fig. 2 backup** curves — axes labelled **00Z / 06Z / 12Z / 18Z** |
| B2 | 06Z-IC pathology | **Fig. 6** and/or **Fig. 3** |
| B3 | vs persistence by IC hour | **Fig. 8** |
| B4 | 00Z vs 18Z spatial | **Fig. 10** (or Fig. 5) |
| B5 | Explainability + 06Z note | **Fig. 11** full |

**Spoken line for backup:**  
“Different columns are different analysis hours under the same one-step protocol — not autoregressive lead times. True AR needs the predicted full state plus forcings.”

---

## Do not show as “lead time”

- Old Fig. 2/3/7/8 titles that say “lead” without IC-hour rename (regenerated scripts fix this).  
- Any slide that implies free-run error growth from 6→24 h.

---

## File checklist after regenerate

| Role | File |
|------|------|
| Headline | `reports/figures/mvula_fig02_primary_00z_onestep.png` |
| Backup curves | `reports/figures/mvula_fig02_t2m_lead_curves.png` |
| Pipeline | `mvula_fig01_pipeline_publication.png` |
| Compute | `mvula_fig09_compute_panel.png` |
| Spatial 00Z | `mvula_fig04_t2m_spatial_plus6h.png` |
| XAI | `mvula_fig11_t2m_xai_attribution.png` |

Regenerate:  
`python scripts/plot_mvula_report_fig02_fig03.py`  
`python evaluation/plot_africa_spatial.py`  
`python scripts/plot_mvula_xai_attribution.py`

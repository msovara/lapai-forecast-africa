# Team note — Track A Step A1 passed

**From:** Mthetho Sovara  
**Date:** 6 August 2026  
**Re:** LapAI Track A coarsening gate (A1) closed on Lengau

---

## Headline

**Track A Step A1 (grid coarsening) has passed** the acceptance gate: **30/30** checks on t2m, u10, v10 at +24 h and +48 h (≤5% RMSE degradation vs Phase 0 teacher, Africa domain).

In practice the coarsened O96 student is **better** than the Phase 0 N320 teacher on these short leads — e.g. t2m +24 h RMSE is ~25–33% lower across all five January 2023 weekly inits.

## What we ran

1. Full fine-tune on `era5_n96_2020_2021.zarr` (2000 steps) — winds OK; **t2m +24 h failed** (cold bias).
2. Extend fine-tune **+15 000 steps** from that checkpoint (~1 h on a 32 GiB V100).
3. Re-forecast five inits → score vs GCS ERA5 → gate.

**Checkpoint (Lengau):** `models/teacher_coarsened.ckpt` →  
`trackA_coarsen_full_extend_runs/.../b828d680-…/inference-last.ckpt`

**Artefacts:**  
`reports/TRACKA_A1_GATE.json` · `reports/TRACKA_COARSEN_SCORECARD.json` ·  
`reports/LAPAI_TRACKA_TEAM_REPORT.md` · `reports/TRACKA_FULL_TRAIN_CHECKLIST.md`

## Notes

- **`tp`** is on the scorecard for reporting but **not** in the A1 gate (by design). It enters the **A2** CRPS gate.
- Lengau ICs still use the offline N320→O96 **grid bridge**; production alignment with Mario’s O96-at-fetch remains open.
- Smoke (50-step) gate failure was expected; this full/extend run is the fair A1 attempt.

## Proposed next steps

1. Start **A2** (attention-head pruning).
2. Confirm with Oxford / Mario: **2024–2025 init calendar** and canonical **O96 IC** path.
3. Optional: 6h-accumulate ERA5 precip before treating `tp` RMSE as primary.

Happy to walk through Streamlit / scorecard on a short call if useful.

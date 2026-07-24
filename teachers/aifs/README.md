# AIFS teacher pathway (LapAI Phase 0)

**Role:** Primary compression teacher — ECMWF AIFS / Anemoi `n320_gt6` on Lengau.

**Artefacts**
- Checkpoint: `models/teacher_n320_gt6/inference.ckpt`
- Forecasts: `data/processed/phase0/forecasts/*_00Z.nc`
- Scorecard: `reports/PHASE0_BASELINE_SCORECARD.json`

**Eval domain:** Africa `lat ∈ [-40, 40]`, `lon ∈ [-20, 70]` (`config/domains.yaml`).

**Note:** Phase 0 NetCDFs produced before 2026-07-24 were cropped to **55°E**.
Re-run Phase 0 forecast write + scorecard after rematerialising with the 70°E box
before comparing AIFS and GraphCast on an identical grid.

**Entry points (unchanged):** `scripts/run_phase0_forecast.py`, `pbs/phase0_baseline_lengau.pbs`,
`scripts/run_phase0_closure.py`.

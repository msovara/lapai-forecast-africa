# Mvula / Code for Earth — Phase 0 baseline sync (2026-06-29)

Share this note with the Mvula team when publishing Phase 0 artifacts.

## Summary

**Phase 0 is closed.** The C4E **`n320_gt6`** teacher runs on CHPC Lengau (V100), produces five Jan 2023 weekly forecasts on the **Africa eval grid**, and is scored against team ERA5 truth in **`gs://code4earth/era5`**.

| Item | Location |
|------|----------|
| Git repo | https://github.com/msovara/lapai-forecast-africa |
| Baseline scorecard | `reports/PHASE0_BASELINE_SCORECARD.json` (commit `491b6db`) |
| Team GCS copies | `gs://code4earth/lapai/PHASE0_BASELINE_SCORECARD.json`, `gs://code4earth/lapai/MVULA_PHASE0_SYNC.md` |
| Forecast NetCDFs | Lengau `~/repos/lapai-forecast/data/processed/phase0/forecasts/` |
| Runbook | `reports/PHASE0_CLOSURE.md` |

## Baseline skill (Africa domain)

Scores are area-weighted RMSE / ACC vs GCS ERA5 truth. Example **init 20230101**:

| Lead | t2m RMSE | t2m ACC | tp RMSE | u10 RMSE |
|------|----------|---------|---------|----------|
| +24h | 2.23 K | 0.93 | 0.00057 m | 3.38 m/s |
| +48h | 7.83 K | 0.45 | 0.00091 m | 3.26 m/s |
| +72h | 14.7 K | 0.04 | 0.0019 m | 4.19 m/s |

Full matrix: 5 inits × 6 leads (24, 48, 72, 120, 168, 240 h) × 4 variables in the scorecard JSON.

## Long-lead t2m (diagnostics only)

**+120h to +240h** t2m shows large RMSE growth and negative ACC on the Africa box. Investigation confirmed:

- Forecast valid times align with scoring (no pipeline bug)
- Fields evolve between 6-hour steps

We treat **+24h to +72h** as the primary comparison window for compression gates until long-lead behaviour is reviewed with the team. **Do not use +240h t2m alone as a pass/fail gate.**

## Reproduce scoring (any collaborator)

```bash
git clone https://github.com/msovara/lapai-forecast-africa.git
cd lapai-forecast-africa
conda activate lapai-anemoi
gcloud auth application-default login
python scripts/run_phase0_closure.py --score-only --forecast-dir data/processed/phase0/forecasts
```

Forecasts must be copied from Lengau or regenerated with `pbs/phase0_baseline_lengau.pbs` (offline CDS cache required on cluster).

## Track A next (compression)

Coarsening acceptance is wired to the Phase 0 scorecard at **+24h and +48h** for **t2m, u10, v10** (max **5% RMSE degradation**):

```bash
python scripts/run_trackA_gate.py --candidate reports/TRACKA_COARSEN_SCORECARD.json
```

Config: `configs/trackA_coarsen.yaml`.

## Open questions for Mvula

1. Confirm **+24h vs +48h** as the primary lead for teacher/student comparison in the challenge protocol.
2. Should long-lead (+120–240h) t2m be in scope for Phase 0 baseline, or deferred to a later validation pass?
3. Preferred artifact delivery: Git LFS, shared GCS prefix under `gs://code4earth/`, or CHPC rsync?

---

*Generated after Lengau job 7318486 (forecasts) and local GCS score-only run 2026-06-29.*

# LapAI student pathway (Track A)

**Role:** Compressed O96 / laptop-deployable student distilled from the AIFS teacher.

**Artefacts**
- Smoke config: `configs/trackA_coarsen.yaml` (50-step / 2018 Q1 Zarr)
- **Full config:** `configs/trackA_coarsen_full.yaml` (2000-step / 2020–2021 Zarr)
- PBS: `pbs/trackA.pbs` (smoke), `pbs/trackA_full.pbs` (full, 24h)
- Forecasts: `data/processed/trackA/forecasts/`
- Gate: `reports/TRACKA_A1_GATE.json`

**Full A1 sequence (Lengau)**
1. Rsync `data/processed/lapai/era5_n96_2020_2021.zarr/` + repo to Lengau  
2. `cd ~/repos/lapai-forecast && qsub pbs/trackA_full.pbs`  
3. After ckpt: `python scripts/run_trackA_closure.py` (forecast + score)  
4. Gate vs Phase 0: `python scripts/run_trackA_gate.py --candidate reports/TRACKA_COARSEN_SCORECARD.json`

**Teachers for comparison (same Africa scorecard)**
1. AIFS Phase 0 — `teachers/aifs`
2. GraphCast Africa — `teachers/graphcast` (GCS; lon to 70°E)

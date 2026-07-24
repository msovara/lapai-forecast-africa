# GraphCast Africa teacher pathway

**Role:** Secondary teacher / baseline for LapAI — Africa-cropped GraphCast forecasts
(avoids global 0.25° polar NaNs that affect global-grid scoring).

**Upstream**
- Code: [africlimate-research/graphcast-africa](https://github.com/africlimate-research/graphcast-africa)
  (clone/submodule under `vendor/graphcast-africa` for local runs)
- Forecasts: `gs://africlimate-ai-fcst-training/graphcast-africa/`
- Domain: `lat ∈ [-40, 40]`, `lon ∈ [-20, 70]` @ 0.25° (matches `config/domains.yaml`)

**1° path:** GraphCast_small (`--model small` in upstream repo) — owned by Sh / local
experiments; wire into LapAI scorecards once 1° Zarrs/NetCDFs are published.

**LapAI adapter:** `teachers.graphcast.gcs_forecasts` — open year/variable Zarrs and
select `(init, lead)` slices for the shared Africa scorecard.

```bash
# Dry-run planned inits/leads
python scripts/score_graphcast_africa.py --dry-run

# Score Jan 2022 weekly inits vs GCS ERA5 → reports/GRAPHCAST_AFRICA_SCORECARD.json
python scripts/score_graphcast_africa.py
```

```python
from teachers.graphcast import open_graphcast_forecast, select_init_lead

da = open_graphcast_forecast("t2m", 2020)
field = select_init_lead(da, init="2020-06-01", lead_hours=24)
```

**Do not** replace the AIFS Phase 0 teacher for Track A compression; use this pathway
for paired skill comparisons and NaN-safe Africa baselines.

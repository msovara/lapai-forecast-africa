# GraphCast Africa teacher pathway

**Role:** Teacher / baseline for LapAI — Africa-cropped GraphCast forecasts
(avoids global 0.25° polar NaNs that affect global-grid scoring).

**Interim policy (2026-07):** **GraphCast-first** while AIFS teacher NaN / rollout trust
is open — see [`reports/GRAPHCAST_FIRST.md`](../../reports/GRAPHCAST_FIRST.md)
(decision, metric contract, scorecard alignment checklist). AIFS Track A stays paused,
not deleted.

**Upstream**
- Code: [africlimate-research/graphcast-africa](https://github.com/africlimate-research/graphcast-africa)
  (clone/submodule under `vendor/graphcast-africa` for local runs)
- Forecasts: `gs://africlimate-ai-fcst-training/graphcast-africa/`
- Domain: `lat ∈ [-40, 40]`, `lon ∈ [-20, 70]` @ 0.25° (matches `config/domains.yaml`)

**1° path (active bench):** GraphCast Small (1°, mesh 2to5) — Chimwemwe / team CDS+ERA5
on-grid verification is the v0 common pipeline (`reports/GRAPHCAST_FIRST.md`).
Wire tables into `reports/GRAPHCAST_SMALL_1DEG_SCORECARD.json`.

**LapAI adapter (0.25° GCS):** `teachers.graphcast.gcs_forecasts` — open year/variable
Zarrs and select `(init, lead)` slices for the shared Africa scorecard.

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

**Do not** silently replace the AIFS compression teacher for Track A; use GraphCast for
near-term verification and paired skill comparisons until AIFS is cleared to resume.

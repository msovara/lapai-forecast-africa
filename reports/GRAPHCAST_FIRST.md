# GraphCast-first mode (interim)

**Status:** Active as of 2026-07-30  
**Decision:** Per Shruti — proceed with **GraphCast Africa** as the critical path until the AIFS teacher NaN / rollout-trust issue is resolved with Mario.  
**Scope:** Evaluation and benchmarking only. LapAI student distillation remains AIFS-based unless the team later decides otherwise.

## What this means

| Stream | Mode | Notes |
|--------|------|--------|
| **GraphCast Africa (1° Small)** | **Go** | Chimwemwe’s CDS/ERA5-on-1° bench is the interim common pipeline |
| **GraphCast Africa (0.25° GCS)** | Parallel when useful | Team GCS forecasts; scorecard year-overlap still a known gap |
| **AIFS Phase 0 / Track A** | **Paused** | Keep artefacts (Zarr, configs, PBS); do not burn GPU on full A1 until unblocked |
| **Shared Africa scorecard** | **Keep** | Same domain + metrics so AIFS can rejoin without forking the product |

## Domain (frozen)

From `config/domains.yaml` / Chimwemwe bench:

- **lat:** −40° … 40°  
- **lon:** −20° … 70°  

## Interim metric contract (Chimwemwe / GraphCast 1°)

- **Model:** GraphCast Small (1°, mesh 2to5)  
- **ICs + truth:** ERA5 via CDS, requested **on the 1° grid** (no extra local regrid)  
- **RMSE:** cosine-latitude weighted vs ERA5  
- **Skill:** \(1 - (\mathrm{RMSE}_\mathrm{forecast} / \mathrm{RMSE}_\mathrm{persistence})^2\)  
  - 1 = perfect · 0 = ties persistence · &lt; 0 = worse than persistence  
- **Reference init (v0):** 2023-01-01 12 UTC  
- **Leads:** report 6–48 h densely; T2M/MSLP through 240 h when available  
- **Precip:** 6 h accumulated; verification may stop at 48 h until CDS gap closed  

### Interpretation caveats

- **T2M skill dip near +30 h** is often a **persistence diurnal artefact** (persistence RMSE collapses); always plot RMSE and skill together.  
- **+6 h skill** may be undefined if persistence RMSE = 0 (same analysis state) — omit or redefine persistence from t−6 h.  
- **State units** in every table/JSON: T2M [K], MSLP [Pa or hPa — pick one], precip [m or mm — pick one].

## AIFS pause — do not delete

Ready to resume later:

- `data/processed/lapai/era5_n96_2020_2021.zarr`  
- `configs/trackA_coarsen_full.yaml`, `pbs/trackA_full.pbs`  
- Phase 0 scorecard / Track A smoke artefacts  

**Unblock criteria (any of):** Mario confirms AIFS NaN reproduction/fix; Shruti clears teacher for further rollouts; agreed short-lead-only AIFS use for Track A.

## Scorecard alignment checklist

Use this before calling a GraphCast run “LapAI-canonical.”

### A. Identity

- [ ] Model name + resolution recorded (`graphcast_small_1deg` vs `graphcast_africa_0p25`)
- [ ] Init date(s) + hour (UTC)
- [ ] Domain matches `config/domains.yaml` africa box
- [ ] Lead hours listed explicitly

### B. Data

- [ ] Forecast grid stated (1° or 0.25°)
- [ ] Truth source stated (CDS ERA5 on-grid / GCS ERA5 / other)
- [ ] No silent regrid between forecast and truth (or regrid method documented)
- [ ] Variable names mapped to LapAI aliases where needed (`2m_temperature`↔`t2m`, etc.)

### C. Metrics

- [ ] Cosine-latitude weights applied the same way
- [ ] Persistence definition written down
- [ ] Skill formula matches interim contract above
- [ ] RMSE + skill both exported (not skill alone)
- [ ] Units declared per variable

### D. Artefacts

- [ ] Tabular results (CSV or JSON)
- [ ] Optional plots: RMSE vs lead, skill vs lead, spatial error maps
- [ ] Scorecard JSON under `reports/` (see schema sketch below)
- [ ] Pointer from this run logged in team thread / commit message

### E. Multi-init readiness (next increment)

- [ ] ≥ 2 inits (prefer weekly Jan 2023 or multi-season)
- [ ] Precip verification path beyond 48 h identified (or explicitly deferred)
- [ ] Same JSON schema as AIFS Phase 0 where possible

## Scorecard JSON sketch

```json
{
  "pathway": "graphcast",
  "model": "graphcast_small_1deg",
  "domain": {"lat": [-40.0, 40.0], "lon": [-20.0, 70.0]},
  "init": "2023-01-01T12:00:00Z",
  "truth": "era5_cds_1deg",
  "metrics": {
    "rmse": "cosine_latitude_weighted",
    "skill": "1 - (rmse_fc / rmse_pers)^2"
  },
  "units": {
    "2m_temperature": "K",
    "mean_sea_level_pressure": "Pa",
    "total_precipitation_6hr": "m"
  },
  "rows": [
    {
      "variable": "2m_temperature",
      "lead_hours": 24,
      "rmse_forecast": 0.756,
      "rmse_persistence": 3.428,
      "skill_score": 0.951
    }
  ]
}
```

Canonical dump path (suggested): `reports/GRAPHCAST_SMALL_1DEG_SCORECARD.json`

## Repo hooks

| Piece | Path |
|-------|------|
| Domain | `config/domains.yaml` |
| GCS 0.25° reader | `teachers/graphcast/` |
| Score script (GCS path) | `scripts/score_graphcast_africa.py` |
| Pathway docs | `teachers/graphcast/README.md` |
| This policy | `reports/GRAPHCAST_FIRST.md` |

## Next actions (owners)

1. **Mthetho** — accept Chimwemwe method on the team thread; ingest first scorecard JSON.  
2. **Chimwemwe** — freeze units + multi-init list; close precip CDS gap when possible.  
3. **Shruti / Mario** — AIFS NaN resolution; signal when AIFS pipeline may resume.  
4. **Team** — keep one evaluation contract; dual teachers, one product.

# Track B — Expanded t2m analysis-forced evaluation (v5 freeze)

**Timestamp (UTC):** 2026-08-26T20:13:54.869279+00:00
**Checkpoint:** `models/student_global_stable_v5.ckpt`
**N inits:** 61 (seasons: DJF, JJA, MAM, SON)
**Leads:** [6, 12, 18, 24]
**K1 coverage:** 3 inits — ['20230101', '20230115', '20230129']

## Protocol

- Analysis-forced only (Case A / Cout=3): IC at `init+(L−6)h` → one +6h step.
- IC + truth: public ARCO ERA5 (no CDS).
- Gate variable: **t2m**. tp out-of-scope (dry-collapse).
- Student and K1 scored with the same cosine-latitude RMSE / ACC / bias vs ERA5.

## Lead-time table (t2m, all inits)

| lead | n | student RMSE | student ACC | student bias | K1 RMSE | K1 ACC | deg vs K1 |
|-----:|--:|-------------:|------------:|-------------:|--------:|-------:|----------:|
| 6 h | 61 | 1.382 | 0.965 | -0.115 | 1.216 | 0.983 | 16.6% |
| 12 h | 61 | 7.800 | 0.446 | -5.330 | 1.698 | 0.960 | 359.8% |
| 18 h | 61 | 4.377 | 0.749 | -1.115 | 1.468 | 0.969 | 204.0% |
| 24 h | 61 | 3.231 | 0.835 | 1.299 | 1.467 | 0.973 | 102.6% |

## Seasonal breakdown (student vs ERA5)

### DJF

| lead | n | RMSE | ACC | bias | deg vs K1 |
|-----:|--:|-----:|----:|-----:|----------:|
| 6 h | 16 | 1.409 | 0.978 | 0.002 | 16.6% |
| 12 h | 16 | 7.928 | 0.508 | -5.395 | 359.8% |
| 18 h | 16 | 4.492 | 0.780 | -1.317 | 204.0% |
| 24 h | 16 | 2.973 | 0.903 | 1.086 | 102.6% |

### JJA

| lead | n | RMSE | ACC | bias | deg vs K1 |
|-----:|--:|-----:|----:|-----:|----------:|
| 6 h | 15 | 1.327 | 0.977 | -0.169 | — |
| 12 h | 15 | 7.823 | 0.551 | -5.409 | — |
| 18 h | 15 | 4.466 | 0.822 | -1.449 | — |
| 24 h | 15 | 3.280 | 0.878 | 1.213 | — |

### MAM

| lead | n | RMSE | ACC | bias | deg vs K1 |
|-----:|--:|-----:|----:|-----:|----------:|
| 6 h | 15 | 1.376 | 0.957 | -0.056 | — |
| 12 h | 15 | 8.005 | 0.327 | -5.403 | — |
| 18 h | 15 | 4.624 | 0.650 | -1.027 | — |
| 24 h | 15 | 3.371 | 0.780 | 1.483 | — |

### SON

| lead | n | RMSE | ACC | bias | deg vs K1 |
|-----:|--:|-----:|----:|-----:|----------:|
| 6 h | 15 | 1.415 | 0.949 | -0.246 | — |
| 12 h | 15 | 7.436 | 0.394 | -5.107 | — |
| 18 h | 15 | 3.916 | 0.739 | -0.653 | — |
| 24 h | 15 | 3.318 | 0.777 | 1.428 | — |

## Critical questions

1. **Does +6h t2m stay strong outside January/DJF?** Yes — non-DJF mean RMSE=1.373 (overall +6h=1.382).
2. **How fast does skill deteriorate through 24h?** RMSE growth +6→+24h = 134%. Seasonal growth: {'DJF': 111.03061188440061, 'JJA': 147.1924470198042, 'MAM': 145.03078997895472, 'SON': 134.46218191175504}.
3. **Does +24h degradation vs K1 persist?** Yes (+24h deg=102.6%; K1 n=3). Across-season +24h RMSE growth persistence: True.

**Verdict summary:** +6h t2m RMSE=1.3820212192984673; +24h RMSE=3.2310266774095906 (+134% vs +6h). K1 deg +6h=16.59632263680999; +24h=102.56721622101622 (K1 n=3). 24h growth persists across seasons=True.

## tp (out of scope)

tp remains out-of-scope for Track B promotion. Samples:
```json
[
  {
    "init_date": "20230101",
    "season": "DJF",
    "lead_hours": 6,
    "student_tp_max_m": 0.0,
    "note": "tp out-of-scope; all-dry collapse if max\u22480"
  },
  {
    "init_date": "20230401",
    "season": "MAM",
    "lead_hours": 6,
    "student_tp_max_m": 0.0,
    "note": "tp out-of-scope; all-dry collapse if max\u22480"
  },
  {
    "init_date": "20230702",
    "season": "JJA",
    "lead_hours": 6,
    "student_tp_max_m": 0.0,
    "note": "tp out-of-scope; all-dry collapse if max\u22480"
  },
  {
    "init_date": "20231002",
    "season": "SON",
    "lead_hours": 6,
    "student_tp_max_m": 0.0,
    "note": "tp out-of-scope; all-dry collapse if max\u22480"
  }
]
```

## Figures

- `bias_L6h`: `reports/figures/trackb_t2m_v5_bias_L006h.png`
- `rmse_L6h`: `reports/figures/trackb_t2m_v5_rmse_L006h.png`
- `spatial_npz_L6h`: `reports/figures/trackb_t2m_v5_spatial_L006h.npz`
- `bias_L24h`: `reports/figures/trackb_t2m_v5_bias_L024h.png`
- `rmse_L24h`: `reports/figures/trackb_t2m_v5_rmse_L024h.png`
- `spatial_npz_L24h`: `reports/figures/trackb_t2m_v5_spatial_L024h.npz`

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=1 MKL_INTERFACE_LAYER=GNU \
  python -u evaluation/trackB_t2m_expanded.py \
    --ckpt models/student_global_stable_v5.ckpt --device cuda \
    --ic_source arco --step_days 5 --leads 6,12,18,24
```

See also `scripts/run_trackB_t2m_expanded_cassava.sh` and README Track B section.


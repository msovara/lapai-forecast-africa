# Track B — Expanded t2m analysis-forced evaluation (v5 freeze)

**Timestamp (UTC):** 2026-08-30T05:13:25.967231+00:00
**Checkpoint:** `models/student_global_stable_v5_t2mRMSE.ckpt`
**N inits:** 20 (seasons: DJF, JJA, MAM, SON)
**Leads:** [6, 12, 24]
**K1 coverage:** 0 inits — []

## Protocol

- Analysis-forced only (Case A / Cout=3): IC at `init+(L−6)h` → one +6h step.
- IC + truth: public ARCO ERA5 (no CDS).
- Gate variable: **t2m**. tp out-of-scope (dry-collapse).
- Student and K1 scored with the same cosine-latitude RMSE / ACC / bias vs ERA5.

## Lead-time table (t2m, all inits)

| lead | n | student RMSE | student ACC | student bias | K1 RMSE | K1 ACC | deg vs K1 |
|-----:|--:|-------------:|------------:|-------------:|--------:|-------:|----------:|
| 6 h | 20 | 1.377 | 0.967 | 0.016 | — | — | — |
| 12 h | 20 | 7.704 | 0.459 | -5.187 | — | — | — |
| 24 h | 20 | 3.343 | 0.833 | 1.425 | — | — | — |

## Seasonal breakdown (student vs ERA5)

### DJF

| lead | n | RMSE | ACC | bias | deg vs K1 |
|-----:|--:|-----:|----:|-----:|----------:|
| 6 h | 5 | 1.452 | 0.976 | 0.096 | — |
| 12 h | 5 | 8.060 | 0.499 | -5.405 | — |
| 24 h | 5 | 3.209 | 0.887 | 1.306 | — |

### JJA

| lead | n | RMSE | ACC | bias | deg vs K1 |
|-----:|--:|-----:|----:|-----:|----------:|
| 6 h | 5 | 1.289 | 0.977 | -0.068 | — |
| 12 h | 5 | 7.688 | 0.549 | -5.291 | — |
| 24 h | 5 | 3.399 | 0.863 | 1.361 | — |

### MAM

| lead | n | RMSE | ACC | bias | deg vs K1 |
|-----:|--:|-----:|----:|-----:|----------:|
| 6 h | 5 | 1.396 | 0.959 | 0.132 | — |
| 12 h | 5 | 7.865 | 0.368 | -5.203 | — |
| 24 h | 5 | 3.543 | 0.780 | 1.655 | — |

### SON

| lead | n | RMSE | ACC | bias | deg vs K1 |
|-----:|--:|-----:|----:|-----:|----------:|
| 6 h | 5 | 1.371 | 0.956 | -0.097 | — |
| 12 h | 5 | 7.201 | 0.421 | -4.848 | — |
| 24 h | 5 | 3.220 | 0.803 | 1.380 | — |

## Critical questions

1. **Does +6h t2m stay strong outside January/DJF?** Yes — non-DJF mean RMSE=1.352 (overall +6h=1.377).
2. **How fast does skill deteriorate through 24h?** RMSE growth +6→+24h = 143%. Seasonal growth: {'DJF': 121.06751387786288, 'JJA': 163.61475689807335, 'MAM': 153.76255961581722, 'SON': 134.95160182305045}.
3. **Does +24h degradation vs K1 persist?** No / insufficient K1 (+24h deg=—; K1 n=0). Across-season +24h RMSE growth persistence: True.

**Verdict summary:** +6h t2m RMSE=1.3770507676989014; +24h RMSE=3.3430776056718656 (+143% vs +6h). K1 deg +6h=None; +24h=None (K1 n=0). 24h growth persists across seasons=True.

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


## Reproduce

```bash
CUDA_VISIBLE_DEVICES=1 MKL_INTERFACE_LAYER=GNU \
  python -u evaluation/trackB_t2m_expanded.py \
    --ckpt models/student_global_stable_v5.ckpt --device cuda \
    --ic_source arco --step_days 5 --leads 6,12,18,24
```

See also `scripts/run_trackB_t2m_expanded_cassava.sh` and README Track B section.


# African t2m baselines (persistence / student / K1)

**Protocol:** trackB_t2m_baselines_vs_expanded
**Generated:** 2026-08-30T06:21:26.978361+00:00

## Definitions

- **persistence_af:** T_hat(valid)=T(IC); IC=init+(L-6)h — matched to AF student protocol
- **persistence_init:** T_hat(valid)=T(init 00Z)
- **climatology:** not computed
- **student:** from TRACKB_T2M_EXPANDED.json (v5 AF)
- **k1:** from expanded JSON where available (n≈3 inits)

## Aggregate RMSE (°C)

*Same magnitude as Kelvin for temperature differences (1 °C ≡ 1 K interval).*

| Lead | Pers(AF) | Pers(init) | Student | Clim | K1 | Student skill vs Pers(AF) |
|-----:|---------:|-----------:|--------:|-----:|---:|--------------------------:|
| +6 h | 2.373 | 2.373 | **1.382** | — | 1.216 | 41.8% |
| +12 h | 6.581 | 6.955 | **7.800** | — | 1.698 | -18.5% |
| +18 h | 3.691 | 4.057 | **4.377** | — | 1.468 | -18.6% |
| +24 h | 3.879 | 1.474 | **3.231** | — | 1.467 | 16.7% |

Rows: 244.


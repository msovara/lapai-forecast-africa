# Mvula v5 — explainability (AI attribution)

**Figure:** [`figures/mvula_fig11_t2m_xai_attribution.png`](figures/mvula_fig11_t2m_xai_attribution.png)  
**Script:** `scripts/plot_mvula_xai_attribution.py`

## What this is

Gradient **saliency** on the frozen student (`student_global_stable_v5.ckpt`):
how much the African-mean predicted **2t** changes with each of the **65** input channels,
plus a spatial sensitivity map. Combined with the packaged AF skill table to explain the **06Z-IC** one-step cold-bias pathology.

## Method (honest limits)

| Item | Detail |
|------|--------|
| Target | African-mean `2t` (Cout index 2) |
| Score | Mean `|∂2t/∂x|` over the Africa land box |
| IC | Checkpoint climate (`input_mean`) + small noise, then training normalization |
| Not yet | ARCO ICs at 00Z vs 06Z for true IC-hour attribution contrast |

This is **model explainability** (what the CNN listens to), not a full physical causal proof.

## Variable-group sensitivity (sum over 13 levels)

| Variable | Relative |∂2t/∂x| |
|----------|-------------------|
| `t` | 44.6% |
| `z` | 24.4% |
| `q` | 16.0% |
| `u` | 8.4% |
| `v` | 6.6% |

## Top-10 channels

| Rank | Channel | Score |
|-----:|---------|------:|
| 1 | `t1000` | 0.005061 |
| 2 | `z925` | 0.0008487 |
| 3 | `q1000` | 0.0006693 |
| 4 | `z400` | 0.0006508 |
| 5 | `q925` | 0.0006344 |
| 6 | `z150` | 0.0003612 |
| 7 | `q850` | 0.0003228 |
| 8 | `t925` | 0.000313 |
| 9 | `z250` | 0.0002998 |
| 10 | `z600` | 0.0002985 |

## 06Z-IC pathology (observed AF package)

- **00Z IC** one-step RMSE ≈ **1.38 °C**; **06Z IC** RMSE ≈ **7.80 °C**.
- **06Z IC** bias ≈ **-5.33 °C** (cold).
- **61/61** inits cold at 06Z IC (range [-5.76, -4.91] °C).
- Protocol: columns are **analysis IC hours** under AF one-step — not autoregressive lead times.
- Persistence also beats the student at 06Z/12Z IC (see Fig. 8 / baselines).


## One-sentence takeaway

**The student is most sensitive to thermodynamic/moisture multilevel fields when forming African 2t; the 06Z-IC collapse is a systematic cold bias under AF, not an unexplained black-box glitch.**

## Regenerate

```bash
python scripts/plot_mvula_xai_attribution.py
```

# Track B report — student MVP gate

**Date:** 2026-08-25  
**Result:** **FAIL** — `1/3` checks (≤15% RMSE degradation vs K1 teacher on cache targets).

## Gate outcome (stable v2)

MVP protocol `trackB_cache_mvp` scored **`models/student_global_stable_v2.ckpt`** (80 epochs, L_A-only, `teacher_k1_cache_t128.zarr` T=128, 6h one-step, channels `tp/msl/2t`) against K1 `teacher_pred` and ERA5 targets in the same store.

| var | student RMSE vs ERA5 | teacher RMSE | degradation vs teacher | prior stable v1 (T=32) |
|-----|----------------------|--------------|------------------------|------------------------|
| tp  | 0.00254              | 0.00210      | **21.1%** (fail ≤15%)  | 22.5%                  |
| msl | 233                  | 249          | **−6.6%** (pass)       | 81.7%                  |
| 2t  | 3.78                 | 2.90         | **30.2%** (fail)       | 86.2%                  |

### Levers applied (v2)

1. **β/γ off** for the full run (`beta=0`, `gamma=0`, feature/spectral start @999) — train on prediction loss L_A only (prior v1 spectral L_C≈202 dominated once γ enabled).
2. **80 epochs** (vs 25), `steps_per_epoch=64`, same channel-norm L_A, head bias, soft tp/msl constraints, grad clip.
3. **Larger cache** — built `teacher_k1_cache_t128.zarr` (T=128, stride 8 from `era5_n96_2020_2021.zarr`, ~14 G, ~42 min on GPU1).

Train: L_A 0.85 → 0.12 (finite throughout; β=γ=0). Log: `/local/Mthetho/logs/trackB_student_stable_v2_train.log`. Gate: `/local/Mthetho/logs/trackB_stable_v2_gate.log`.

**Pass/fail:** still **FAIL** overall (need 3/3), but msl now beats teacher on-cache; 2t roughly halved vs v1; tp nearly unchanged.

**Prior artifacts kept:** `models/student_global_stable.ckpt` (v1), `models/student_global.ckpt` (catastrophic) — do not promote.

**Artifacts:** `reports/TRACKB_STUDENT_SCORECARD.json`, `reports/TRACKB_GATE.json`, runner `evaluation/trackB_gate.py`.

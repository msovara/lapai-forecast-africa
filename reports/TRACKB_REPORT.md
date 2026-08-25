# Track B report — student MVP gate

**Date:** 2026-08-25  
**Result:** **FAIL** — `2/3` checks (≤15% RMSE degradation vs K1 teacher on cache targets). Closest yet: only **tp** remains above the bar.

## Gate outcome (stable v3 — current)

MVP protocol `trackB_cache_mvp` scored **`models/student_global_stable_v3.ckpt`** (resume from v2, 80 more epochs, channel-weighted L_A, `teacher_k1_cache_t256.zarr` T=256, 6h one-step, channels `tp/msl/2t`) against K1 `teacher_pred` and ERA5 targets in the same store.

| var | student RMSE vs ERA5 | teacher RMSE | degradation vs teacher | v2 (T=128) | v1 (T=32) |
|-----|----------------------|--------------|------------------------|------------|-----------|
| tp  | 0.00255              | 0.00211      | **21.2%** (fail ≤15%)  | 21.1%      | 22.5%     |
| msl | 207                  | 250          | **−17.2%** (pass)      | −6.6%      | 81.7%     |
| 2t  | 2.50                 | 2.90         | **−13.8%** (pass)      | 30.2%      | 86.2%     |

### Levers applied (v3)

1. **Channel-weighted L_A** — `channel_weights=[3.0, 0.4, 3.0]` for `(tp, msl, 2t)` so tp/2t dominate the loss; msl already passed under v2.
2. **Continue from `student_global_stable_v2.ckpt`** — 80 epochs, `steps_per_epoch=64`, β/γ still off, prior stabilizations kept (channel-norm, soft tp, grad clip).
3. **Larger cache** — built `teacher_k1_cache_t256.zarr` (T=256, stride 4 from `era5_n96_2020_2021.zarr`, ~85 min on GPU1).

Train: L_A ~0.34 → ~0.27 (finite; weighted scale). Log: `/local/Mthetho/logs/trackB_student_stable_v3_train.log`. Gate: `/local/Mthetho/logs/trackB_stable_v3_gate.log`.

**Pass/fail:** still **FAIL** overall (need 3/3), but **msl and 2t both beat the teacher on-cache**; tp stuck ~21%.

## Prior: stable v2

| var | degradation | |
|-----|-------------|---|
| tp  | 21.1% fail  | |
| msl | −6.6% pass  | |
| 2t  | 30.2% fail  | |

Levers: β/γ=0, 80 epochs, T=128 cache. See logs `trackB_student_stable_v2_train.log` / `trackB_stable_v2_gate.log`.

**Artifacts kept:** `models/student_global_stable_v3.ckpt` (current), `…_v2.ckpt`, `…_stable.ckpt` (v1), `models/student_global.ckpt` (catastrophic) — do not promote broken ckpt.

**Artifacts:** `reports/TRACKB_STUDENT_SCORECARD.json`, `reports/TRACKB_GATE.json`, runner `evaluation/trackB_gate.py`.

### Suggested next (tp only)

- Heavier tp weight (e.g. `[5, 0.3, 2]`) or tp-focused continue-train from v3; and/or accept documented tp soft-fail and move to held-out Jan-2023 multi-lead eval.

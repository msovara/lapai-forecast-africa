# Track B — State closure verdict (Cout=3 vs Cin=65)

**Date:** 2026-08-26  
**Scope:** Inspect v5 student model/config/cache before any further forecasts.  
**Out of scope:** CDS 18Z chase, further tp tuning, large new forecast campaigns.

---

## Verdict: **Case A (intentional Cout=3)**

Genuine **free-run evaluation is not technically possible** with the current v5 student: the head predicts only three surface variables and there is **no** decoder/reconstruction path back to the 65-channel atmospheric state required as the next-step input.

Evaluation must remain **analysis-forced / one-step** (ERA5 IC → +6 h prediction). Multi-lead scoring is only valid when each lead re-injects analysis ICs (as in `--student_leads`).

---

## Answers to the five questions

### 1. What are the 3 predicted variables? (order)

**Order (head / cache / losses): `tp`, `msl`, `2t`**

| Index | Name | Meaning |
|------:|------|---------|
| 0 | `tp` | total precipitation |
| 1 | `msl` | mean sea-level pressure |
| 2 | `2t` | 2 m temperature (scored / discussed as `t2m`) |

Evidence:

- Cache builder: `TARGET_NAMES = ["tp", "msl", "2t"]` in `training/build_teacher_feature_cache.py`
- Train fallback stats / comments: `_FALLBACK_MEAN = [5e-4, 1.01e5, 278.0]` and `channel_weights (tp,msl,2t)` in `training/train_student.py`
- Loss soft constraints: `apply_soft_physical_constraints` documents Cout=3 `(tp, msl, 2t)` in `utils/losses_distillation.py`
- Held-out eval mapping: `chan = {"tp": 0, "msl": 1, "t2m": 2}` in `evaluation/trackB_held_out_jan2023.py`
- Configs: `configs/student_global.yaml`, `student_global_v4.yaml`, `student_global_v5.yaml` all comment `# Cout order: tp, msl, 2t`

### 2. Are they t2m + something else?

**Yes.** The three outputs are **`tp` + `msl` + `2t`**, matching Track B history. Channel 2 is ERA5 name **`2t`** (not `t2m`); eval aliases it to `t2m`.

v5 checkpoint confirmation (Cassava `models/student_global_stable_v5.ckpt`):

- `cfg.out_channels = 3`, `cfg.in_channels_raw = 65`
- `head.weight` shape `(3, 256, 1, 1)`, `head.bias` shape `(3,)`
- `channel_weights = [6.0, 0.25, 2.5]` (tp, msl, 2t)
- `target_mean ≈ [5.89e-4, 1.009e5, 278.9]` — physical scales for precip (m), MSLP (Pa), 2 m temperature (K)
- `tp_mode = softplus_soft`

### 3. Is the model intentionally a partial-state student?

**Yes.** Architecture and docs treat Cout=3 as a **headline / distillation MVP**, not a full-state emulator.

- `LapAIStudentConfig.out_channels: int = 3` with docstring: *“Tp, MSLP, T2m headline outputs for distillation demo; extend for full state.”* (`lapai_inference/model.py`)
- Input remains full Cin=65 (`5` multilevel vars × `13` levels); only the **output head** is partial
- Pressure mixer `13→3` is a **vertical compression inside the backbone**, not a full-state output decoder

### 4. Was it trained to reproduce only selected variables?

**Yes.** Supervision targets and teacher predictions in the distillation cache are only the three selected fields.

- Builder extracts only `TARGET_NAMES` for `era5_target` and slices K1 outputs to the same three for `teacher_pred`
- Dataset contract: `era5_target` / `teacher_pred` are `(T, Cout, H, W)` with Cout from cache (3); `state_in` is `(T, Cin, H, W)` with Cin=65
- Cassava `teacher_k1_cache_t256.zarr`: `state_in (256, 65, 181, 360)`, `era5_target (256, 3, 181, 360)`, `teacher_pred (256, 3, 181, 360)`
- Loss `L_A` (+ optional precip terms on index 0) compares student `pred` to those Cout=3 targets — never to a 65-ch next state

### 5. Is there a decoder/reconstruction layer 3→65?

**No.**

Searched repo for decoder / reconstruct / state_out / free-run / rollout closure paths:

- Student forward returns only `{"pred", "feat_stage2", "feat_stage3"}` — `pred` is the 3-channel head
- `lapai_inference/postprocess.py` is a **placeholder** denormalize stub, not a state reconstructor
- v5 ckpt has **no** decoderish parameter keys (`decoder`, `recon`, `state_out`, …)
- Held-out protocol explicitly states student cannot free-run because Cout=3 cannot update the 65-ch state

---

## Case A vs Case B

| Case | Meaning | Finding |
|------|---------|---------|
| **A** | Intentional Cout=3 | **Confirmed** — design of model, cache, losses, configs, and eval |
| **B** | Config/training bug | **Rejected** — not a mis-set `out_channels`; no missing decoder bug to “fix” |

**Not a bug:** Cin=65 / Cout=3 asymmetry is the MVP student contract. Closing free-run would require a **new** full-state (or reconstructive) head and matching cache/loss redesign — that is future work, not a tiny config flip. Do **not** launch an expensive retrain for state closure until that design is specified.

---

## Implications for free-run

| Capability | Status |
|------------|--------|
| Analysis-forced one-step (+6 h) | Supported |
| Analysis-forced multi-lead (re-IC each lead) | Supported (already used for 6 h / 24 h) |
| Free-running multi-step rollout | **Not possible** with v5 |
| PLAN multi-day free-run / z500 / t850 student skill | **Not claimable** on this head |

---

## Recommended evaluation framing (locked for Code4Earth handover)

1. **Stop** treating free-run / full-state rollout as a Track B deliverable for this checkpoint family.
2. **Keep** analysis-forced scoring as the honest protocol: primary metric **t2m (`2t`)**; **tp out-of-scope** (already declared); **msl** on-cache only unless a truth stem is added.
3. **Production statistical eval (in progress / freeze):** multi-season 2023 inits, leads `{6,12,18,24}` h, ARCO ERA5 ICs — see `evaluation/trackB_t2m_expanded.py`, `reports/TRACKB_T2M_EXPANDED.md`.
4. **Do not** chase CDS auth, tp recovery, Cout redesign, or v6 architecture before handover.
5. If free-run is later required: open a **separate design track** for Cout≥65 (or explicit 3→65 reconstruction + teacher-aligned state targets). Prefer a written design + tiny PoC head smoke before any large retrain.

---

## Artifacts / how to re-verify

- Cassava ckpt: `/local/Mthetho/lapai-forecast/models/student_global_stable_v5.ckpt`
- Inspect helper: `scripts/_inspect_v5_state_closure.py`
- Expanded t2m eval: `evaluation/trackB_t2m_expanded.py` + `scripts/run_trackB_t2m_expanded_cassava.sh`
- Results: `reports/TRACKB_T2M_EXPANDED.json` / `.md`, figures under `reports/figures/`
- Related narrative: `reports/TRACKB_REPORT.md` (held-out / tp out-of-scope sections)

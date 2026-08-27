# Mvula — Code for Earth 2026 Final Report

**Project:** Mvula (African Stream)  
**Repository:** [msovara/lapai-forecast-africa](https://github.com/msovara/lapai-forecast-africa)  
**Report date:** 2026-08-27  
**Close-out horizon:** 23 September 2026  
**Frozen student:** `models/student_global_stable_v5.ckpt`  
**Evidence commit (packaged eval + dashboard):** `3b7cde4`  
**Release tag (intended):** `trackb-v5-c4e`

---

## 1. Executive summary

Mvula set out to shrink advanced AI weather models so useful forecasting can run on ordinary computers, with an Africa focus. By Code for Earth close-out we deliver a **defensible scoped result**, not the full original 10-day free-running laptop AIFS:

> **Mvula demonstrates substantial model compression (~6× smaller on disk than the accepted K1 teacher) while retaining useful short-range African t2m skill under an analysis-forced protocol. The +6 h result is relatively strong (~+17% RMSE vs K1; ACC≈0.97); skill degrades substantially by +24 h (~+103% vs K1). The project therefore supports a laptop-scale, short-range African forecasting use case rather than the originally envisioned 10-day free-running system.**

**Claim boundary (read this first):**

| We claim | We do **not** claim |
|----------|---------------------|
| Compressed student (~9 MiB, ~2.2 M params) vs K1 teacher | 10-day autonomous / free-running student forecasts |
| Analysis-forced African **t2m** skill at +6…+24 h | PLAN Week-9 multi-var skill table (`z500`/`t850`/`tp` @ 24–240 h free-run) |
| CPU inference of the student head (Cassava proxy; real laptop TO COMPLETE) | Finished ONNX / INT8 NMHS product |
| Open code + reproducible AF eval artefacts | Operational NMHS pilot or LoRA country adapters |
| Honest Case A architecture limit (Cout=3) | That v5 “is” a closed dynamical NWP emulator |

---

## 2. Original plan vs final scope

| Original Mvula / PLAN intent | Final close-out scope |
|------------------------------|------------------------|
| Shrink AIFS for laptop use | Track A coarsen + K1 prune → Track B CNN student (~6× vs K1) |
| 10-day free-running forecast | **Not demonstrated** (v5 cannot free-run) |
| Full-state student I/O | **Cout=3** (`tp`, `msl`, `2t`); Cin=65 input |
| Week-9 multi-var ≤15% skill table | **t2m-only** AF leads {6,12,18,24} vs K1/ERA5 |
| LoRA + quantisation + ONNX product | Scaffold / proxy only; productisation remains |
| Africa extremes ≤20% gate | Africa-domain AF scoring; formal extremes gate not closed |

Control documents: [`PLAN.md`](../PLAN.md), [`MVULA_CODE4EARTH_STATUS_MATRIX.md`](MVULA_CODE4EARTH_STATUS_MATRIX.md), [`TRACKB_STATE_CLOSURE.md`](TRACKB_STATE_CLOSURE.md).

---

## 3. What was achieved

### 3.1 Track A — compression teacher path

- Phase 0 N320 teacher baseline closed.
- A1 grid coarsening gate **passed** (≤5% vs Phase 0).
- A2 round-1 **K1** (1/16 heads + recovery) **accepted** as distillation teacher with documented tp trade-off.
- Evidence: `TRACKA_REPORT.md`, `TRACKA_A1_GATE.json`, prune scorecards.

### 3.2 Track B — distilled student v5

- InceptionNeXt-style CNN student: **Cin=65 → Cout=3** (`tp`, `msl`, `2t`).
- Frozen checkpoint: `student_global_stable_v5.ckpt`.
- **Case A (intentional):** no 3→65 decoder; evaluation is **analysis-forced** only ([`TRACKB_STATE_CLOSURE.md`](TRACKB_STATE_CLOSURE.md)).
- On-cache MVP: msl/2t beat K1; tp soft-fail then declared **out-of-scope** after dry-collapse.

### 3.3 African t2m evaluation (primary skill deliverable)

Source: [`TRACKB_T2M_EXPANDED.md`](TRACKB_T2M_EXPANDED.md) / `.json` (commit `3b7cde4`).

| Item | Value |
|------|--------|
| Protocol | Analysis-forced: IC at `init+(L−6)h` → one +6 h student step |
| Domain | Africa box |
| Inits | **61** across **DJF / MAM / JJA / SON** |
| Leads | **6, 12, 18, 24 h** |
| Truth / ICs | Public ARCO ERA5 (no CDS dependency for this campaign) |
| Baseline | K1 where available (**n=3** Jan-2023 weekly inits for deg%) |

**Lead-time table (t2m, all inits):**

| lead | n | student RMSE | student ACC | bias | K1 RMSE | deg vs K1 |
|-----:|--:|-------------:|------------:|-----:|--------:|----------:|
| 6 h | 61 | **1.382** | **0.965** | −0.12 | 1.216 | **+16.6%** |
| 12 h | 61 | 7.800 | 0.446 | −5.33 | 1.698 | +359.8% |
| 18 h | 61 | 4.377 | 0.749 | −1.12 | 1.468 | +204.0% |
| 24 h | 61 | **3.231** | **0.835** | +1.30 | 1.467 | **+102.6%** |

**Critical findings:**

1. **+6 h stays strong outside DJF** — non-DJF mean RMSE≈1.37 (overall 1.38).
2. **Skill deteriorates quickly** — RMSE growth +6→+24 h ≈ **+134%**; persists in all seasons.
3. **+24 h degradation vs K1 persists** — ≈**+103%** (K1 n=3).

Spatial maps: `reports/figures/trackb_t2m_v5_{rmse,bias}_L{006,024}h.png`.

### 3.4 Model compression & laptop inference (deployment deliverable)

Source: [`MVULA_LAPTOP_BENCHMARK.md`](MVULA_LAPTOP_BENCHMARK.md).

| Metric | Student v5 | K1 teacher |
|--------|------------|------------|
| Disk size | **8.8 MiB** | 52.8 MiB (~**6×** larger) |
| Parameters | **2.17 M** | (Anemoi GT) |
| +6 h step (CPU, 4 threads) | **~2.0 s** | — |
| AF 4-lead infer-only | **~8 s** | — |
| GPU required for student forward | **No** | Typically yes |

**Caveat:** timings are a **Cassava CPU proxy**, not yet a measured Intel i7 / 16 GB consumer laptop. Real-laptop re-bench is the preferred remaining experiment.

### 3.5 Open source & tooling

- Public GitHub repo and packaged reports.
- Streamlit status dashboard: `streamlit run streamlit_status.py` / `run_status_dashboard.bat`.
- Repro scripts: `evaluation/trackB_t2m_expanded.py`, `scripts/run_trackB_t2m_expanded_cassava.sh`, `scripts/bench_mvula_laptop_v5.py`.

---

## 4. What remains / was not demonstrated

| Item | Status | Note |
|------|--------|------|
| Real i7/16 GB laptop wall-clock + peak RAM | **TO COMPLETE** | Highest-value remaining experiment |
| ONNX / INT8 package | **NOT SHOWN** | Optional; only if &lt;1 day and risk-free |
| LoRA Africa adapters | **NOT SHOWN** | Scaffold only |
| Free-run / Cout=65 v6 | **NOT POSSIBLE** on v5 | Future architecture |
| 10-day student forecast | **NOT SHOWN** | Do not claim |
| PLAN Week-9 full skill table | **NOT SHOWN** | Deferred |
| tp skill | **FAILED** | Out-of-scope for this head |
| Formal Africa extremes ≤20% gate | **NOT CLOSED** | — |
| CDS 18Z parity ICs | **Deferred** | ARCO sufficient for packaged campaign |

**Explicit non-goals until after close-out packaging:** free-run redesign, tp recovery, CDS chase, full LoRA, more AF campaigns.

---

## 5. Case A / K1 → Mvula progression

```
AIFS / Phase 0 teacher
        ↓  coarsen (A1) + prune (K1)
K1 GraphTransformer teacher  (~53 MiB; free-run capable)
        ↓  distill (Track B)
Mvula student v5             (~9 MiB; Cout=3; AF only)
        ↓  evaluate
African AF t2m skill + CPU size/speed evidence
```

**Case A verdict:** Cout=3 is **intentional** MVP design. Channels are **`tp`, `msl`, `2t`**. There is **no** reconstruction path to Cin=65, so **genuine free-run is technically impossible**. Multi-lead scores are valid only with **re-injected analysis ICs**.

---

## 6. Checkpoint locations

| Artefact | Path |
|----------|------|
| Frozen student (Cassava) | `/local/Mthetho/lapai-forecast/models/student_global_stable_v5.ckpt` |
| Size | 8.9 MiB on disk (`sha256` prefix `5cf7080054bff60e…`) |
| Repo symlink / local name | `models/student_global_stable_v5.ckpt` |
| Teacher (Cassava) | `models/teacher_pruned.ckpt` → K1 `inference-last.ckpt` under `trackA_prune_k1_from_a1bplus_runs/...` |
| Packaged eval commit | `3b7cde4` on `main` |

Checkpoints are **not** stored in git (by design). Document the Cassava path in release notes when tagging.

---

## 7. Reproducibility (minimum path)

```text
Clone repo
  → create env (conda: environment-credit.yml / lapai-credit; or pip -e ".[dev]")
  → obtain student_global_stable_v5.ckpt (Cassava path above or scp)
  → run AF t2m eval (ARCO ICs) OR view packaged TRACKB_T2M_EXPANDED.*
  → optional: CPU laptop bench
  → streamlit run streamlit_status.py
```

See README § “Close-out quickstart” and [`TRACKB_METHODOLOGY_HANDOVER.md`](TRACKB_METHODOLOGY_HANDOVER.md).

---

## 8. Evidence checklist (control document)

Use this list to close gaps **without** new science campaigns:

| Evidence | Present? | Location |
|----------|----------|----------|
| Status matrix | Yes | `MVULA_CODE4EARTH_STATUS_MATRIX.md` |
| Expanded AF t2m tables | Yes | `TRACKB_T2M_EXPANDED.md` / `.json` |
| Spatial RMSE/bias maps | Yes | `reports/figures/trackb_t2m_v5_*.png` |
| K1 comparison at +6/+24 | Yes (n=3 for deg%) | same |
| Seasonal breakdown | Yes | same |
| Case A write-up | Yes | `TRACKB_STATE_CLOSURE.md` |
| Laptop size / CPU proxy | Yes | `MVULA_LAPTOP_BENCHMARK.*` |
| Real i7/16 GB bench | **No** | remaining |
| ONNX smoke | **No** | optional |
| FINAL_REPORT | **This file** | `reports/FINAL_REPORT.md` |
| Annotated release tag | Pending | `trackb-v5-c4e` |
| Streamlit shows expanded results | Yes | `streamlit_status.py` |

---

## 9. Future work (post–Code for Earth)

1. Real consumer-laptop benchmark (replace Cassava proxy caveat).  
2. If pursuing PLAN alignment: full-state (Cout≥65) student + free-run curriculum — **new architecture**, not a v5 tweak.  
3. Precip head redesign (current Cout=3 tp collapsed).  
4. LoRA / ENACTS regional adapters and extremes suite.  
5. ONNX Runtime packaging for NMHS distribution.

---

## 10. Team & programme

**Team:** Chimwemwe Chanda, Mthetho Sovara, Samuel Mathekga, Gabriel Elim, Fima Sichone.  
**Mentors:** Shruti Nath, Rendani Mbuvha, Mario Santa Cruz.  
**Programme:** ECMWF Code for Earth 2026 — African Stream.  
**Community link:** AfriClimate AI (relevance = open method + Africa-scored demo; not an ops pilot in this window).

---

## 11. One-sentence close-out

**Mvula v5 is a successful compression-and-short-range-t2m demonstration for Africa on a CPU-scale footprint — not a finished 10-day free-running laptop AIFS.**

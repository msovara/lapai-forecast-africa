# Mvula Code for Earth — Status Matrix

**Date:** 2026-08-26  
**Repo:** [msovara/lapai-forecast-africa](https://github.com/msovara/lapai-forecast-africa)  
**Frozen student:** `models/student_global_stable_v5.ckpt` (Cout=3: `tp` / `msl` / `2t`)  
**Primary close-out deliverables:** (1) analysis-forced **t2m skill**, (2) **laptop/deployment demonstration**.

> Status labels: **DONE** · **PARTIAL** · **RUNNING** · **NOT SHOWN** · **NOT POSSIBLE** · **FAILED** · **TO COMPLETE**

---

## Dual deliverables (handover framing)

| Deliverable | Status | Notes |
|-------------|--------|-------|
| Forecast skill (AF t2m, frozen v5) | **RUNNING** → package when finished | Production multi-season campaign on Cassava GPU1 (`trackB_t2m_expanded`; ~73 inits × leads 6/12/18/24). Do not kill. |
| Laptop / deployment demonstration | **PARTIAL** | CPU-only timed demo + size shrink documented (`MVULA_LAPTOP_BENCHMARK.md`). ONNX + consumer-laptop parity **TO COMPLETE**. |

---

## Objective → Status → Evidence

| # | Original Mvula / PLAN objective | Status | Evidence / honest note |
|---|----------------------------------|--------|-------------------------|
| 1 | **AIFS compression** (Track A structural compression of teacher) | **PARTIAL** | Phase 0 AIFS baseline closed; Track A path A1 → A1b → A1b+ → **K1 pruned GraphTransformer** accepted as distillation teacher (`reports/TRACKA_REPORT.md`, `PHASE0_CLOSURE.md`). Not a full AIFS→laptop single-artefact stack. |
| 2 | **Grid coarsening** (N320→N96 / O96→O48) | **DONE** | A1 GNN coarsen gate **30/30** ≤5% vs Phase 0 (`TRACKA_A1_GATE.json` / COARSEN scorecards). |
| 3 | **Attention-head pruning** | **PARTIAL** | A2 round-1 **K1** (1/16 heads + recovery) **accepted** with documented tp trade-off (31/40 vs A1b+; t2m clean). A2 rounds 2–3 **not started** (`TRACKA_REPORT.md`). |
| 4 | **LoRA** (Africa regional adapters) | **NOT SHOWN** | Config/scaffold exists (`configs/lora_africa.yaml`, `utils/lora_adapters.py`); no trained adapters / `LORA_REPORT.md` skill sweep. |
| 5 | **Quantization** (INT8 / ONNX size) | **NOT SHOWN** | PLAN §7.2 stretch; no ONNX export or quantisation run in this close-out. |
| 6 | **t2m retention** (skill vs teacher / ERA5) | **PARTIAL** (+ **RUNNING** production eval) | On-cache: 2t beats K1. Held-out Jan-2023 AF +6h Africa: ~+16–21% RMSE vs K1, ACC≈0.97 (`TRACKB_REPORT.md`, `TRACKB_HELD_OUT_JAN2023_V5.json`). Expanded multi-month AF campaign **RUNNING**. |
| 7 | **Multi-lead evaluation** | **PARTIAL** | Teacher free-run leads {6,24,72,120,240}h scored. Student: **analysis-forced** leads {6,24} done; expanded {6,12,18,24} **RUNNING**. Not free-run multi-day student leads. |
| 8 | **African evaluation** | **PARTIAL** | Africa box scoring throughout Track A/B gates and held-out. Extremes suite (`eval_africa_extremes.py` / ≤20% extremes budget) **not** closed as a formal gate. |
| 9 | **Laptop inference** | **PARTIAL** | v5 runs **CPU-only** (~2 s / +6h step, ~9 MiB ckpt) — see `MVULA_LAPTOP_BENCHMARK.md`. Consumer i7/16 GB end-to-end + ONNX package **TO COMPLETE**. |
| 10 | **Model size reduction** | **DONE** (student vs K1 teacher) | Student **8.8 MiB** / **2.17 M** params vs K1 teacher **52.8 MiB** (~**6×** smaller on disk). Full AIFS public ckpt shrink not re-benchmarked here. |
| 11 | **Inference speed-up** | **PARTIAL** | Small CNN head is fast on CPU (~2 s/step proxy). No formal AIFS/K1 vs student wall-clock table on identical hardware for 10-day free-run (student free-run N/A). |
| 12 | **10-day forecast** | **NOT POSSIBLE** (v5 student) | PLAN 40×6h free-run requires full atmospheric state. v5 **Cout=3** cannot close Cin=65 — see `TRACKB_STATE_CLOSURE.md`. Teacher (K1) can free-run; student cannot. |
| 13 | **Free-running v5** | **NOT POSSIBLE** | Intentional Case A (`TRACKB_STATE_CLOSURE.md`). Do not claim free-run. |
| 14 | **tp prediction** | **FAILED** / out-of-scope | Held-out all-dry (ACC≈0, POD₁ₘₘ=0) after v5 recovery attempt. Declared out-of-scope for Cout=3 MVP (`TRACKB_REPORT.md`). |
| 15 | **Open-source repository** | **DONE** | Public GitHub: `msovara/lapai-forecast-africa`. |
| 16 | **Reproducibility** | **PARTIAL** | Scripts, configs, scorecards, methodology handover (`TRACKB_METHODOLOGY_HANDOVER.md`). Full one-command laptop repro + IC caches packaging still hardening. |
| 17 | **Documentation** | **PARTIAL** | PLAN, Track A/B reports, state closure, methodology, this matrix, laptop bench. Final `FINAL_REPORT.md` / polished NMHS README **TO COMPLETE**. |
| 18 | **Community / local relevance** | **PARTIAL** (honest) | Africa-domain eval + NMHS laptop goal are real. No LoRA/ENACTS country adapters shipped; no operational NMHS pilot completed in this window. Relevance = open method + Africa-scored AF t2m demo, not a finished ops product. |

---

## Achieved vs remains (one-page handover)

### Achieved (defensible now)

- Track A coarsening gate + accepted K1 pruned teacher for Track B.
- Distilled **~2.2 M-param** student (v5) with **t2m** skill under **analysis-forced** protocol on Africa.
- Honest limitation docs: no free-run, tp out-of-scope, Case A Cout=3.
- Open repo + reproducibility scaffolding + CPU size/speed shrink evidence.

### Remains (do not over-claim)

- Finish & publish production AF t2m campaign artefacts (`TRACKB_T2M_EXPANDED.*`).
- Consumer-laptop / ONNX packaging and 16 GB RAM demonstration.
- LoRA, quantisation, full PLAN Week-9 variable table (`z500`/`t850`), free-run 10-day student.
- Formal African extremes ≤20% gate.

### Explicit non-goals for close-out

- Do **not** reopen free-run / Cout=65 v6 training unless a realistic existing route appears (none for Cout=3).
- Do **not** chase tp on this head.
- Do **not** disrupt the running GPU1 AF t2m campaign.

---

## Cross-links

| Doc | Role |
|-----|------|
| [`TRACKB_STATE_CLOSURE.md`](TRACKB_STATE_CLOSURE.md) | Why free-run / 10-day student is impossible |
| [`TRACKB_METHODOLOGY_HANDOVER.md`](TRACKB_METHODOLOGY_HANDOVER.md) | AF protocol + scoring lock |
| [`MVULA_LAPTOP_BENCHMARK.md`](MVULA_LAPTOP_BENCHMARK.md) | Size / CPU timing |
| [`TRACKB_PLAN_ALIGNMENT_ROADMAP.md`](TRACKB_PLAN_ALIGNMENT_ROADMAP.md) | Revised close-out priority order |
| [`TRACKA_REPORT.md`](TRACKA_REPORT.md) | Compression / prune verdict |
| [`TRACKB_REPORT.md`](TRACKB_REPORT.md) | Student MVP + held-out narrative |

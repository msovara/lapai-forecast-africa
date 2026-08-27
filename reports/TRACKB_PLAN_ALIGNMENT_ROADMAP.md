# Track B / Mvula — Close-out roadmap (revised)

**Date:** 2026-08-26  
**Intent:** Match the original Mvula Code for Earth description — **forecast skill** + **laptop/deployment demonstration**. Best-effort PLAN stretch only after those are safe. **Do not** reopen free-run / Cout=65 v6 as the main path.

**Guaranteed handover artefact:** frozen v5 analysis-forced **t2m** evaluation.  
**Stretch:** packaging, docs honesty, optional cheap extras that do not risk the month.

---

## Priority order (revised)

| P | Item | Time box | Status |
|---|------|----------|--------|
| **1** | Finish / package production AF **t2m** eval (frozen v5) | Until campaign exits | **DONE** |
| **2** | **Laptop benchmark** (size, CPU step time, teacher vs student) | 0.5 day | **DONE** — measured on **i7-11800H / 32 GB** Windows laptop (`MVULA_LAPTOP_BENCHMARK.*`) |
| **3** | **Status matrix** (`MVULA_CODE4EARTH_STATUS_MATRIX.md`) | 0.5 day | **DONE** |
| **4** | Document **65→3 / no 10-day free-run** limitation | 0.25 day | **DONE** — link [`TRACKB_STATE_CLOSURE.md`](TRACKB_STATE_CLOSURE.md) |
| **5** | Freeze v5 + package README / repro notes | 1–2 days | **DONE** — `FINAL_REPORT.md`, tag `trackb-v5-c4e` |
| **6** | Honest “achieved / remains” for handover | with P3–P5 | **DONE** in status matrix + FINAL_REPORT |

### Explicitly deprioritised (do not reopen unless mandatory)

- Free-run student / Cout=65 v6 train  
- Architecture spikes for full-state reconstruction  
- tp recovery on Cout=3  
- LoRA / ONNX / z500–t850 heads — only after P1–P6 are solid, and only if time remains  

**Kill criteria (stretch):** if GPU time or calendar slips, stop at P1+P3+P4 documentation freeze; ship AF t2m + limitation docs + laptop size/CPU (now measured on consumer laptop).

---

## Dual deliverables checklist

### A — Forecast skill

- [x] Protocol lock: analysis-forced t2m, ARCO IC, same scoring vs K1 where available  
- [x] Production campaign JSON/MD/figures (`TRACKB_T2M_EXPANDED.*`) committed  
- [x] Short interpretation paragraph for NMHS readers  

### B — Laptop / deployment demo

- [x] Disk size student vs K1 teacher  
- [x] CPU-only forward timing (proxy host)  
- [x] Same bench on a real consumer laptop (i7-11800H / 32 GB)  
- [ ] Optional stretch: ONNX export smoke (not required for MVP close-out)  

---

## Related artefacts

| File | Role |
|------|------|
| [`MVULA_CODE4EARTH_STATUS_MATRIX.md`](MVULA_CODE4EARTH_STATUS_MATRIX.md) | Objective coverage |
| [`MVULA_LAPTOP_BENCHMARK.md`](MVULA_LAPTOP_BENCHMARK.md) | Deployment numbers |
| [`TRACKB_STATE_CLOSURE.md`](TRACKB_STATE_CLOSURE.md) | Free-run impossibility |
| [`TRACKB_METHODOLOGY_HANDOVER.md`](TRACKB_METHODOLOGY_HANDOVER.md) | AF methodology |
| [`PLAN.md`](../PLAN.md) | Original 12-week contract (reference only) |

# Mvula laptop / deployment benchmark (v5 freeze)

**Date:** 2026-08-27  
**Checkpoint:** `models/student_global_stable_v5.ckpt`  
**Machine-readable:** [`MVULA_LAPTOP_BENCHMARK.json`](MVULA_LAPTOP_BENCHMARK.json)  
**Script:** `scripts/bench_mvula_laptop_v5.py`

---

## Hardware (this run)

| Item | Value |
|------|--------|
| Host | `MSOVARA-NB2` (Windows 10/11) |
| Host class | **`consumer_laptop`** (not Cassava proxy) |
| CPU | **11th Gen Intel Core i7-11800H @ 2.30 GHz** (8 cores / 16 threads) |
| RAM | **31.7 GiB** total (~9.7 GiB available at run start) |
| Device | **CPU only** (`CUDA_VISIBLE_DEVICES` empty; torch 2.10+cpu) |
| Threads | OMP/MKL/torch = **4** |

---

## Headline numbers

| Metric | Student v5 | K1 teacher (`teacher_pruned.ckpt`) |
|--------|------------|-------------------------------------|
| Disk size | **8.8 MiB** | **52.8 MiB** (~**6.0×** larger) |
| Parameters | **2.169 M** | (Anemoi GT) |
| Channels | Cin=65 → Cout=**3** (`tp`,`msl`,`2t`) | Full-state capable |
| Load time (cold) | **0.04 s** | — |
| Mean +6 h step (synth IC, 7 timed) | **2.53 s** | — |
| AF package estimate (4 leads, infer only) | **~10.1 s** | — |
| Peak process RSS | **~1.19 GiB** | — |
| GPU required | **No** | Typically yes |
| Free-run 10-day | **Not supported** | Supported |

**vs prior Cassava CPU proxy:** same ~9 MiB / ~6× shrink; step time **~2.0 s → ~2.5 s** on this i7 (still seconds-scale); peak RSS higher here (~1.2 GiB vs ~0.5 GiB proxy — different env baseline).

---

## What this answers

**Can Mvula v5 run on a normal laptop?**  
**Yes** for the student head: Intel i7 class CPU, no discrete GPU needed, checkpoint &lt;10 MiB, inference ~2.5 s per +6 h step, peak RSS ~1.2 GiB (fits comfortably in 16 GB+ RAM).

**Still not a finished NMHS product:**

- Timing excludes ERA5 IC download/build (ARCO/CDS).
- No ONNX / INT8 package in this close-out.
- Analysis-forced only — **not** a 10-day free-running forecast ([`TRACKB_STATE_CLOSURE.md`](TRACKB_STATE_CLOSURE.md)).

---

## How to reproduce

**Path B (recommended on a laptop):**

```bash
cd /path/to/lapai-forecast
# conda activate mvula-enduser   # from environment-mvula-enduser.yml
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python run_mvula.py bench
# same as: python -u scripts/bench_mvula_laptop_v5.py
# → reports/MVULA_LAPTOP_BENCHMARK.json
```

**Path A (Apptainer):** bind-mount `models/` (and `reports/` to persist JSON) — see [`containers/README.md`](../containers/README.md).

Windows (PowerShell) used for this measurement with conda env `lapai-anemoi` (CPU torch).
---

## Relation to Mvula dual deliverables

1. **Skill** — [`TRACKB_T2M_EXPANDED.md`](TRACKB_T2M_EXPANDED.md) / [`FINAL_REPORT.md`](FINAL_REPORT.md).  
2. **Deployment** — this document: **measured** laptop CPU size/speed for frozen v5.

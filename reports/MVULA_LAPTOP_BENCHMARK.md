# Mvula laptop / deployment benchmark (v5 freeze)

**Date:** 2026-08-26  
**Checkpoint:** `models/student_global_stable_v5.ckpt`  
**Machine-readable:** [`MVULA_LAPTOP_BENCHMARK.json`](MVULA_LAPTOP_BENCHMARK.json)  
**Script:** `scripts/bench_mvula_laptop_v5.py`

---

## Caveat (read first)

Timed run was **CPU-only on Cassava** (`CUDA_VISIBLE_DEVICES` empty, `OMP/MKL/torch` threads = **4**). This is a **proxy** for “runs without a GPU,” **not** a measured Intel i7 / 16 GB consumer laptop. Peak RSS is process memory on a large host (baseline RSS already high from the conda env).

---

## Headline numbers

| Metric | Student v5 | K1 teacher (`teacher_pruned.ckpt`) |
|--------|------------|-------------------------------------|
| Disk size | **8.8 MiB** (9 228 506 B) | **52.8 MiB** (55 351 696 B) |
| Shrink factor (disk) | — | **~6.0×** larger than student |
| Parameters | **2.169 M** | (Anemoi GT; not re-counted here) |
| Channels | Cin=65 → Cout=**3** (`tp`,`msl`,`2t`) | Full AIFS-style state rollout capable |
| Device | **CPU** (GPU **not** required for student forward) | Typically GPU for training/forecast |
| Load time (cold) | **0.02 s** | — |
| Mean +6 h step (synth IC, 7 timed runs) | **1.99 s** | — |
| AF package estimate (4 leads: 6/12/18/24) | **~8.0 s** inference only | — |
| Hypothetical 40× head-only | ~80 s | **Not a valid 10-day student forecast** |
| Peak process RSS (this run) | **~471 MiB** | — |
| Free-run 10-day | **Not supported** | Supported (teacher NCs exist) |

---

## What “laptop deployable” means for v5

**Yes, for the student head:**

- Checkpoint is small (~9 MiB).
- Forward pass is CPU-only and ~2 s/step at 1° (181×360) with 4 threads on the proxy host.
- A short **analysis-forced** lead package (4 steps) is seconds of inference, not minutes — **excluding** ERA5 IC download/build.

**Not yet a finished Mvula laptop product:**

- No ONNX Runtime package / INT8 quantisation.
- IC pipeline (ARCO/CDS) + Python deps dominate real laptop friction.
- Cannot advertise a **10-day free-running** forecast from v5 (see [`TRACKB_STATE_CLOSURE.md`](TRACKB_STATE_CLOSURE.md)).
- Consumer 16 GB end-to-end wall-clock **TO COMPLETE** on a real laptop.

---

## Storage / deps (order-of-magnitude)

| Item | Approx. need |
|------|----------------|
| Student ckpt | ~9 MiB |
| Teacher ckpt (optional compare) | ~53 MiB |
| Python env (torch + xarray stack) | multi-GB (env-dependent) |
| ARCO IC npy cache (per init×time) | tens of MB each if cached |
| Forecast NetCDFs | small per init/lead |

---

## How to reproduce

```bash
cd /path/to/lapai-forecast
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_INTERFACE_LAYER=GNU
# optional: export LAPAI_REPO=/path/to/lapai-forecast
python -u scripts/bench_mvula_laptop_v5.py
# → reports/MVULA_LAPTOP_BENCHMARK.json
```

Prefer a **real laptop** re-run before the final Code for Earth demo; keep this Cassava CPU proxy as the interim evidence.

---

## Relation to Mvula dual deliverables

1. **Skill** — separate AF t2m campaign (do not use this bench to claim skill).  
2. **Deployment** — this document supports the shrink + CPU-inference claim for the distilled student.

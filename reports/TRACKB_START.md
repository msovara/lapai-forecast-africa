# Track B start checklist (Cassava)

**Teacher:** K1 pruned GraphTransformer — `models/teacher_pruned.ckpt`  
→ `trackA_prune_k1_from_a1bplus_runs/.../f52c98d7-4530-49d4-a230-309fa005e92a/inference-last.ckpt`  
**Decision:** see `TRACKA_REPORT.md` (accept K1; no A2 round 2).

---

## Prerequisites

| # | Check | How |
|---|--------|-----|
| 1 | Repo on Cassava | `cd /local/Mthetho/lapai-forecast` |
| 2 | GPU 1 free | `nvidia-smi -i 1` |
| 3 | K1 symlink | `readlink -f models/teacher_pruned.ckpt` ends in `f52c98d7-.../inference-last.ckpt` |
| 4 | ERA5 O96 Zarr | `test -d data/processed/lapai/era5_n96_2020_2021.zarr` |
| 5 | Track B env | Prefer `lapai-credit` from `environment-credit-lengau.yml` (or reuse `lapai-anemoi` for **synth smoke only** / cache build) |
| 6 | Editable install | `pip install -e . --no-deps` so `lapai_inference` imports |
| 7 | Distillation cache | Smoke cache: `data/processed/lapai/teacher_k1_cache_smoke.zarr` (real K1 hooks). Full years: rebuild with `--era5 .../era5_n96_2020_2021.zarr` |

---

## 1) Smoke (few steps, no teacher cache)

Pin GPU 1. Synthetic batches exercise CNN + losses A/(B/C later epochs).

```bash
ssh cassava-gpu   # or your jump path
cd /local/Mthetho/lapai-forecast
bash scripts/run_trackB_smoke_cassava.sh
```

Expect: finite `epoch 1 loss=...` and `saved ... student_smoke.pt`.

---

## 2) Teacher feature cache (K1 hooks)

**Builder:** `training/build_teacher_feature_cache.py` (needs **`lapai-anemoi`** + GPU1).

**Layer map (K1 has 8 processor layers, not 16):** PLAN placeholders L10/L14 → `processor.proc[5]` / `processor.proc[7]` (depth-proportional). See `configs/teacher_aifs.yaml` `layer_map`.

Smoke subset (8 ICs from `era5_n96_smoke.zarr`, 6h lead, 181×360):

```bash
export CUDA_VISIBLE_DEVICES=1 MKL_INTERFACE_LAYER=GNU
source /local/Mthetho/miniforge3/etc/profile.d/conda.sh
conda activate /local/Mthetho/envs/lapai-anemoi
bash scripts/run_trackB_teacher_cache_smoke.sh
# → data/processed/lapai/teacher_k1_cache_smoke.zarr (~879M)
# keys: state_in, era5_target, teacher_pred, teacher_feat_L10, teacher_feat_L14 (+ init_id, lead_hours)
```

Larger fill (still one-step 6h unless you change `--lead_steps` / add rollout):

```bash
python -u training/build_teacher_feature_cache.py \
  --era5 data/processed/lapai/era5_n96_2020_2021.zarr \
  --teacher models/teacher_pruned.ckpt \
  --out data/processed/lapai/teacher_k1_cache.zarr \
  --samples 200 --stride 4 --overwrite
```

Synthetic-only fallback: `python training/build_demo_zarr_cache.py`.

---

## 3) Train with cache

```bash
export CUDA_VISIBLE_DEVICES=1 MKL_INTERFACE_LAYER=GNU
source /local/Mthetho/miniforge3/etc/profile.d/conda.sh
conda activate /local/Mthetho/envs/lapai-credit
pip install -e . --no-deps
python -u training/train_student.py \
  --config configs/student_global.yaml \
  --cache data/processed/lapai/teacher_k1_cache_smoke.zarr \
  --epochs 25 --device cuda \
  --out models/student_global.ckpt
```

Cache smoke (1 epoch × 2 steps): `bash scripts/run_trackB_train_cache_smoke.sh`.

Lengau PBS (when moving off Cassava): `qsub pbs/student.pbs`.

Configs: `configs/student_global.yaml`, `configs/student_distill.yaml` — both point **`teacher_ckpt: models/teacher_pruned.ckpt`**.

---

## Blockers (as of 2026-08-25)

- ~~**`lapai-credit` env not present**~~ — `/local/Mthetho/envs/lapai-credit` (PyTorch 2.5.1 + CUDA). Use `export MKL_INTERFACE_LAYER=GNU` before activate.
- ~~**No real teacher distillation cache**~~ — smoke cache landed (`teacher_k1_cache_smoke.zarr`); full multi-year / 24h-lead fill still TODO.
- **Full Track B scale:** rebuild over `era5_n96_2020_2021.zarr`; optional multi-step rollout for 24h `teacher_pred` (builder default is 1×6h step). Feature regrid is NN from O48 hidden mesh → 1° grid.
- ~~Layer indices 10/14 placeholders~~ — mapped for K1: L10→5, L14→7.

### Synth smoke result (2026-08-25, Cassava GPU1)

```text
teacher_pruned -> .../f52c98d7-4530-49d4-a230-309fa005e92a/inference-last.ckpt
epoch 1 loss=1.489842 beta=False gamma=False cache=None
saved models/student_smoke.pt params 2169377
EXIT:0
```

Log: `logs/trackB_smoke_latest.log`.

### K1 cache smoke (2026-08-25)

```text
data/processed/lapai/teacher_k1_cache_smoke.zarr  (~879M)
T=8  cin=65 cout=3 d10=256 d14=256 grid=181x360
layer_map L10->proc[5] L14->proc[7]
Log: logs/trackB_teacher_cache_smoke.log
```

### Cache-backed train smoke (2026-08-25, `lapai-credit` GPU1)

```text
epoch 1 loss=24345.078125 beta=False gamma=False
cache=data/processed/lapai/teacher_k1_cache_smoke.zarr
saved models/student_cache_smoke.pt params 2169377
EXIT:0
```

Log: `logs/trackB_train_cache_smoke.log`. High raw loss is expected (physical ERA5 units; feature loss inactive until epoch 11).

---

## 4) MVP gate vs K1 (2026-08-25) — **FAILED 0/3**

25-epoch `models/student_global.ckpt` on medium cache scored with `evaluation/trackB_gate.py` (≤15% RMSE vs `teacher_pred`, vars `tp/msl/2t`, lead 6h, global).

| Var | Student RMSE vs ERA5 | Teacher RMSE vs ERA5 | Degradation vs teacher |
|-----|----------------------|----------------------|------------------------|
| tp  | 46.44                | 0.0021               | ≫ 15% (fail)           |
| msl | 42274                | 254                  | ≫ 15% (fail)           |
| 2t  | 46.70                | 2.90                 | ≫ 15% (fail)           |

Reports: `TRACKB_GATE.json`, `TRACKB_STUDENT_SCORECARD.json`, narrative `TRACKB_REPORT.md`.  
Caveats: cache T=32 / 6h one-step only; train set = score set; not PLAN multi-lead Jan-2023 IC gate. Failure mode: unphysical preds after feature-loss spike at epoch 11 (see `/local/Mthetho/logs/trackB_student_train.log`).

# Track A Step A2 — start here

**Date:** 2026-08-06  
**Status:** Scaffolding landed; **A1b GraphTransformer O96 fine-tune must run before head pruning.**

## Why A1 cannot be pruned directly

A1 (`trackA_coarsen_full` / extend) trained `model=gnn` for 32 GiB V100 memory.
**GNN processors have no attention heads.** PLAN §4.2 pruning only applies to
GraphTransformer (`num_heads`, `lin_query/key/value`, …).

| Checkpoint | Architecture | Head-prunable? |
|------------|--------------|----------------|
| `teacher_n320_gt6` | GraphTransformer, 16 heads | Yes (N320) |
| `teacher_coarsened` (A1) | GNN O96 | **No** |
| `teacher_gt_coarsened` (A1b) | GraphTransformer O96 | **Yes — A2 input** |

## Operator sequence

### 1) A1b — GraphTransformer on O96 Zarr

```bash
# laptop → Lengau: sync configs/trackA_gt_coarsen.yaml scripts/submit_trackA_gt_coarsen_32gb.sh
ssh lengau
cd ~/repos/lapai-forecast
bash scripts/submit_trackA_gt_coarsen_32gb.sh   # dual-queue gpu2005/gpu2006
# after Exit 0:
ln -sfn "$(readlink -f models/trackA_gt_coarsen_runs/checkpoint/*/inference-last.ckpt)" \
        models/teacher_gt_coarsened.ckpt
```

Config: `configs/trackA_gt_coarsen.yaml` (256 ch / 8 layers / **16 heads**, 2000 steps smoke-scale; raise later).

### 2) A2 round 1 — soft-mask 10% heads + recovery fine-tune

```bash
qsub pbs/trackA_prune.pbs
# or:
$LAPAI_PYTHON training/train_trackA.py --step prune --config configs/trackA_prune.yaml
```

Writes:
- `reports/TRACKA_A2_PRUNE_ROUND1.json` (which heads dropped)
- `models/teacher_pruned_masked.ckpt` then anemoi recovery under `models/trackA_prune_runs/`
- symlink target: `models/teacher_pruned.ckpt`

### 3) Forecast → score → A2 gate

Gate vs **A1 scorecard** (not Phase 0), variables **t2m, u10, v10, tp**, ≤**3%** RMSE degradation (deterministic proxy for PLAN CRPS).

```bash
# forecast with teacher_pruned.ckpt (reuse trackA_forecast.pbs + prune forecast_dir)
$LAPAI_PYTHON scripts/run_trackA_gate.py \
  --candidate reports/TRACKA_PRUNE_SCORECARD.json \
  --config configs/trackA_prune.yaml \
  --out reports/TRACKA_A2_GATE.json
```

## Implementation notes

- Soft-mask zeros head slices in `lin_*` / `projection` weights (survives ckpt I/O).
- Importance MVP: per-head weight L1 (`utils/head_prune.py`). Activation×grad hooks are a follow-up.
- `training/train_trackA.py --step prune --suggest-invocation` prints the full path.

## Risks

- GT O96 transfer from full `n320_gt6` (1024-ch) into 256-ch may under-transfer; if skill is poor, raise capacity or match teacher dims (may need multi-GPU / activation ckpt).
- First A1b job may OOM — dual-queue 32 GiB hosts; reduce `num_layers` / `num_heads` if needed.
- **Triton JIT needs modern GCC.** Job 7358489 failed when Triton used `/bin/gcc` 4.8.5 (`stdatomic.h` missing). `lapai_load_cuda_gpu` now loads `gcc/9.2.0` and sets `CC`/`CXX`.

#!/usr/bin/env bash
# A2 round-1 reduce-K: soft-mask 1/16 heads (~6%), then 3-epoch recovery.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-GNU}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export KMP_DUPLICATE_LIB_OK=TRUE

source /local/Mthetho/miniforge3/etc/profile.d/conda.sh
conda activate /local/Mthetho/envs/lapai-anemoi
cd /local/Mthetho/lapai-forecast

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES} $(date -Is)"
test -f models/teacher_gt_coarsened.ckpt
test -d data/processed/lapai/era5_n96_2020_2021.zarr

echo "[A2 K1] convert A1b inference -> warmstart"
python -u scripts/convert_inference_to_warmstart_ckpt.py \
  --in models/teacher_gt_coarsened.ckpt \
  --out models/teacher_gt_coarsened_warmstart.ckpt \
  --force

echo "[A2 K1] soft-mask fraction=0.0625 (1 of 16 heads)"
python -u - <<'PY'
from pathlib import Path
from utils.head_prune import prune_checkpoint, write_prune_report

warm = Path("models/teacher_gt_coarsened_warmstart.ckpt")
masked = Path("models/teacher_pruned_masked_k1.ckpt")
report = prune_checkpoint(warm, masked, fraction=0.0625, scope="processor", importance="weight_l1")
out = Path("reports/TRACKA_A2_PRUNE_ROUND1_K1.json")
out.parent.mkdir(parents=True, exist_ok=True)
write_prune_report(report, out)
print("wrote", out, "n_drop_sample", {k: v for k, v in list(report.get("dropped_heads", {}).items())[:2]})
print("masked_mb", round(masked.stat().st_size / 1e6, 1))
PY

echo "[A2 K1] recovery max_epochs=3 lr=5e-5"
python -u - <<'PY'
from pathlib import Path
from training.train_trackA import _run_coarsen_anemoi, _read_config

cfg = _read_config(Path("configs/trackA_prune.yaml"))
anemoi = dict(cfg.get("anemoi") or {})
anemoi["warm_start"] = "models/teacher_pruned_masked_k1.ckpt"
anemoi["output_root"] = "models/trackA_prune_k1_runs"
anemoi["output_ckpt"] = "models/teacher_pruned.ckpt"
anemoi["max_steps"] = 12000
anemoi["max_epochs"] = 3
cfg["anemoi"] = anemoi
raise SystemExit(_run_coarsen_anemoi(cfg))
PY
rc=$?
echo "[A2 K1] train exit=${rc}"

latest=$(find models/trackA_prune_k1_runs/checkpoint -name 'inference-last.ckpt' 2>/dev/null | sort | tail -n 1 || true)
if [[ -n "${latest}" ]]; then
  ln -sfn "$(readlink -f "${latest}")" models/teacher_pruned.ckpt
  ls -lh models/teacher_pruned.ckpt
fi
echo "[A2 K1] done $(date -Is) rc=${rc}"
exit "${rc}"

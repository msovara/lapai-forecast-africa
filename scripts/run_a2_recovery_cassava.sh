#!/usr/bin/env bash
# A2 recovery fine-tune on Cassava (cpt-gpu084).
# Remasks the Lightning warm-start (preserves hyper_parameters), then trains.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-GNU}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

source /local/Mthetho/miniforge3/etc/profile.d/conda.sh
conda activate /local/Mthetho/envs/lapai-anemoi
cd /local/Mthetho/lapai-forecast

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES} $(date -Is)"
nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-gpu=name,memory.used,memory.total --format=csv,noheader

test -f models/teacher_gt_coarsened.ckpt
test -d data/processed/lapai/era5_n96_2020_2021.zarr

echo "[A2] convert A1b inference → warmstart"
python -u scripts/convert_inference_to_warmstart_ckpt.py \
  --in models/teacher_gt_coarsened.ckpt \
  --out models/teacher_gt_coarsened_warmstart.ckpt \
  --force

echo "[A2] soft-mask heads on warmstart → teacher_pruned_masked.ckpt"
python -u - <<'PY'
from pathlib import Path
from utils.head_prune import prune_checkpoint, write_prune_report

warm = Path("models/teacher_gt_coarsened_warmstart.ckpt")
masked = Path("models/teacher_pruned_masked.ckpt")
report = prune_checkpoint(warm, masked, fraction=0.10, scope="processor", importance="weight_l1")
out = Path("reports/TRACKA_A2_PRUNE_ROUND1.json")
out.parent.mkdir(parents=True, exist_ok=True)
write_prune_report(report, out)
print("wrote", out, "masked_mb", round(masked.stat().st_size / 1e6, 1))
PY

echo "[A2] recovery fine-tune (max_steps=2000, lr=5e-5)"
python -u - <<'PY'
from pathlib import Path
from training.train_trackA import _run_coarsen_anemoi, _read_config

cfg = _read_config(Path("configs/trackA_prune.yaml"))
anemoi = dict(cfg.get("anemoi") or {})
anemoi["warm_start"] = "models/teacher_pruned_masked.ckpt"
anemoi["output_root"] = "models/trackA_prune_runs"
cfg["anemoi"] = anemoi
raise SystemExit(_run_coarsen_anemoi(cfg))
PY
rc=$?
echo "[A2] train exit=${rc}"

latest=$(find models/trackA_prune_runs/checkpoint -name 'inference-last.ckpt' 2>/dev/null | sort | tail -n 1 || true)
if [[ -n "${latest}" ]]; then
  ln -sfn "$(readlink -f "${latest}")" models/teacher_pruned.ckpt
  ls -lh models/teacher_pruned.ckpt
fi
echo "[A2] done $(date -Is) rc=${rc}"
exit "${rc}"

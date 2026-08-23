#!/usr/bin/env bash
# A2 FULL prune + recovery on Cassava (cpt-gpu084).
# Convert inference ckpt → Lightning warm-start, soft-mask heads, then recovery
# fine-tune. Isolated _full output paths; does not overwrite smoke A2 artifacts.
# Export MKL before set -u / conda activate (nounset otherwise kills conda).
export MKL_INTERFACE_LAYER=GNU
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
set -euo pipefail

source /local/Mthetho/miniforge3/etc/profile.d/conda.sh
conda activate /local/Mthetho/envs/lapai-anemoi
cd /local/Mthetho/lapai-forecast

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES} $(date -Is)"
nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-gpu=name,memory.used,memory.total --format=csv,noheader

test -e models/teacher_gt_coarsened.ckpt
test -d data/processed/lapai/era5_n96_2020_2021.zarr
echo "[A2] teacher_gt_coarsened.ckpt -> $(readlink -f models/teacher_gt_coarsened.ckpt)"
ls -l models/teacher_gt_coarsened.ckpt
echo "[A2] dataset=data/processed/lapai/era5_n96_2020_2021.zarr"

echo "[A2] convert A1b inference -> warmstart_full"
python -u scripts/convert_inference_to_warmstart_ckpt.py \
  --in models/teacher_gt_coarsened.ckpt \
  --out models/teacher_gt_coarsened_warmstart_full.ckpt \
  --force

echo "[A2] soft-mask heads on warmstart -> teacher_pruned_masked_full.ckpt"
python -u - <<'PY'
from pathlib import Path
from utils.head_prune import prune_checkpoint, write_prune_report

warm = Path("models/teacher_gt_coarsened_warmstart_full.ckpt")
masked = Path("models/teacher_pruned_masked_full.ckpt")
report = prune_checkpoint(warm, masked, fraction=0.10, scope="processor", importance="weight_l1")
out = Path("reports/TRACKA_A2_PRUNE_ROUND1_FULL.json")
out.parent.mkdir(parents=True, exist_ok=True)
write_prune_report(report, out)
print("wrote", out, "masked_mb", round(masked.stat().st_size / 1e6, 1))
PY

echo "[A2] recovery fine-tune (max_steps=2000, lr=5e-5, output=trackA_prune_runs_full)"
python -u - <<'PY'
from pathlib import Path
from training.train_trackA import _run_coarsen_anemoi, _read_config

cfg = _read_config(Path("configs/trackA_prune_full.yaml"))
anemoi = dict(cfg.get("anemoi") or {})
anemoi["warm_start"] = "models/teacher_pruned_masked_full.ckpt"
anemoi["output_root"] = "models/trackA_prune_runs_full"
anemoi["output_ckpt"] = "models/teacher_pruned_full.ckpt"
anemoi["masked_ckpt"] = "models/teacher_pruned_masked_full.ckpt"
cfg["anemoi"] = anemoi
raise SystemExit(_run_coarsen_anemoi(cfg))
PY
rc=$?
echo "[A2] train exit=${rc}"

latest=$(find models/trackA_prune_runs_full/checkpoint -name 'inference-last.ckpt' 2>/dev/null | sort | tail -n 1 || true)
if [[ -n "${latest}" ]]; then
  ln -sfn "$(readlink -f "${latest}")" models/teacher_pruned_full.ckpt
  ls -lh models/teacher_pruned_full.ckpt
fi
echo "[A2] done $(date -Is) rc=${rc}"
exit "${rc}"

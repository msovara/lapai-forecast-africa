#!/usr/bin/env bash
# Strengthen A1b: continue GT coarsen from current teacher_gt_coarsened.ckpt
# for 5 epochs (~12k steps), then K1 soft-mask (1/16 heads) + 3-epoch recovery.
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
nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-gpu=name,memory.used,memory.total --format=csv,noheader
test -e models/teacher_gt_coarsened.ckpt
test -d data/processed/lapai/era5_n96_2020_2021.zarr

echo "[A1b+] extend GT coarsen max_epochs=5 from current teacher_gt_coarsened.ckpt"
python -u - <<'PY'
from pathlib import Path
from training.train_trackA import _run_coarsen_anemoi, _read_config

cfg = _read_config(Path("configs/trackA_gt_coarsen.yaml"))
anemoi = dict(cfg.get("anemoi") or {})
anemoi["warm_start"] = "models/teacher_gt_coarsened.ckpt"
anemoi["output_root"] = "models/trackA_gt_coarsen_extend_runs"
anemoi["output_ckpt"] = "models/teacher_gt_coarsened.ckpt"
anemoi["max_steps"] = 20000
anemoi["max_epochs"] = 5
# keep lr from hydra_overrides if any; A1b yaml has no explicit lr — anemoi default OK
cfg["anemoi"] = anemoi
raise SystemExit(_run_coarsen_anemoi(cfg))
PY
rc=$?
echo "[A1b+] train exit=${rc}"
latest=$(find models/trackA_gt_coarsen_extend_runs/checkpoint -name 'inference-last.ckpt' 2>/dev/null | sort | tail -n 1 || true)
if [[ -z "${latest}" ]]; then
  echo "[A1b+] ERROR: no inference-last.ckpt"
  exit 1
fi
ln -sfn "$(readlink -f "${latest}")" models/teacher_gt_coarsened.ckpt
ls -lh models/teacher_gt_coarsened.ckpt
echo "[A1b+] extended ckpt ready $(date -Is)"

echo "[A2 K1] convert extended A1b -> warmstart"
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
print("wrote", out, "masked_mb", round(masked.stat().st_size / 1e6, 1))
print("dropped", report.get("dropped_heads") or report.get("drop") or list(report.keys())[:8])
PY

echo "[A2 K1] recovery max_epochs=3 lr=5e-5"
python -u - <<'PY'
from pathlib import Path
from training.train_trackA import _run_coarsen_anemoi, _read_config

cfg = _read_config(Path("configs/trackA_prune.yaml"))
anemoi = dict(cfg.get("anemoi") or {})
anemoi["warm_start"] = "models/teacher_pruned_masked_k1.ckpt"
anemoi["output_root"] = "models/trackA_prune_k1_from_a1bplus_runs"
anemoi["output_ckpt"] = "models/teacher_pruned.ckpt"
anemoi["max_steps"] = 12000
anemoi["max_epochs"] = 3
cfg["anemoi"] = anemoi
raise SystemExit(_run_coarsen_anemoi(cfg))
PY
rc2=$?
echo "[A2 K1] train exit=${rc2}"
latest2=$(find models/trackA_prune_k1_from_a1bplus_runs/checkpoint -name 'inference-last.ckpt' 2>/dev/null | sort | tail -n 1 || true)
if [[ -n "${latest2}" ]]; then
  ln -sfn "$(readlink -f "${latest2}")" models/teacher_pruned.ckpt
  ls -lh models/teacher_pruned.ckpt
fi
echo "[pipeline] done $(date -Is) a1b_rc=${rc} prune_rc=${rc2}"
exit "${rc2}"

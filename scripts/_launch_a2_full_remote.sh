#!/usr/bin/env bash
# One-shot: retarget teacher symlink, collision-check, launch a2_prune_full tmux.
export MKL_INTERFACE_LAYER=GNU
set -euo pipefail
cd /local/Mthetho/lapai-forecast

echo "===== collision ====="
hostname
tmux ls || true
pgrep -af 'train_trackA|anemoi-training|run_a2' || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
echo "===== gpu procs ====="
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv

echo "===== file check ====="
file /local/Mthetho/run_a2_prune_full.sh /local/Mthetho/lapai-forecast/scripts/run_a2_prune_full.sh /local/Mthetho/lapai-forecast/configs/trackA_prune_full.yaml
chmod +x /local/Mthetho/run_a2_prune_full.sh /local/Mthetho/lapai-forecast/scripts/run_a2_prune_full.sh
head -c 20 /local/Mthetho/run_a2_prune_full.sh | od -An -tx1 | head -1
grep -E 'dataset:|output_root:|output_ckpt:|masked_ckpt:|max_steps:' configs/trackA_prune_full.yaml

echo "===== teacher symlink ====="
A1B=/local/Mthetho/lapai-forecast/models/trackA_gt_coarsen_runs/checkpoint/079799cd-7432-4805-b365-51e909928c4f/inference-last.ckpt
test -f "$A1B"
ls -lh "$A1B"
if [ -f models/teacher_gt_coarsened.ckpt ] && [ ! -L models/teacher_gt_coarsened.ckpt ]; then
  if [ ! -e models/teacher_gt_coarsened_aug19.ckpt ]; then
    mv models/teacher_gt_coarsened.ckpt models/teacher_gt_coarsened_aug19.ckpt
    echo "moved regular teacher to teacher_gt_coarsened_aug19.ckpt"
  else
    echo "aug19 backup already exists; unlinking current regular file"
    rm -f models/teacher_gt_coarsened.ckpt
  fi
fi
ln -sfn "$A1B" models/teacher_gt_coarsened.ckpt
ls -l models/teacher_gt_coarsened.ckpt
readlink -f models/teacher_gt_coarsened.ckpt
readlink -f models/teacher_gt_coarsened.ckpt | grep -q 079799cd-7432-4805-b365-51e909928c4f
echo "teacher symlink OK"

if tmux has-session -t a2_prune_full 2>/dev/null; then
  echo "tmux a2_prune_full already exists — not duplicating"
  tmux ls
  exit 0
fi

pick_gpu() {
  local i used
  for i in 1 2 3 4 5 6 7; do
    used=$(nvidia-smi -i "$i" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if [ "${used:-999999}" -lt 200 ]; then
      echo "$i"
      return 0
    fi
  done
  echo "no free GPU in 1-7" >&2
  return 1
}
GPU=$(pick_gpu)
echo "selected GPU=$GPU"

mkdir -p /local/Mthetho/logs
: > /local/Mthetho/logs/a2_prune_full.log

tmux new-session -d -s a2_prune_full "export MKL_INTERFACE_LAYER=GNU CUDA_VISIBLE_DEVICES=${GPU}; bash /local/Mthetho/run_a2_prune_full.sh > /local/Mthetho/logs/a2_prune_full.log 2>&1"
sleep 2
tmux ls
echo "===== launched gpu=${GPU} ====="

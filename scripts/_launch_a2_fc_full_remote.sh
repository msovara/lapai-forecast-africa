#!/usr/bin/env bash
# Collision-check, pick GPU, launch a2_fc_gate_full tmux (pkill-resistant script name).
export MKL_INTERFACE_LAYER=GNU
set -euo pipefail
cd /local/Mthetho/lapai-forecast

echo "===== collision ====="
hostname
date -Is
tmux ls || true
pgrep -af 'run_phase0_forecast|run_tracka_a2_full_fcgate|a2_fc_gate' || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv || true

if tmux has-session -t a2_fc_gate_full 2>/dev/null; then
  echo "tmux a2_fc_gate_full already exists — not duplicating"
  tmux ls
  exit 0
fi

if tmux has-session -t a2_fc_gate 2>/dev/null; then
  echo "===== a2_fc_gate is SMOKE rescore; leaving it; using a later GPU ====="
  tmux capture-pane -pt a2_fc_gate -S -12 || true
fi

echo "===== file check ====="
cp -f /local/Mthetho/run_a2_forecast_gate_full.sh /local/Mthetho/run_tracka_a2_full_fcgate.sh
chmod +x /local/Mthetho/run_a2_forecast_gate_full.sh /local/Mthetho/run_tracka_a2_full_fcgate.sh /local/Mthetho/lapai-forecast/scripts/run_a2_forecast_gate_full.sh
file /local/Mthetho/run_tracka_a2_full_fcgate.sh
readlink -f models/teacher_pruned_full.ckpt
test -d data/processed/lapai/era5_n96_2020_2021.zarr
readlink -f models/teacher_pruned_full.ckpt | grep -q 31676a83-4919-4741-acaa-118fced72eab
echo "ckpt uuid OK"

pick_gpu() {
  local i used
  local prefer_skip_1=0
  if tmux has-session -t a2_fc_gate 2>/dev/null; then
    prefer_skip_1=1
  fi
  for i in 1 2 3 4 5 6 7; do
    if [ "$prefer_skip_1" -eq 1 ] && [ "$i" -eq 1 ]; then
      continue
    fi
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
echo "selected GPU=${GPU}"

mkdir -p /local/Mthetho/logs
{
  echo ""
  echo "===== relaunch $(date -Is) gpu=${GPU} ====="
} >> /local/Mthetho/logs/a2_forecast_gate_full.log

tmux new-session -d -s a2_fc_gate_full bash --noprofile --norc -c "export MKL_INTERFACE_LAYER=GNU CUDA_VISIBLE_DEVICES=${GPU}; exec bash /local/Mthetho/run_tracka_a2_full_fcgate.sh >> /local/Mthetho/logs/a2_forecast_gate_full.log 2>&1"
sleep 4
tmux ls
echo "===== launched gpu=${GPU} ====="
tail -n 40 /local/Mthetho/logs/a2_forecast_gate_full.log || true

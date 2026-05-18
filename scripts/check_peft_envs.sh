#!/usr/bin/env bash
# Run on Lengau: finds user-local conda envs that have both peft and torch.
set -euo pipefail
for d in /home/msovara/.conda/envs/*/bin/python; do
  [[ -x "$d" ]] || continue
  if "$d" -c "import peft, torch" 2>/dev/null; then
    echo "PEFT_TORCH_OK $d"
  fi
done
echo "done"

#!/usr/bin/env python3
"""Write a torchinfo layer table for frozen Mvula v5.

Usage (from repo root, CPU env with torch + torchinfo):
  python scripts/visualize_mvula_v5_arch.py

Output:
  reports/MVULA_V5_TORCHINFO.txt
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

CKPT = _REPO / "models" / "student_global_stable_v5.ckpt"
OUT = _REPO / "reports" / "MVULA_V5_TORCHINFO.txt"


def main() -> int:
    import torch
    from evaluation.trackB_gate import _load_student

    if not CKPT.is_file():
        print(f"missing checkpoint: {CKPT}", file=sys.stderr)
        return 1

    try:
        from torchinfo import summary
    except ImportError:
        print("Installing torchinfo...", file=sys.stderr)
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torchinfo"])
        from torchinfo import summary

    net, _meta = _load_student(CKPT, torch.device("cpu"))
    n = sum(p.numel() for p in net.parameters())
    s = summary(
        net,
        input_size=(1, 65, 181, 360),
        device="cpu",
        verbose=0,
        col_names=("input_size", "output_size", "num_params", "kernel_size"),
    )
    text = (
        "Mvula v5 architecture summary (torchinfo)\n"
        f"checkpoint: {CKPT.as_posix()}\n"
        f"params: {n:,} ({n / 1e6:.3f} M)\n"
        "Note: model forward returns a dict; torchinfo top-level output shape "
        "may reflect an intermediate tensor. Param counts are authoritative.\n"
        "Claim boundary: Case A AF African t2m only (Cout=3) — not free-run / 10-day.\n\n"
        f"{s}\n"
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

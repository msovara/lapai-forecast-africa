#!/usr/bin/env python3
from pathlib import Path

ROOT = Path("/home/msovara/lustre/anemoi-weather-quest")
FILES = [
    "anemoi_train_common.sh",
    "train_jra3q_anemoi.pbs",
    "train_jra3q_anemoi_probe.pbs",
    "train_jra3q_anemoi_gpu_smoke.pbs",
    "train_jra3q_anemoi_gpu_short.pbs",
    "train_jra3q_anemoi_gpu_scale12.pbs",
    "scripts/probe_zarr_read.py",
]

BAD_MARKERS = ("tain_jra", "anemoi.taining", "msovaa@", "dataloade.", "taining.")

for name in FILES:
    path = ROOT / name
    text = path.read_text(encoding="utf-8")
    fixed = text.replace("\r\n", "\n").replace("\r", "\n")
    path.write_text(fixed, encoding="utf-8")
    if any(m in fixed for m in BAD_MARKERS):
        raise SystemExit(f"ERROR: corruption markers still in {path}")
    print(f"fixed: {path}")

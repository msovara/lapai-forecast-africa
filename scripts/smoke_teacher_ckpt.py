#!/usr/bin/env python3
"""Verify the AIFS Single v1.0 checkpoint file exists and is loadable via PyTorch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.teacher_yaml import parse_teacher_aifs_yaml_file, teacher_checkpoint_resolved  # noqa: E402


def repo_root() -> Path:
    return _REPO_ROOT


def main() -> int:
    p = argparse.ArgumentParser(description="Smoke-test AIFS teacher .ckpt on disk")
    p.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Path to teacher .ckpt (default: configs/teacher_aifs.yaml → local_checkpoint_relative)",
    )
    p.add_argument(
        "--max-state-keys",
        type=int,
        default=24,
        help="Print at most this many top-level state_dict keys (if present)",
    )
    args = p.parse_args()

    rr = repo_root()
    pins = parse_teacher_aifs_yaml_file(rr / "configs" / "teacher_aifs.yaml")
    if pins.get("huggingface_repo_id"):
        rev_disp = pins.get("revision") or "(not set in yaml)"
        print(
            "LapAI teacher yaml pins: "
            f"repo={pins.get('huggingface_repo_id')} "
            f"file={pins.get('checkpoint_filename', '?')} "
            f"revision={rev_disp}",
            file=sys.stderr,
        )

    ckpt = args.ckpt or teacher_checkpoint_resolved(rr)
    if ckpt is None:
        print("Could not resolve checkpoint path; pass --ckpt", file=sys.stderr)
        return 1
    ckpt = ckpt.resolve()
    if not ckpt.is_file():
        print(f"Missing file: {ckpt}\nRun: python scripts/download_teacher_ckpt.py", file=sys.stderr)
        return 1

    size_mb = ckpt.stat().st_size / (1024 * 1024)
    print(f"checkpoint: {ckpt}\nsize_MiB: {size_mb:.1f}")

    try:
        import torch
    except ImportError:
        print("torch not installed — file presence check only, OK.")
        return 0

    try:
        obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        print(f"torch.load failed: {exc}", file=sys.stderr)
        return 1

    print(f"torch.load type: {type(obj).__name__}")
    if isinstance(obj, dict):
        keys = list(obj.keys())[: args.max_state_keys]
        print("top_keys:", keys)
        sd = obj.get("state_dict")
        if isinstance(sd, dict):
            sk = list(sd.keys())[: args.max_state_keys]
            print("state_dict_sample:", sk)

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

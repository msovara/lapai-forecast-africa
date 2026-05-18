#!/usr/bin/env python3
"""Verify the AIFS Single v1.0 checkpoint file exists and is loadable via PyTorch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def checkpoint_from_teacher_yaml() -> Path | None:
    cfg = repo_root() / "configs" / "teacher_aifs.yaml"
    if not cfg.is_file():
        return None
    rel: str | None = None
    for line in cfg.read_text(encoding="utf-8").splitlines():
        s = line.split("#", 1)[0].strip()
        if s.startswith("local_checkpoint_relative:"):
            rel = s.split(":", 1)[1].strip().strip('"').strip("'")
            break
    if not rel:
        return None
    return (repo_root() / rel).resolve()


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

    ckpt = args.ckpt or checkpoint_from_teacher_yaml()
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

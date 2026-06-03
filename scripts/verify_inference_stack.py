#!/usr/bin/env python3
"""Lightweight readiness checks before `anemoi-inference run` on Lengau or a workstation."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.teacher_yaml import (  # noqa: E402
    parse_teacher_aifs_yaml_file,
    resolve_teacher_config_path,
)


def repo_root() -> Path:
    return _REPO_ROOT


def main() -> int:
    p = argparse.ArgumentParser(description="LapAI: verify inference prerequisites")
    p.add_argument(
        "--config",
        default=None,
        help="Teacher YAML (default: env LAPAI_TEACHER_CONFIG or configs/teacher_aifs.yaml)",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if teacher checkpoint missing or anemoi-inference not on PATH",
    )
    args = p.parse_args()

    rr = repo_root().resolve()
    ok = True
    print(f"repo_root: {rr}")

    cfg_path = resolve_teacher_config_path(rr, args.config)
    print(f"teacher_config: {cfg_path}")
    pins = parse_teacher_aifs_yaml_file(cfg_path)
    if pins.get("huggingface_repo_id"):
        print(
            "teacher_hf: "
            f"repo={pins.get('huggingface_repo_id')} "
            f"file={pins.get('checkpoint_filename', '?')} "
            f"revision={pins.get('revision', 'floating')}"
        )

    rel = pins.get("local_checkpoint_relative")
    if rel is None:
        print(f"teacher_ckpt_path: MISSING ({cfg_path} / local_checkpoint_relative)")
        if args.strict:
            ok = False
    else:
        ck = (rr / rel).resolve()
        if ck.is_file():
            print(f"teacher_ckpt_exists: OK — {ck} ({ck.stat().st_size / (1024**3):.2f} GiB)")
        else:
            print(f"teacher_ckpt_exists: MISSING — {ck}", file=sys.stderr)
            if args.strict:
                ok = False

    exe = shutil.which("anemoi-inference")
    print(f"anemoi-inference_path: {exe or 'MISSING'}")
    if exe is None and args.strict:
        ok = False

    for dist_name in ("anemoi-inference", "anemoi_inference"):
        try:
            print(f"anemoi-inference_pip_distribution: {importlib.metadata.version(dist_name)}")
            break
        except importlib.metadata.PackageNotFoundError:
            continue
    else:
        print("anemoi-inference_pip_distribution: NOT_INSTALLED")

    if importlib.util.find_spec("torch") is None:
        print("torch: NOT IMPORTABLE")
        if args.strict:
            ok = False
    else:
        import torch

        print(f"torch: {torch.__version__} cuda_available={torch.cuda.is_available()}")

    if importlib.util.find_spec("anemoi") is not None:
        try:
            importlib.import_module("anemoi.inference")
            print("anemoi.inference: import OK")
        except Exception as exc:  # noqa: BLE001
            print(f"anemoi.inference: import failed ({exc})", file=sys.stderr)
            # CLI may still work via console_scripts
    else:
        print("anemoi: package not found — install lapai-anemoi env")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

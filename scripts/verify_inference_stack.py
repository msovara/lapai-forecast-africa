#!/usr/bin/env python3
"""Lightweight readiness checks before `anemoi-inference run` on Lengau or a workstation."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _teacher_yaml_pins() -> dict[str, str]:
    """Parse a few colon keys from ``configs/teacher_aifs.yaml`` (best-effort, no YAML dep)."""
    cfg = repo_root() / "configs" / "teacher_aifs.yaml"
    out: dict[str, str] = {}
    if not cfg.is_file():
        return out
    watch = (
        "huggingface_repo_id",
        "checkpoint_filename",
        "revision",
        "local_checkpoint_relative",
    )
    for line in cfg.read_text(encoding="utf-8").splitlines():
        s = line.split("#", 1)[0].strip()
        for key in watch:
            prefix = f"{key}:"
            if s.startswith(prefix):
                val = s.split(":", 1)[1].strip().strip("\"'")
                if val and val.lower() not in ("null", "~", "none"):
                    out[key] = val
                break
    return out


def teacher_yaml_checkpoint_relative() -> str | None:
    pins = _teacher_yaml_pins()
    return pins.get("local_checkpoint_relative")


def main() -> int:
    p = argparse.ArgumentParser(description="LapAI: verify inference prerequisites")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if teacher checkpoint missing or anemoi-inference not on PATH",
    )
    args = p.parse_args()

    rr = repo_root().resolve()
    ok = True
    print(f"repo_root: {rr}")

    pins = _teacher_yaml_pins()
    if pins.get("huggingface_repo_id"):
        print(
            "teacher_hf: "
            f"repo={pins.get('huggingface_repo_id')} "
            f"file={pins.get('checkpoint_filename', '?')} "
            f"revision={pins.get('revision', 'floating')}"
        )

    rel = teacher_yaml_checkpoint_relative()
    if rel is None:
        print("teacher_ckpt_path: MISSING (configs/teacher_aifs.yaml / local_checkpoint_relative)")
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

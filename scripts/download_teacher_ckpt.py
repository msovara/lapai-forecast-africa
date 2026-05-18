#!/usr/bin/env python3
"""Download AIFS Single v1.0 teacher weights from Hugging Face into models/teacher/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _revision_from_teacher_yaml() -> str | None:
    """Return pinned HF revision from configs/teacher_aifs.yaml, or None if unset."""
    cfg = _repo_root() / "configs" / "teacher_aifs.yaml"
    if not cfg.is_file():
        return None
    for line in cfg.read_text(encoding="utf-8").splitlines():
        s = line.split("#", 1)[0].strip()
        if s.startswith("revision:"):
            raw = s.split(":", 1)[1].strip().strip("\"'")
            if not raw or raw.lower() in ("null", "~", "none"):
                return None
            return raw
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="Download ecmwf/aifs-single-1.0 checkpoint")
    p.add_argument(
        "--repo-id",
        default="ecmwf/aifs-single-1.0",
        help="Hugging Face model repo",
    )
    p.add_argument(
        "--filename",
        default="aifs-single-mse-1.0.ckpt",
        help="Checkpoint file name on the HF repo",
    )
    p.add_argument(
        "--revision",
        default=_revision_from_teacher_yaml(),
        help="HF revision (branch, tag, or commit). "
        "Defaults to configs/teacher_aifs.yaml revision; use empty string via --revision \"\" for floating main",
    )
    p.add_argument(
        "--local-dir",
        default="models/teacher",
        help="Directory to write the file into (created if missing)",
    )
    args = p.parse_args()

    revision = args.revision if args.revision else None

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "Install huggingface_hub:  pip install huggingface_hub",
            file=sys.stderr,
        )
        print(
            f"Or download manually from https://huggingface.co/{args.repo_id}/tree/main",
            file=sys.stderr,
        )
        return 1

    path = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        local_dir=args.local_dir,
        revision=revision,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

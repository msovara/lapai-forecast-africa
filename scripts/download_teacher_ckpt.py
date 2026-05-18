#!/usr/bin/env python3
"""Download AIFS Single v1.0 teacher weights from Hugging Face into models/teacher/."""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    p = argparse.ArgumentParser(description="Download ecmwf/aifs-single-1.0 checkpoint")
    p.add_argument(
        "--repo-id",
        default="ecmwf/aifs-single-1.0",
        help="Hugging Face model repo",
    )
    p.add_argument(
        "--filename",
        default="aifs_single_v1.0.ckpt",
        help="Checkpoint file name on the HF repo",
    )
    p.add_argument(
        "--revision",
        default=None,
        help="Optional HF revision (branch, tag, or commit)",
    )
    p.add_argument(
        "--local-dir",
        default="models/teacher",
        help="Directory to write the file into (created if missing)",
    )
    args = p.parse_args()

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
        revision=args.revision,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

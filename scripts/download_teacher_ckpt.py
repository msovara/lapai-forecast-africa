#!/usr/bin/env python3
"""Download AIFS Single v1.0 teacher weights from Hugging Face into models/teacher/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.teacher_yaml import parse_teacher_aifs_yaml_file  # noqa: E402


def _repo_root() -> Path:
    return _REPO_ROOT


def main() -> int:
    rr = _repo_root()

    p = argparse.ArgumentParser(description="Download a teacher checkpoint from Hugging Face")
    p.add_argument(
        "--config",
        default="configs/teacher_aifs.yaml",
        help="Teacher YAML providing default pins (repo-relative or absolute). "
        "Use configs/teacher_n320_gt6.yaml for the Code-for-Earth challenge teacher.",
    )
    p.add_argument(
        "--repo-id",
        default=None,
        help="Hugging Face model repo (default: huggingface_repo_id from --config)",
    )
    p.add_argument(
        "--filename",
        default=None,
        help="Checkpoint file name on the HF repo (default: checkpoint_filename from teacher yaml)",
    )
    p.add_argument(
        "--revision",
        default=None,
        help="HF revision (branch, tag, or commit). "
        "Omit to use revision from configs/teacher_aifs.yaml; use --revision \"\" for floating main",
    )
    p.add_argument(
        "--local-dir",
        default="models/teacher",
        help="Directory to write the file into (created if missing)",
    )
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = rr / cfg_path
    pins = parse_teacher_aifs_yaml_file(cfg_path)

    repo_id = args.repo_id or pins.get("huggingface_repo_id") or "ecmwf/aifs-single-1.0"
    filename = args.filename or pins.get("checkpoint_filename")
    if not filename:
        print(
            f"ERROR: no checkpoint_filename in {cfg_path} and --filename not given.\n"
            f"List files at https://huggingface.co/{repo_id}/tree/main (log in if gated) "
            "and pass --filename <name>.",
            file=sys.stderr,
        )
        return 2
    if args.revision is None:
        revision = pins.get("revision")
    else:
        rev = args.revision.strip() if isinstance(args.revision, str) else args.revision
        revision = rev if rev else None

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "Install huggingface_hub:  pip install huggingface_hub",
            file=sys.stderr,
        )
        print(
            f"Or download manually from https://huggingface.co/{repo_id}/tree/main",
            file=sys.stderr,
        )
        return 1

    rev_disp = revision if revision is not None else "floating main (HF default branch)"
    print(
        f"LapAI HF download: repo={repo_id} file={filename} revision={rev_disp} local_dir={args.local_dir}",
        file=sys.stderr,
    )

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=args.local_dir,
        revision=revision,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

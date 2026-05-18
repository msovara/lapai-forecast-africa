#!/usr/bin/env python3
"""Write an inference YAML with absolute checkpoint path; run `anemoi-inference run`."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_checkpoint_from_teacher_yaml() -> Path | None:
    cfg = repo_root() / "configs" / "teacher_aifs.yaml"
    if not cfg.is_file():
        return None
    rel: str | None = None
    for line in cfg.read_text(encoding="utf-8").splitlines():
        s = line.split("#", 1)[0].strip()
        if s.startswith("local_checkpoint_relative:"):
            rel = s.split(":", 1)[1].strip().strip("\"'")
            break
    if not rel:
        return None
    return (repo_root() / rel).resolve()


def substitute_checkpoint_yaml(text: str, ckpt_abs: Path) -> str:
    def repl(m: re.Match[str]) -> str:
        return f"{m.group(1)}{ckpt_abs.as_posix()}"

    out, n = re.subn(
        r"(^checkpoint:\s*)(.+?)\s*$",
        repl,
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise ValueError("Template needs exactly one top-level line starting with checkpoint:")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="LapAI: anemoi-inference launcher (teacher ckpt)")
    p.add_argument(
        "--template",
        type=Path,
        default=Path("configs/inference_aifs_minimal.yaml"),
    )
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument(
        "--anemoi-inference-binary",
        type=Path,
        default=None,
        help="Explicit anemoi-inference executable",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print rendered YAML + planned command line only",
    )
    args = p.parse_args()

    rr = repo_root()
    os.chdir(rr)

    template = args.template if args.template.is_absolute() else (rr / args.template).resolve()
    if not template.is_file():
        print(f"Missing template {template}", file=sys.stderr)
        return 1

    if args.checkpoint is not None:
        ck = args.checkpoint if args.checkpoint.is_absolute() else (rr / args.checkpoint).resolve()
    else:
        ck_maybe = read_checkpoint_from_teacher_yaml()
        ck = ck_maybe.resolve() if ck_maybe is not None else None

    if ck is None or not ck.is_file():
        print(
            "Teacher checkpoint missing. Run:\n  python scripts/download_teacher_ckpt.py\n"
            "or pass --checkpoint PATH",
            file=sys.stderr,
        )
        return 1

    try:
        rendered = substitute_checkpoint_yaml(template.read_text(encoding="utf-8"), ck)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.anemoi_inference_binary is None:
        which = shutil.which("anemoi-inference")
        exe_path = Path(which) if which else None
    else:
        exe_path = args.anemoi_inference_binary

    if exe_path is None or not exe_path.is_file():
        print(
            "Could not find `anemoi-inference` on PATH.\n"
            "  conda activate lapai-anemoi\n"
            "or pass --anemoi-inference-binary /path/to/anemoi-inference",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        print(rendered, end="")
        print("\n# would run roughly:\n# anemoi-inference run /tmp/lapai_aifs_inference_....yaml\n", file=sys.stderr)
        return 0

    fd, tmp = tempfile.mkstemp(prefix="lapai_aifs_inference_", suffix=".yaml", text=True)
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(rendered)
        cmd = [str(exe_path), "run", str(tmp_path)]
        print(" ".join(cmd), flush=True)
        return subprocess.call(cmd)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export single-page LapAI progress poster (HTML -> PDF + PNG) via Edge headless."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HTML = REPO / "reports" / "LAPAI_PROGRESS_POSTER.html"
PDF = REPO / "reports" / "LAPAI_PROGRESS_POSTER.pdf"
PNG = REPO / "reports" / "LAPAI_PROGRESS_POSTER.png"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


def main() -> int:
    if not HTML.is_file():
        print(f"Missing {HTML}", file=sys.stderr)
        return 1
    if not EDGE.is_file():
        print(f"Edge not found at {EDGE}", file=sys.stderr)
        return 1

    uri = HTML.resolve().as_uri()
    cmds = [
        [
            str(EDGE),
            "--headless=new",
            "--disable-gpu",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=5000",
            f"--print-to-pdf={PDF}",
            uri,
        ],
        [
            str(EDGE),
            "--headless=new",
            "--disable-gpu",
            "--window-size=1600,2400",
            f"--screenshot={PNG}",
            uri,
        ],
    ]
    for cmd in cmds:
        print("Running:", " ".join(cmd))
        proc = subprocess.run(cmd, cwd=str(REPO / "reports"), capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout, file=sys.stderr)
            print(proc.stderr, file=sys.stderr)
            return proc.returncode
    for path in (PDF, PNG):
        print(f"Wrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

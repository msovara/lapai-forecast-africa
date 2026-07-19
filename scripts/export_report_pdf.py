#!/usr/bin/env python3
"""Build Marp source from LAPAI_TRACKA_TEAM_REPORT.md and export PDF."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "reports" / "LAPAI_TRACKA_TEAM_REPORT.md"
MARP = REPO / "reports" / "LAPAI_TRACKA_TEAM_REPORT.marp.md"
PDF = REPO / "reports" / "LAPAI_TRACKA_TEAM_REPORT.pdf"

FRONTMATTER = """---
marp: true
theme: default
paginate: true
size: A4
style: |
  section { font-size: 22px; }
  section.lead h1 { font-size: 34px; }
  section h2 { color: #1565c0; }
  table { font-size: 17px; }
  blockquote { font-size: 20px; }
header: "LapAI-Forecast · Track A report"
footer: "July 2026 · CHPC Lengau"
---

<!-- _class: lead -->

"""


def build_marp_source(md_text: str) -> str:
    text = md_text.replace("\r\n", "\n")
    text = re.sub(r"\n---\n", "\n\n", text)
    parts = re.split(r"\n(?=## )", text, maxsplit=0)
    slides: list[str] = [FRONTMATTER + parts[0].strip() + "\n"]
    for part in parts[1:]:
        slides.append("\n---\n\n")
        slides.append(part.strip() + "\n")
    return "".join(slides)


def main() -> int:
    if not SRC.is_file():
        print(f"Missing {SRC}", file=sys.stderr)
        return 1
    marp_body = build_marp_source(SRC.read_text(encoding="utf-8"))
    MARP.write_text(marp_body, encoding="utf-8")

    cmd = [
        "npx",
        "--yes",
        "@marp-team/marp-cli",
        "--no-stdin",
        str(MARP),
        "--pdf",
        "-o",
        str(PDF),
    ]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(REPO / "reports"), capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        return proc.returncode
    print(f"Wrote {PDF} ({PDF.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

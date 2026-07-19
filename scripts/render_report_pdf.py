#!/usr/bin/env python3
"""Render LAPAI_TRACKA_TEAM_REPORT.md to a simple A4 PDF (fpdf2)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from fpdf import FPDF

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MD = REPO / "reports" / "LAPAI_TRACKA_TEAM_REPORT.md"
DEFAULT_PDF = REPO / "reports" / "LAPAI_TRACKA_TEAM_REPORT.pdf"


class ReportPDF(FPDF):
    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "LapAI-Forecast — Track A progress report", align="R", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Page {self.page_no()}/{{nb}}", align="C")


def _sanitize(text: str) -> str:
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    text = text.replace("\u2192", "->").replace("\u2264", "<=")
    text = text.replace("\u00b0", " deg")
    text = text.replace("\u00b7", " ")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _strip_md_inline(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    return text


def render(md_path: Path, pdf_path: Path) -> Path:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    pdf = ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 18, 18)
    pdf.add_page()

    body_w = pdf.w - pdf.l_margin - pdf.r_margin
    in_code = False
    table_rows: list[list[str]] = []

    def flush_table() -> None:
        nonlocal table_rows
        if not table_rows:
            return
        pdf.set_font("Courier", "", 7.5)
        for ri, row in enumerate(table_rows):
            if ri == 1 and all(re.match(r"^[-:\s|]+$", c.replace(" ", "")) for c in row):
                continue
            line = " | ".join(_strip_md_inline(c) for c in row)
            pdf.multi_cell(body_w, 4, _sanitize(line))
        table_rows = []
        pdf.ln(2)

    for raw in lines:
        line = raw.rstrip()

        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            pdf.set_font("Courier", "", 8)
            pdf.multi_cell(body_w, 4, _sanitize(line))
            continue

        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            table_rows.append(cells)
            continue
        flush_table()

        if not line.strip():
            pdf.ln(2)
            continue

        if line.startswith("# "):
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 16)
            pdf.multi_cell(body_w, 8, _sanitize(_strip_md_inline(line[2:].strip())))
            pdf.ln(2)
        elif line.startswith("## "):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(body_w, 7, _sanitize(_strip_md_inline(line[3:].strip())))
            pdf.ln(1)
        elif line.startswith("### "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(body_w, 6, _sanitize(_strip_md_inline(line[4:].strip())))
        elif line.startswith("- "):
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(body_w, 5, _sanitize("- " + _strip_md_inline(line[2:].strip())))
        elif line.startswith("---"):
            pdf.ln(2)
            pdf.set_draw_color(200, 200, 200)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(3)
        elif line.startswith("*") and line.endswith("*"):
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(80, 80, 80)
            pdf.multi_cell(body_w, 5, _sanitize(_strip_md_inline(line.strip("* ").strip())))
            pdf.set_text_color(0, 0, 0)
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(body_w, 5, _sanitize(_strip_md_inline(line)))

    flush_table()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(pdf_path))
    return pdf_path


def main() -> int:
    md = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MD
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PDF
    if not md.is_file():
        print(f"Missing markdown: {md}", file=sys.stderr)
        return 1
    path = render(md, out)
    print(f"Wrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

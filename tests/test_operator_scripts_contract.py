"""Static checks that operator-facing scripts retain expected cues (no LapAI deps required)."""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def test_lapai_inference_gate_err_trap_documents_runbook() -> None:
    text = (_REPO / "scripts" / "lapai_inference_gate.sh").read_text(encoding="utf-8")
    assert "trap " in text
    assert "RUNBOOK_BASELINE_LENGAU.md" in text


def test_smoke_teacher_ckpt_documents_yaml_pins() -> None:
    text = (_REPO / "scripts" / "smoke_teacher_ckpt.py").read_text(encoding="utf-8")
    assert "LapAI teacher yaml pins:" in text


def test_download_teacher_ckpt_documents_hf_summary() -> None:
    text = (_REPO / "scripts" / "download_teacher_ckpt.py").read_text(encoding="utf-8")
    assert "LapAI HF download:" in text


def test_run_aifs_inference_documents_template_stderr() -> None:
    text = (_REPO / "scripts" / "run_aifs_inference.py").read_text(encoding="utf-8")
    assert "LapAI dry-run template:" in text
    assert "LapAI inference template:" in text

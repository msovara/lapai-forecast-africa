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


def test_runbook_lists_minimal_lengau_path() -> None:
    text = (_REPO / "reports" / "RUNBOOK_BASELINE_LENGAU.md").read_text(encoding="utf-8")
    assert "Minimal copy-paste path" in text
    assert "lapai_inference_gate.sh" in text


def test_lengau_doc_links_runbook_minimal_path() -> None:
    text = (_REPO / "docs" / "LENGAU.md").read_text(encoding="utf-8")
    assert "RUNBOOK_BASELINE_LENGAU.md" in text
    assert "Minimal copy-paste path" in text


def test_requirements_txt_utf8_not_utf16_bom() -> None:
    raw = (_REPO / "requirements.txt").read_bytes()
    assert not raw.startswith(b"\xff\xfe") and not raw.startswith(b"\xfe\xff")
    assert b"numpy>=" in raw


def test_readme_documents_optional_requirements_install() -> None:
    text = (_REPO / "README.md").read_text(encoding="utf-8")
    assert "pip install -r requirements.txt" in text


def test_student_pbs_links_teacher_phase_doc() -> None:
    text = (_REPO / "pbs" / "student.pbs").read_text(encoding="utf-8")
    assert "RUNBOOK_BASELINE_LENGAU.md" in text


def test_lora_pbs_links_ops_docs() -> None:
    text = (_REPO / "pbs" / "lora.pbs").read_text(encoding="utf-8")
    assert "LENGAU.md" in text

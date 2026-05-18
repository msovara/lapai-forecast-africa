"""Smoke tests for verify_inference_stack CLI (no teacher weights required)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def test_verify_inference_stack_prints_hf_pin():
    proc = subprocess.run(
        [sys.executable, "scripts/verify_inference_stack.py"],
        cwd=_REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    out = proc.stdout
    assert "ecmwf/aifs-single-1.0" in out
    assert "teacher_hf:" in out
    assert "aifs-single-mse-1.0.ckpt" in out
    assert "f0bb02c" in out
    assert "anemoi-inference_pip_distribution:" in out

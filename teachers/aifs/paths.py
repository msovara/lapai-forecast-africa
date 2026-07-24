"""Paths for the AIFS (Anemoi n320_gt6) teacher artefacts."""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

TEACHER_CKPT = _REPO / "models" / "teacher_n320_gt6" / "inference.ckpt"
PHASE0_FORECAST_DIR = _REPO / "data" / "processed" / "phase0" / "forecasts"
PHASE0_SCORECARD = _REPO / "reports" / "PHASE0_BASELINE_SCORECARD.json"

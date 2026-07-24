"""LapAI student pathways (Track A coarsened O96 and beyond)."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKA_FORECAST_DIR = _REPO_ROOT / "data" / "processed" / "trackA" / "forecasts"
TRACKA_COARSEN_CFG = _REPO_ROOT / "configs" / "trackA_coarsen.yaml"
TRACKA_A1_GATE = _REPO_ROOT / "reports" / "TRACKA_A1_GATE.json"

__all__ = ["TRACKA_FORECAST_DIR", "TRACKA_COARSEN_CFG", "TRACKA_A1_GATE"]

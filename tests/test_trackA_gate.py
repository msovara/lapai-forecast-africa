"""Track A coarsen gate tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.trackA_gate import compare_scorecards, run_coarsen_gate


def _scorecard(rmse: float, init: str = "20230101", lead: int = 24) -> dict:
    return {
        "results": [
            {
                "init_date": init,
                "lead_hours": lead,
                "variables": {
                    "t2m": {"rmse": rmse, "acc": 0.9},
                    "u10": {"rmse": 3.0, "acc": 0.1},
                    "v10": {"rmse": 3.0, "acc": 0.1},
                },
            }
        ]
    }


def test_gate_passes_within_threshold():
    base = _scorecard(2.0)
    cand = _scorecard(2.05)
    report = compare_scorecards(base, cand, variables=["t2m"], leads_hours=[24], max_rmse_degradation_pct=5.0)
    assert report["passed"] is True
    assert report["checks"][0]["degradation_pct"] == pytest.approx(2.5, abs=0.01)


def test_gate_fails_beyond_threshold():
    base = _scorecard(2.0)
    cand = _scorecard(2.5)
    report = compare_scorecards(base, cand, variables=["t2m"], leads_hours=[24], max_rmse_degradation_pct=5.0)
    assert report["passed"] is False


def test_run_coarsen_gate_with_config(tmp_path):
    baseline = tmp_path / "base.json"
    candidate = tmp_path / "cand.json"
    baseline.write_text(json.dumps(_scorecard(2.0)), encoding="utf-8")
    candidate.write_text(json.dumps(_scorecard(2.04)), encoding="utf-8")
    cfg = {
        "gate": {
            "baseline_scorecard": str(baseline),
            "leads_hours": [24],
            "variables": ["t2m"],
            "max_rmse_degradation_pct": 5.0,
        }
    }
    report = run_coarsen_gate(candidate, baseline_scorecard=baseline, config=cfg)
    assert report["passed"] is True

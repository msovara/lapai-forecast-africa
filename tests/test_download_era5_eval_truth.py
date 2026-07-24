"""Unit tests for ERA5 eval-truth CDS request builder (no network)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _load_module():
    path = _REPO / "scripts" / "download_era5_eval_truth.py"
    spec = importlib.util.spec_from_file_location("download_era5_eval_truth", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_cds_area_africa():
    mod = _load_module()
    area = mod._cds_area_nws_e(((-40.0, 40.0), (-20.0, 70.0)))
    assert area == [40.0, -20.0, -40.0, 70.0]


def test_build_cds_request_maps_variables():
    mod = _load_module()
    req = mod.build_cds_request(
        2024,
        ["t2m", "tp", "u10", "v10"],
        [40.0, -20.0, -40.0, 70.0],
        {"cds": {"times": ["00:00", "06:00", "12:00", "18:00"]}},
    )
    assert req["year"] == "2024"
    assert "2m_temperature" in req["variable"]
    assert "total_precipitation" in req["variable"]
    assert req["area"] == [40.0, -20.0, -40.0, 70.0]
    assert req["time"] == ["00:00", "06:00", "12:00", "18:00"]


def test_parse_years_from_eval_period():
    mod = _load_module()
    years = mod._parse_years(None, {"test_period": {"start": "2023-01-01", "end": "2025-12-31"}})
    assert years == [2023, 2024, 2025]

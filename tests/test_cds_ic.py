"""CDS IC cache key tests (no network)."""

from __future__ import annotations

from pathlib import Path

from utils.cds_ic import cache_path, list_required_cache_keys


def test_cache_path_is_deterministic(tmp_path: Path) -> None:
    req = {"dataset": "reanalysis-era5-single-levels", "date": "20230101", "time": "0000"}
    p1 = cache_path(req, tmp_path)
    p2 = cache_path(req, tmp_path)
    assert p1 == p2
    assert p1.name.endswith(".grib2")


def test_phase0_inits_expect_twenty_cache_files() -> None:
    inits = ["20230101", "20230108"]
    paths = list_required_cache_keys(inits)
    assert len(paths) == 8  # 2 inits × 2 times × 2 datasets

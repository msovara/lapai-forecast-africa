"""Offline unit tests for GraphCast Africa scorecard helpers."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.graphcast_africa_scorecard import _parse_init, _valid_time, load_graphcast_score_config


def test_parse_init_formats():
    a = _parse_init("20220115")
    b = _parse_init("2022-01-15")
    assert a == b
    assert str(a)[:10] == "2022-01-15"


def test_valid_time_offset():
    init = _parse_init("20220101")
    assert _valid_time(init, 24) == np.datetime64("2022-01-02T00:00:00")


def test_load_default_config():
    cfg = load_graphcast_score_config()
    assert cfg.get("pathway") == "graphcast_africa"
    assert "20220101" in cfg.get("inits", [])
    assert 24 in cfg.get("leads_hours", [])

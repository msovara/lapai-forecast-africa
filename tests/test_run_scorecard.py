"""Tests for the Mvua multi-variable scorecard driver."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from evaluation.run_scorecard import (
    _aggregate_daily,
    _read_eval_config,
    _split_csv,
    _to_tensors,
)


def test_split_csv():
    assert _split_csv(None) is None
    assert _split_csv("") is None
    assert _split_csv("t2m, tp ,u10") == ["t2m", "tp", "u10"]


def test_read_eval_config_lists(tmp_path):
    cfg = tmp_path / "eval.yaml"
    cfg.write_text(
        "ground_truth: era5\n"
        "variables: [t2m, tp, u10, v10]\n"
        "temporal_resolutions: [6h, daily]\n",
        encoding="utf-8",
    )
    parsed = _read_eval_config(cfg)
    assert parsed["ground_truth"] == "era5"
    assert parsed["variables"] == ["t2m", "tp", "u10", "v10"]
    assert parsed["temporal_resolutions"] == ["6h", "daily"]


def _make_da(values, var):  # type: ignore[no-untyped-def]
    xr = pytest.importorskip("xarray")
    times = np.array(
        [np.datetime64("2023-01-01T00") + np.timedelta64(6 * i, "h") for i in range(values.shape[0])]
    )
    lat = np.linspace(-10.0, 10.0, values.shape[1])
    lon = np.linspace(0.0, 30.0, values.shape[2])
    return xr.DataArray(
        values,
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": lat, "longitude": lon},
        name=var,
    )


def test_aggregate_daily_tp_sums_others_mean():
    pytest.importorskip("xarray")
    # 8 six-hourly steps = 2 days, 2x3 grid of ones
    vals = np.ones((8, 2, 3), dtype=np.float32)
    tp = _make_da(vals, "tp")
    t2m = _make_da(vals, "t2m")

    tp_daily = _aggregate_daily(tp, "tp")
    t2m_daily = _aggregate_daily(t2m, "t2m")

    assert tp_daily.sizes["time"] == 2
    # precipitation accumulates: 4 six-hourly ones per day -> 4
    assert float(tp_daily.isel(time=0).mean()) == pytest.approx(4.0)
    # temperature averages -> 1
    assert float(t2m_daily.isel(time=0).mean()) == pytest.approx(1.0)


def test_to_tensors_shapes_and_weights():
    pytest.importorskip("xarray")
    vals = np.random.default_rng(0).standard_normal((3, 5, 7)).astype(np.float32)
    ap = _make_da(vals, "t2m")
    at = _make_da(vals.copy(), "t2m")
    pv, tv, w = _to_tensors(ap, at)
    assert pv.shape == (3, 1, 5, 7)
    assert tv.shape == (3, 1, 5, 7)
    assert w.shape == (5,)
    # identical fields -> zero RMSE through the metric
    from evaluation.eval_skill import compute_skill_metrics

    m = compute_skill_metrics(pv, tv, w)
    assert m["rmse"] == 0.0

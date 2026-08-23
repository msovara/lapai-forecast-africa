"""Tests for the Mvua multi-variable scorecard driver."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from evaluation.run_scorecard import (
    _aggregate_daily,
    _crop_domain,
    _read_eval_config,
    _resolve_domain,
    _split_csv,
    _to_tensors,
)

_REPO = Path(__file__).resolve().parents[1]

_AFRICA_CFG = {
    "domain": {
        "default": "africa",
        "regions": {
            "africa": {"lat": [-40.0, 40.0], "lon": [-20.0, 70.0]},
            "global": {"lat": [-90.0, 90.0], "lon": [-180.0, 180.0]},
        },
    }
}


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


def test_resolve_domain_default_and_global():
    name, lat_r, lon_r = _resolve_domain(_AFRICA_CFG, None)
    assert name == "africa"
    assert lat_r == (-40.0, 40.0)
    assert lon_r == (-20.0, 70.0)
    # global is an explicit no-op (full grid, no crop)
    assert _resolve_domain(_AFRICA_CFG, "global") is None
    # no domain config -> no crop
    assert _resolve_domain({}, None) is None


def test_eval_yaml_africa_is_original_trackA_box():
    cfg = _read_eval_config(_REPO / "configs" / "eval.yaml")
    name, lat_r, lon_r = _resolve_domain(cfg, None)
    assert name == "africa"
    assert lat_r == (-40.0, 40.0)
    assert lon_r == (-20.0, 55.0)


def test_crop_domain_handles_0_360_longitude():
    xr = pytest.importorskip("xarray")
    lat = np.linspace(-90.0, 90.0, 73)  # 2.5 deg
    lon = np.linspace(0.0, 357.5, 144)  # 0..360 convention
    da = xr.DataArray(
        np.zeros((lat.size, lon.size), dtype=np.float32),
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
    )
    cropped = _crop_domain(da, (-40.0, 40.0), (-20.0, 70.0))
    # latitude trimmed into the box
    assert float(cropped.latitude.min()) >= -40.0
    assert float(cropped.latitude.max()) <= 40.0
    # longitude wraps: -20..70 maps to 340..360 and 0..70 in the 0..360 grid
    lon180 = ((cropped.longitude.values + 180) % 360) - 180
    assert lon180.min() >= -20.0
    assert lon180.max() <= 70.0
    assert cropped.longitude.size < da.longitude.size


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

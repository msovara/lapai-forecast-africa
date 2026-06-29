"""Unit tests for GCS ERA5 truth loader (no network)."""

from __future__ import annotations

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from evaluation.gcs_era5_truth import _select_tp_at_valid_time, align_truth_to_pred


def test_align_truth_sorts_descending_latitude():
    lat = np.arange(-40.0, 40.25, 0.25)
    lon = np.arange(-20.0, 55.25, 0.25)
    pred = xr.DataArray(
        np.full((lat.size, lon.size), 280.0),
        coords={"latitude": lat, "longitude": lon},
        dims=("latitude", "longitude"),
    )
    truth = xr.DataArray(
        np.linspace(270.0, 290.0, lat.size)[:, None] * np.ones((lat.size, lon.size)),
        coords={"latitude": lat[::-1], "longitude": lon},
        dims=("latitude", "longitude"),
    )
    aligned = align_truth_to_pred(truth, pred)
    assert aligned.shape == pred.shape
    # truth at -40 N is 290 K; at 40 N is 270 K once latitude order is normalised
    assert float(aligned.isel(latitude=0, longitude=0)) == pytest.approx(290.0)
    assert float(aligned.isel(latitude=-1, longitude=0)) == pytest.approx(270.0)


def test_select_tp_at_valid_time_hour_offset():
    ref = np.datetime64("2023-01-01T12:00:00")
    times = np.array([ref], dtype="datetime64[ns]")
    steps = np.arange(1, 13)
    data = np.zeros((1, steps.size, 2, 2), dtype=np.float32)
    data[0, 11, :, :] = 0.42  # step=12 -> valid 2023-01-02 00Z
    da = xr.DataArray(
        data,
        coords={"time": times, "step": steps, "latitude": [0, 1], "longitude": [0, 1]},
        dims=("time", "step", "latitude", "longitude"),
    )
    picked = _select_tp_at_valid_time(da, np.datetime64("2023-01-02T00:00:00"))
    assert picked.shape == (2, 2)
    assert float(picked.mean()) == pytest.approx(0.42)

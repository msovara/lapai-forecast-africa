"""Offline unit tests for GraphCast Africa forecast selectors."""

from __future__ import annotations

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from teachers.graphcast.gcs_forecasts import select_init_lead


def _synthetic_forecast():
    times = np.array(
        [
            np.datetime64("2020-01-01T00:00:00"),
            np.datetime64("2020-01-02T00:00:00"),
            np.datetime64("2020-01-03T00:00:00"),
        ],
        dtype="datetime64[ns]",
    )
    leads = np.array([6, 24, 48], dtype="timedelta64[h]")
    lat = np.linspace(-40.0, 40.0, 5)
    lon = np.linspace(-20.0, 70.0, 7)
    data = np.full((3, 3, 5, 7), 290.0, dtype=np.float32)
    data[0, :, :, :] = np.nan  # hole init
    return xr.DataArray(
        data,
        dims=("time", "prediction_timedelta", "latitude", "longitude"),
        coords={
            "time": times,
            "prediction_timedelta": leads,
            "latitude": lat,
            "longitude": lon,
        },
        name="2m_temperature",
    )


def test_select_init_lead_ok():
    da = _synthetic_forecast()
    out = select_init_lead(da, init="2020-01-02", lead_hours=24)
    assert out.ndim == 2
    assert float(out.mean()) == pytest.approx(290.0)
    assert float(out.longitude.max()) == pytest.approx(70.0)


def test_select_init_lead_rejects_nan_hole():
    da = _synthetic_forecast()
    with pytest.raises(ValueError, match="all-NaN"):
        select_init_lead(da, init="2020-01-01", lead_hours=24)

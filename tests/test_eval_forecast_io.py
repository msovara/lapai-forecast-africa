"""Tests for eval NetCDF writer."""

from __future__ import annotations

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from utils.eval_forecast_io import write_eval_regridded_netcdf


def test_write_eval_regridded_netcdf_preserves_step_differences(monkeypatch, tmp_path):
    lat = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    lon = np.array([0.0, 1.0], dtype=np.float32)

    def fake_regrid(fields, *, variables, cache_root=None, domain="africa"):
        del cache_root, domain
        out = {}
        for eval_name in variables:
            for model_name, alias in (("2t", "t2m"), ("tp", "tp"), ("10u", "u10"), ("10v", "v10")):
                if alias != eval_name or model_name not in fields:
                    continue
                val = float(np.mean(fields[model_name]))
                out[eval_name] = np.full((lat.size, lon.size), val, dtype=np.float32)
        return out, lat, lon

    monkeypatch.setattr(
        "utils.eval_forecast_io.regrid_state_fields_to_africa",
        fake_regrid,
    )

    states = [
        {"date": np.datetime64("2023-01-01T06:00:00"), "fields": {"2t": np.array([280.0])}},
        {"date": np.datetime64("2023-01-01T12:00:00"), "fields": {"2t": np.array([285.0])}},
    ]
    out = tmp_path / "fc.nc"
    write_eval_regridded_netcdf(states, out, reference_date=__import__("datetime").datetime(2023, 1, 1))

    ds = xr.open_dataset(out)
    assert not np.allclose(ds["t2m"].isel(time=0).values, ds["t2m"].isel(time=1).values)
    assert float(ds["t2m"].isel(time=0).mean()) == pytest.approx(280.0)
    assert float(ds["t2m"].isel(time=1).mean()) == pytest.approx(285.0)

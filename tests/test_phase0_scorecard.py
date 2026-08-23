"""Phase 0 scorecard unit tests (local truth; no GCS required)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]


def _make_eval_forecast(seed: int, init: str = "20230101"):
    xr = pytest.importorskip("xarray")
    rng = np.random.default_rng(seed)
    times = np.array(
        [np.datetime64(f"{init[:4]}-{init[4:6]}-{init[6:8]}T00") + np.timedelta64(6 * i, "h") for i in range(40)]
    )
    lat = np.arange(-40.0, 40.25, 0.25)
    lon = np.arange(-20.0, 70.25, 0.25)
    shape = (times.size, lat.size, lon.size)
    return xr.Dataset(
        {
            "t2m": (("time", "latitude", "longitude"), (280.0 + rng.standard_normal(shape)).astype("float32")),
            "tp": (("time", "latitude", "longitude"), np.abs(rng.standard_normal(shape) * 0.001).astype("float32")),
            "u10": (("time", "latitude", "longitude"), rng.standard_normal(shape).astype("float32")),
            "v10": (("time", "latitude", "longitude"), rng.standard_normal(shape).astype("float32")),
        },
        coords={"time": times, "latitude": lat, "longitude": lon},
    )


def test_load_phase0_config():
    from evaluation.phase0_scorecard import load_phase0_config

    cfg = load_phase0_config(_REPO / "configs" / "phase0_baseline.yaml")
    assert cfg.get("phase") == 0
    assert "20230101" in cfg.get("inits", [])
    assert "t2m" in cfg.get("variables", [])


def test_phase0_scorecard_local_identical(tmp_path):
    pytest.importorskip("xarray")
    from evaluation.phase0_scorecard import build_phase0_scorecard, write_phase0_scorecard

    fc = _make_eval_forecast(0)
    pred = tmp_path / "20230101_00Z.nc"
    truth = tmp_path / "era5_truth.zarr"
    fc.to_netcdf(pred)
    fc.to_zarr(truth, mode="w")

    cfg = {
        "variables": ["t2m", "tp"],
        "leads_hours": [24],
        "domain": "africa",
        "truth": {"source": "local", "local_zarr": str(truth)},
    }
    sc = build_phase0_scorecard([pred], config=cfg, truth_source="local")
    out = tmp_path / "scorecard.json"
    write_phase0_scorecard(sc, out)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["phase"] == 0
    assert len(payload["results"]) == 1
    t2m = payload["results"][0]["variables"]["t2m"]
    assert t2m["rmse"] == pytest.approx(0.0, abs=1e-5)


def test_subset_eval_domain_crops_a2_lon70_to_a1():
    pytest.importorskip("xarray")
    import xarray as xr

    from evaluation.phase0_scorecard import _eval_africa_box, _subset_eval_domain

    lat = np.arange(-40.0, 40.25, 0.25)
    lon = np.arange(-20.0, 70.25, 0.25)
    da = xr.DataArray(
        np.zeros((lat.size, lon.size), dtype=np.float32),
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
    )
    assert da.sizes["longitude"] == 361
    box = _eval_africa_box()
    assert box is not None
    cropped = _subset_eval_domain(da, box[0], box[1])
    assert cropped.sizes["latitude"] == 321
    assert cropped.sizes["longitude"] == 301
    assert float(cropped.longitude.min()) == pytest.approx(-20.0)
    assert float(cropped.longitude.max()) == pytest.approx(55.0)


def test_eval_skill_alias_2t():
    pytest.importorskip("xarray")
    import xarray as xr

    from evaluation.eval_skill import _dataarray_from_dataset

    ds = xr.Dataset({"2t": (("latitude", "longitude"), np.zeros((3, 4)))})
    da = _dataarray_from_dataset(ds, "t2m")
    assert da.name == "2t"


def test_run_phase0_closure_dry_run(capsys):
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, str(_REPO / "scripts" / "run_phase0_closure.py"), "--dry-run"],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "20230101" in proc.stdout or "20230101" in proc.stderr

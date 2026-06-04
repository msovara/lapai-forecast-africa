"""End-to-end smoke test for the scorecard driver over synthetic Zarr stores.

Exercises the real CLI path: open Zarr -> Africa-domain crop -> 6h/daily
aggregation -> RMSE/ACC -> JSON. Requires the [data] extra (xarray + zarr).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]


def _make_dataset(seed: int):  # type: ignore[no-untyped-def]
    xr = pytest.importorskip("xarray")
    rng = np.random.default_rng(seed)
    times = np.array(
        [np.datetime64("2023-01-01T00") + np.timedelta64(6 * i, "h") for i in range(8)]
    )
    lat = np.linspace(-50.0, 50.0, 21)  # spans beyond the Africa box -> crop has work to do
    lon = np.linspace(-30.0, 60.0, 19)
    shape = (times.size, lat.size, lon.size)
    return xr.Dataset(
        {
            "t2m": (("time", "latitude", "longitude"), (280.0 + rng.standard_normal(shape)).astype("float32")),
            "tp": (("time", "latitude", "longitude"), np.abs(rng.standard_normal(shape)).astype("float32")),
        },
        coords={"time": times, "latitude": lat, "longitude": lon},
    )


def test_scorecard_e2e_identical_zarr(tmp_path):
    pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    ds = _make_dataset(0)
    pred = tmp_path / "pred.zarr"
    truth = tmp_path / "truth.zarr"
    ds.to_zarr(pred, mode="w")
    ds.to_zarr(truth, mode="w")  # identical fields -> RMSE 0

    out = tmp_path / "scorecard.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.run_scorecard",
            "--pred", str(pred),
            "--truth", str(truth),
            "--variables", "t2m,tp",
            "--temporal", "6h,daily",
            "--domain", "africa",
            "--out", str(out),
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    sc = json.loads(out.read_text(encoding="utf-8"))
    assert sc["domain"] == "africa"
    rows = sc["results"]
    assert len(rows) == 4  # 2 variables x {6h, daily}
    for r in rows:
        assert "rmse" in r, r
        assert r["rmse"] == pytest.approx(0.0, abs=1e-5)
        assert r["n_times"] >= 1

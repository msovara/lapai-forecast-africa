"""Tests for cosine-latitude RMSE / ACC."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from evaluation.eval_skill import compute_skill_metrics
from lapai_inference.cache_schema import cosine_latitude_weights, lat_lon_mesh

_REPO = Path(__file__).resolve().parents[1]


def test_compute_skill_identical_arrays():
    pred = torch.randn(2, 1, 13, 24)
    era5 = pred.clone()
    lat, _ = lat_lon_mesh(pred.shape[-2], pred.shape[-1])
    w = torch.from_numpy(cosine_latitude_weights(lat).squeeze()).float()
    m = compute_skill_metrics(pred, era5, w)
    assert m["rmse"] == 0.0
    assert abs(m["acc"] - 1.0) < 1e-5


def test_blob_subprocess_json(tmp_path):
    blob = tmp_path / "slice.pt"
    pred = torch.ones(1, 1, 181, 360) * 0.5
    era5 = torch.ones_like(pred) * (-0.5)
    torch.save({"pred": pred, "era5": era5}, blob)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.eval_skill",
            "--blob",
            str(blob),
        ],
        cwd=_REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    lat, _ = lat_lon_mesh(pred.shape[-2], pred.shape[-1])
    w = torch.from_numpy(cosine_latitude_weights(lat).squeeze()).float()
    lib = compute_skill_metrics(pred, era5, w)
    cli = json.loads(proc.stdout.strip())
    assert abs(cli["rmse"] - lib["rmse"]) < 1e-5
    assert abs(cli["acc"] - lib["acc"]) < 1e-5


def test_parse_isel_arg():
    from evaluation.eval_skill import _parse_isel_arg

    assert _parse_isel_arg("") == {}
    assert _parse_isel_arg(None) == {}
    assert _parse_isel_arg("time=0,step=-1") == {"time": 0, "step": -1}


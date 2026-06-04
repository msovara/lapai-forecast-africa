"""Tests for the feature-attribution scaffold (pure-torch paths; no captum needed)."""

from __future__ import annotations

import torch

from evaluation.attribution_shap import (
    _demo_model,
    collapse_level_suffix,
    group_importances,
    per_channel_abs_importance,
    saliency_attributions,
)


def test_collapse_level_suffix():
    assert collapse_level_suffix("t_850") == "t"
    assert collapse_level_suffix("z_500") == "z"
    assert collapse_level_suffix("t2m") == "t2m"
    assert collapse_level_suffix("u10") == "u10"


def test_per_channel_abs_importance_shape():
    attr = torch.ones(2, 4, 3, 5)
    scores = per_channel_abs_importance(attr)
    assert scores.shape == (4,)
    assert torch.allclose(scores, torch.full((4,), 2.0 * 3 * 5))


def test_group_importances_collapse_levels():
    scores = torch.tensor([1.0, 2.0, 3.0])
    names = ["t_850", "t_500", "u10"]
    grouped = group_importances(scores, names, collapse_levels=True)
    assert grouped == {"t": 3.0, "u10": 3.0}


def test_saliency_runs_on_demo_model():
    model = _demo_model(3)
    x = torch.randn(1, 3, 8, 8)
    attr = saliency_attributions(model, x)
    assert attr.shape == x.shape
    # at least some sensitivity to the input
    assert float(attr.abs().sum()) > 0.0

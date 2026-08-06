#!/usr/bin/env python3
"""Unit tests for soft attention-head masking (no Anemoi install required)."""

from __future__ import annotations

import torch

from utils.head_prune import (
    apply_soft_head_mask,
    assert_has_attention_heads,
    discover_gt_lin_keys,
    score_heads_weight_l1,
    select_heads_to_drop,
)


def _fake_gt_state(num_heads: int = 4, head_dim: int = 8, in_ch: int = 16) -> dict[str, torch.Tensor]:
    out = num_heads * head_dim
    prefix = "model.processor.proc.0."
    state = {}
    for lin in ("lin_query", "lin_key", "lin_value", "lin_self", "lin_edge"):
        w = torch.randn(out, in_ch)
        # make head 0 uniquely small so it is selected for drop
        w[0:head_dim].mul_(0.01)
        state[f"{prefix}{lin}.weight"] = w
    state[f"{prefix}projection.weight"] = torch.randn(in_ch, out)
    return state


def test_discover_and_score_drop_lowest_head():
    state = _fake_gt_state()
    assert_has_attention_heads(state, scope="processor")
    blocks = discover_gt_lin_keys(state, scope="processor")
    assert len(blocks) == 1
    scores = score_heads_weight_l1(state, num_heads=4, scope="processor")
    drop = select_heads_to_drop(scores, fraction=0.25, num_heads=4)
    prefix = next(iter(drop))
    assert drop[prefix][0] == 0  # head 0 was scaled down


def test_apply_soft_mask_zeros_slices():
    state = _fake_gt_state()
    prefix = "model.processor.proc.0."
    apply_soft_head_mask(state, {prefix: [0]}, num_heads=4)
    w = state[f"{prefix}lin_query.weight"]
    assert float(w[0:8].abs().sum()) == 0.0
    assert float(w[8:16].abs().sum()) > 0.0
    proj = state[f"{prefix}projection.weight"]
    assert float(proj[:, 0:8].abs().sum()) == 0.0


def test_gnn_like_state_rejected():
    state = {"model.processor.proc.0.mlp.0.weight": torch.randn(8, 8)}
    try:
        assert_has_attention_heads(state)
        raised = False
    except RuntimeError:
        raised = True
    assert raised

#!/usr/bin/env python3
"""Feature-attribution scaffold for LapAI models (saliency now, SHAP-ready).

Why: deciding *which input variables / pressure levels / regions* drive a forecast
(esp. precipitation for the Mvua team) and which attention channels matter least for
Track A pruning. This explains model behaviour; it does NOT measure forecast skill
(use evaluation/run_scorecard.py for skill vs ERA5).

Two attribution backends:
  * saliency  — pure torch |d(target)/d(input)|, always available (default).
  * shap      — captum GradientShap (pip install captum), richer baseline-referenced
                attributions, used once a runnable model checkpoint exists.

Quick check (no checkpoint needed):
  python -m evaluation.attribution_shap --demo
Real use:
  python -m evaluation.attribution_shap --checkpoint model.pt --input sample.pt \\
      --channel-names channels.txt --method shap --collapse-levels
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

_LEVEL_SUFFIX = re.compile(r"^([A-Za-z]+)_\d+$")


def collapse_level_suffix(name: str) -> str:
    """`t_850` -> `t` (group pressure-level channels by variable); `t2m`/`u10` unchanged."""
    m = _LEVEL_SUFFIX.match(name)
    return m.group(1) if m else name


def per_channel_abs_importance(attr: torch.Tensor) -> torch.Tensor:
    """[N, C, H, W] attribution -> [C] summed absolute importance over batch + grid."""
    if attr.dim() != 4:
        raise ValueError(f"expected [N,C,H,W] attribution, got shape {tuple(attr.shape)}")
    return attr.abs().sum(dim=(0, 2, 3))


def group_importances(
    channel_scores: torch.Tensor,
    channel_names: list[str],
    collapse_levels: bool = False,
) -> dict[str, float]:
    """Sum per-channel scores into named groups (optionally collapsing `_<level>`)."""
    scores = channel_scores.tolist()
    if len(scores) != len(channel_names):
        raise ValueError(f"{len(scores)} scores vs {len(channel_names)} channel names")
    out: dict[str, float] = {}
    for score, name in zip(scores, channel_names):
        key = collapse_level_suffix(name) if collapse_levels else name
        out[key] = out.get(key, 0.0) + float(score)
    return out


def saliency_attributions(model: torch.nn.Module, x: torch.Tensor, scalar_target_fn=None) -> torch.Tensor:
    """Pure-torch saliency: gradient of a scalar target wrt the input field."""
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    out = model(x)
    target = scalar_target_fn(out) if scalar_target_fn is not None else out.mean()
    model.zero_grad(set_to_none=True)
    target.sum().backward()
    if x.grad is None:
        raise RuntimeError("input received no gradient; is the model differentiable wrt x?")
    return x.grad.detach()


def shap_attributions(
    model: torch.nn.Module,
    x: torch.Tensor,
    baselines: torch.Tensor | None = None,
    scalar_target_fn=None,
) -> torch.Tensor:
    """captum GradientShap over a scalarised model output. Requires `pip install captum`."""
    try:
        from captum.attr import GradientShap
    except ImportError as exc:  # noqa: BLE001
        raise SystemExit(
            "SHAP backend needs captum:  pip install captum\n"
            "Or use --method saliency (pure torch, no extra deps)."
        ) from exc

    class _Scalarised(torch.nn.Module):
        def __init__(self, m: torch.nn.Module, fn) -> None:
            super().__init__()
            self.m = m
            self.fn = fn

        def forward(self, inp: torch.Tensor) -> torch.Tensor:
            out = self.m(inp)
            if self.fn is not None:
                return self.fn(out).reshape(-1)
            return out.mean(dim=tuple(range(1, out.dim()))).reshape(-1)

    wrapped = _Scalarised(model, scalar_target_fn)
    if baselines is None:
        baselines = torch.zeros_like(x)
    return GradientShap(wrapped).attribute(x, baselines=baselines)


def _demo_model(channels: int) -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Conv2d(channels, 4, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(4, channels, kernel_size=3, padding=1),
    )


def _load_module(path: Path) -> torch.nn.Module:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.nn.Module):
        return obj
    raise SystemExit(
        f"{path} did not contain an nn.Module. Save a (scripted) module, "
        "or wire your own loader for the LapAI student/teacher."
    )


def _read_channel_names(path: Path | None, channels: int) -> list[str]:
    if path is not None and path.is_file():
        names = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(names) != channels:
            raise SystemExit(f"channel-names has {len(names)} entries, model expects {channels}")
        return names
    return [f"c{i}" for i in range(channels)]


def main() -> int:
    p = argparse.ArgumentParser(description="LapAI feature attribution (saliency / SHAP)")
    p.add_argument("--checkpoint", type=Path, default=None, help="torch-saved nn.Module")
    p.add_argument("--input", type=Path, default=None, help="torch blob with key 'x' = [N,C,H,W]")
    p.add_argument("--channel-names", type=Path, default=None, help="one channel name per line")
    p.add_argument("--method", choices=["saliency", "shap"], default="saliency")
    p.add_argument("--collapse-levels", action="store_true", help="group `_<level>` channels by variable")
    p.add_argument("--demo", action="store_true", help="run on a tiny random model (no checkpoint)")
    p.add_argument("--out", type=Path, default=None, help="write JSON importances here (default stdout)")
    args = p.parse_args()

    if args.demo:
        channels = 6
        model = _demo_model(channels)
        x = torch.randn(2, channels, 16, 32)
        names = _read_channel_names(args.channel_names, channels)
    else:
        if args.checkpoint is None or args.input is None:
            raise SystemExit("Need --checkpoint and --input (or use --demo).")
        model = _load_module(args.checkpoint)
        blob = torch.load(args.input, map_location="cpu", weights_only=False)
        x = blob["x"] if isinstance(blob, dict) else blob
        if not isinstance(x, torch.Tensor) or x.dim() != 4:
            raise SystemExit("--input must provide a [N,C,H,W] tensor (key 'x').")
        names = _read_channel_names(args.channel_names, x.shape[1])

    if args.method == "shap":
        attr = shap_attributions(model, x)
    else:
        attr = saliency_attributions(model, x)

    scores = per_channel_abs_importance(attr)
    groups = group_importances(scores, names, collapse_levels=args.collapse_levels)
    ranked = dict(sorted(groups.items(), key=lambda kv: kv[1], reverse=True))
    payload = json.dumps(
        {"method": args.method, "collapse_levels": args.collapse_levels, "importance": ranked},
        indent=2,
    )
    if args.out is not None:
        args.out.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

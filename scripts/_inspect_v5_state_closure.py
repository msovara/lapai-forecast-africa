#!/usr/bin/env python3
"""One-shot inspect of student_global_stable_v5.ckpt for Track B state-closure verdict."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from lapai_inference.model import LapAIStudentCNN, LapAIStudentConfig  # noqa: E402


def main() -> None:
    p = Path(sys.argv[1] if len(sys.argv) > 1 else "models/student_global_stable_v5.ckpt")
    ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
    print("path", p.resolve())
    print("top_keys", sorted(ckpt.keys()))
    for k in [
        "tp_mode",
        "channel_weights",
        "normalize_targets",
        "normalize_inputs",
        "africa_mix",
        "precip_log1p_weight",
    ]:
        if k in ckpt:
            print(k, ckpt[k])
    cfg = ckpt.get("cfg")
    print("cfg", cfg)
    if "target_mean" in ckpt:
        print("target_mean", ckpt["target_mean"].tolist())
    if "target_std" in ckpt:
        print("target_std", ckpt["target_std"].tolist())

    key = "model" if "model" in ckpt else ("state_dict" if "state_dict" in ckpt else None)
    if key is None:
        raise SystemExit("no model/state_dict in ckpt")
    sd = ckpt[key]
    for name, t in sd.items():
        if "head" in name:
            print("tensor", name, tuple(t.shape))

    if isinstance(cfg, dict):
        fields = set(LapAIStudentConfig.__dataclass_fields__)
        c = LapAIStudentConfig(**{k: v for k, v in cfg.items() if k in fields})
    else:
        c = cfg if cfg is not None else LapAIStudentConfig()
    net = LapAIStudentCNN(c)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print("out_channels", net.cfg.out_channels)
    print("in_channels_raw", net.cfg.in_channels_raw)
    print("head.weight", tuple(net.head.weight.shape))
    print("head.bias", net.head.bias.detach().tolist())
    print("missing", missing[:5], "n=", len(missing))
    print("unexpected", unexpected[:5], "n=", len(unexpected))
    # No decoder modules expected
    decoderish = [n for n in sd if any(s in n.lower() for s in ("decoder", "recon", "state_out", "upsample"))]
    print("decoderish_keys", decoderish)

    # Optional: confirm distillation cache shapes if present.
    for cand in (
        "data/processed/lapai/teacher_k1_cache_t256.zarr",
        "data/processed/lapai/teacher_k1_cache.zarr",
        "data/processed/lapai/teacher_k1_cache_smoke.zarr",
    ):
        zp = Path(cand)
        if not zp.exists():
            continue
        import zarr

        z = zarr.open(str(zp), mode="r")
        g = z["lapai_cache"] if "lapai_cache" in z else z
        print("cache", zp)
        print("  attrs", {k: g.attrs.get(k) for k in g.attrs})
        for key in ("state_in", "era5_target", "teacher_pred"):
            if key in g:
                print(f"  {key}", g[key].shape)


if __name__ == "__main__":
    main()

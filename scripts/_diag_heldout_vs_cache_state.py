#!/usr/bin/env python3
"""Diagnose train-cache state_in vs held-out CDS IC construction + pred ranges."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
VARS = ["t", "u", "v", "q", "z"]


def _block_stats(name: str, block: np.ndarray) -> None:
    print(
        f"  {name}: mean={block.mean():.5g} std={block.std():.5g} "
        f"min={block.min():.5g} max={block.max():.5g}"
    )


def cache_stats(path: Path, n: int = 8) -> None:
    import zarr

    z = zarr.open(str(path), mode="r")
    g = z["lapai_cache"] if "lapai_cache" in z else z
    s = np.asarray(g["state_in"][: min(n, g["state_in"].shape[0])], dtype=np.float64)
    print(f"\n=== CACHE {path.name} state_in {g['state_in'].shape} sample={s.shape} ===")
    for vi, v in enumerate(VARS):
        _block_stats(v, s[:, vi * 13 : (vi + 1) * 13])
    # per-level t/z sanity
    for lev_i, lev in enumerate([50, 500, 1000]):
        idx = VARS.index("t") * 13 + LEVELS.index(lev)
        _block_stats(f"t_{lev}", s[:, idx])
        idxz = VARS.index("z") * 13 + LEVELS.index(lev)
        _block_stats(f"z_{lev}", s[:, idxz])
    e = np.asarray(g["era5_target"][: min(n, g["era5_target"].shape[0])], dtype=np.float64)
    t = np.asarray(g["teacher_pred"][: min(n, g["teacher_pred"].shape[0])], dtype=np.float64)
    for i, name in enumerate(["tp", "msl", "2t"]):
        _block_stats(f"era5_{name}", e[:, i])
        _block_stats(f"teach_{name}", t[:, i])


def era5_zarr_stats(path: Path) -> None:
    import zarr

    ez = zarr.open(str(path), mode="r")
    print(f"\n=== ERA5 zarr {path.name} data={ez['data'].shape} ===")
    for k in ("variables", "start", "end", "frequency", "resolution"):
        if k in ez.attrs:
            v = ez.attrs[k]
            print(k, (v[:8] if isinstance(v, list) and len(v) > 8 else v))
    for key in ("dates", "times", "time"):
        if key in ez:
            d = np.asarray(ez[key][:])
            print(key, d.shape, d[0], "...", d[-1])
    vars_ = list(ez.attrs["variables"])
    vidx = {n: i for i, n in enumerate(vars_)}
    f0 = np.asarray(ez["data"][0, :, 0, :], dtype=np.float64)
    for v in VARS:
        idxs = [vidx[f"{v}_{lev}"] for lev in LEVELS]
        _block_stats(v, f0[idxs])
    for n in ("tp", "msl", "2t"):
        _block_stats(n, f0[vidx[n]])


def cds_state_stats(init: str, cds_cache: Path, n320: Path) -> np.ndarray:
    from evaluation.trackB_held_out_jan2023 import build_student_state_from_cds

    state, meta = build_student_state_from_cds(
        init, cache_dir=cds_cache, n320_latlon_path=n320, allow_download=False
    )
    print(f"\n=== CDS held-out state {init} shape={state.shape} meta={meta} ===")
    for vi, v in enumerate(VARS):
        _block_stats(v, state[vi * 13 : (vi + 1) * 13])
    for lev in (50, 500, 1000):
        idx = VARS.index("t") * 13 + LEVELS.index(lev)
        _block_stats(f"t_{lev}", state[idx])
        idxz = VARS.index("z") * 13 + LEVELS.index(lev)
        _block_stats(f"z_{lev}", state[idxz])
    return state


def pred_on_state(ckpt: Path, state: np.ndarray, tag: str) -> None:
    import torch
    from evaluation.trackB_gate import _load_student
    from utils.losses_distillation import apply_soft_physical_constraints

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, meta = _load_student(ckpt, device)
    with torch.no_grad():
        x = torch.as_tensor(state[None, ...], dtype=torch.float32, device=device)
        out = net(x)["pred"]
        if bool(meta.get("physical_constraints", True)):
            out = apply_soft_physical_constraints(out)
        pred = out[0].detach().cpu().numpy()
    print(f"\n=== student pred on {tag} device={device} ===")
    for i, n in enumerate(["tp", "msl", "2t"]):
        a = pred[i]
        print(
            f"  {n}: mean={a.mean():.5g} std={a.std():.5g} min={a.min():.5g} max={a.max():.5g} "
            f"p99={np.quantile(a, 0.99):.5g} frac>0={(a > 0).mean():.4f}"
        )
        if n == "tp":
            # 1mm / 6h ≈ 0.001 m
            print(f"  tp frac>=1mm: {(a >= 0.001).mean():.6f}")


def pred_on_cache(ckpt: Path, cache: Path, idx: int = 0) -> None:
    import zarr

    z = zarr.open(str(cache), mode="r")
    g = z["lapai_cache"] if "lapai_cache" in z else z
    state = np.asarray(g["state_in"][idx], dtype=np.float32)
    era5 = np.asarray(g["era5_target"][idx], dtype=np.float32)
    teach = np.asarray(g["teacher_pred"][idx], dtype=np.float32)
    pred_on_state(ckpt, state, f"cache[{idx}]")
    print("  (targets for reference)")
    for i, n in enumerate(["tp", "msl", "2t"]):
        print(
            f"  era5 {n}: mean={era5[i].mean():.5g} max={era5[i].max():.5g}; "
            f"teach mean={teach[i].mean():.5g} max={teach[i].max():.5g}"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=ROOT / "models/student_global_stable_v3.ckpt")
    p.add_argument("--cache", type=Path, default=ROOT / "data/processed/lapai/teacher_k1_cache_t256.zarr")
    p.add_argument("--era5", type=Path, default=ROOT / "data/processed/lapai/era5_n96_2020_2021.zarr")
    p.add_argument("--cds_cache", type=Path, default=Path.home() / ".cache/aifs-africa/era5")
    p.add_argument("--n320", type=Path, default=ROOT / "data/processed/lapai/n320_latlons.npy")
    p.add_argument("--init", default="20230101")
    p.add_argument("--skip_cds", action="store_true")
    p.add_argument("--skip_pred", action="store_true")
    args = p.parse_args()

    cache_stats(args.cache)
    era5_zarr_stats(args.era5)
    if not args.skip_pred:
        pred_on_cache(args.ckpt, args.cache, 0)
    if not args.skip_cds:
        state = cds_state_stats(args.init, args.cds_cache, args.n320)
        if not args.skip_pred:
            pred_on_state(args.ckpt, state, f"cds_{args.init}")


if __name__ == "__main__":
    main()

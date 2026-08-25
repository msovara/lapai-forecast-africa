#!/usr/bin/env python3
"""
Build Track B ``lapai_cache`` Zarr from K1 pruned GraphTransformer hooks.

Requires ``lapai-anemoi`` (anemoi + torch CUDA). Writes grids on the student
1° mesh (default 181×360).

Layer map (K1 has ``model.processor.num_layers=8``, not 16):
  PLAN placeholders L10/L14 → processor.proc[5] / processor.proc[7]
  (depth-proportional: ~10/16 and ~14/16 of an 8-layer stack).

Smoke example::

  export CUDA_VISIBLE_DEVICES=1 MKL_INTERFACE_LAYER=GNU
  conda activate /local/Mthetho/envs/lapai-anemoi
  python training/build_teacher_feature_cache.py \\
    --era5 data/processed/lapai/era5_n96_smoke.zarr \\
    --teacher models/teacher_pruned.ckpt \\
    --out data/processed/lapai/teacher_k1_cache_smoke.zarr \\
    --samples 8 --stride 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lapai_inference.cache_schema import SCHEMA_VERSION, init_zarr_store, lat_lon_mesh, zarr_append

# Depth-proportional map for 8-layer K1 GT (PLAN L10/L14 were AIFS placeholders).
DEFAULT_L10_INDEX = 5
DEFAULT_L14_INDEX = 7

# Student Cin=65 = 5 vars × 13 pressure levels (PLAN / LapAIStudentConfig).
LEVELS_13 = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
STATE_VARS = ["t", "u", "v", "q", "z"]
TARGET_NAMES = ["tp", "msl", "2t"]  # Cout=3


def _load_era5(path: Path):
    import zarr

    z = zarr.open(str(path), mode="r")
    variables = list(z.attrs["variables"])
    return z, variables


def _name_index(variables: list[str]) -> dict[str, int]:
    return {n: i for i, n in enumerate(variables)}


def _o96_to_latlon_grid(
    values: np.ndarray,
    lat_nodes: np.ndarray,
    lon_nodes: np.ndarray,
    lat_t: np.ndarray,
    lon_t: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour map from flattened O96 nodes (N,) or (C,N) → (C,H,W) or (H,W)."""
    from scipy.interpolate import NearestNDInterpolator

    lon_nodes = np.mod(lon_nodes, 360.0)
    pts = np.column_stack([lat_nodes, lon_nodes])
    lon_grid, lat_grid = np.meshgrid(lon_t, lat_t)
    query = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

    squeeze = False
    if values.ndim == 1:
        values = values[None, :]
        squeeze = True
    out_c = []
    for c in range(values.shape[0]):
        interp = NearestNDInterpolator(pts, values[c])
        out_c.append(interp(query).reshape(lat_t.size, lon_t.size).astype(np.float32))
    out = np.stack(out_c, axis=0)
    return out[0] if squeeze else out


def _hidden_to_latlon_grid(
    feat: np.ndarray,
    lat_h: np.ndarray,
    lon_h: np.ndarray,
    lat_t: np.ndarray,
    lon_t: np.ndarray,
) -> np.ndarray:
    """feat (N_hidden, D) → (D, H, W) via nearest neighbour."""
    return _o96_to_latlon_grid(feat.T, lat_h, lon_h, lat_t, lon_t)


def _build_state_in(
    frame: np.ndarray,
    vidx: dict[str, int],
    lat_n: np.ndarray,
    lon_n: np.ndarray,
    lat_t: np.ndarray,
    lon_t: np.ndarray,
) -> np.ndarray:
    """frame (C,N) → (65,H,W)."""
    channels = []
    for var in STATE_VARS:
        for lev in LEVELS_13:
            key = f"{var}_{lev}"
            if key not in vidx:
                raise KeyError(f"Missing ERA5 variable {key}")
            channels.append(frame[vidx[key]])
    stacked = np.stack(channels, axis=0)  # (65, N)
    return _o96_to_latlon_grid(stacked, lat_n, lon_n, lat_t, lon_t)


def _build_targets(
    frame: np.ndarray,
    vidx: dict[str, int],
    lat_n: np.ndarray,
    lon_n: np.ndarray,
    lat_t: np.ndarray,
    lon_t: np.ndarray,
) -> np.ndarray:
    chans = []
    for name in TARGET_NAMES:
        if name not in vidx:
            raise KeyError(name)
        chans.append(frame[vidx[name]])
    return _o96_to_latlon_grid(np.stack(chans, axis=0), lat_n, lon_n, lat_t, lon_t)


def _teacher_pred_from_output(
    y: torch.Tensor,
    name_to_index: dict[str, int],
    output_idx: torch.Tensor,
    lat_n: np.ndarray,
    lon_n: np.ndarray,
    lat_t: np.ndarray,
    lon_t: np.ndarray,
) -> np.ndarray:
    """
    y: (B,T,E,N,C_out88) model output channels in model-output order.
    Map tp/msl/2t via data-space names → output tensor positions.
    """
    # output_idx maps model-output positions → data variable indices
    data_pos = output_idx.cpu().numpy().tolist()
    inv = {int(d): i for i, d in enumerate(data_pos)}
    want = []
    for name in TARGET_NAMES:
        d_i = name_to_index[name]
        if d_i not in inv:
            raise KeyError(f"{name} (data idx {d_i}) not in model outputs")
        want.append(inv[d_i])
    arr = y.detach().float().cpu().numpy()[0, 0, 0]  # (N, C)
    sel = arr[:, want].T  # (3, N)
    return _o96_to_latlon_grid(sel, lat_n, lon_n, lat_t, lon_t)


def load_teacher(ckpt: Path, device: torch.device):
    obj = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    obj = obj.to(device)
    try:
        obj.graph_data = obj.graph_data.to(device)
    except Exception:
        pass
    obj.eval()
    return obj


def run_teacher_step(
    obj,
    xin_full_btn_c: torch.Tensor,
    l10: int,
    l14: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    xin_full_btn_c: (B,T,N,C=101) physical units.
    Returns (y_hat, feat10, feat14) on CPU float32.
    """
    device = next(obj.parameters()).device
    di = obj.data_indices["data"]
    input_idx = di.data.input.full.long().to(device)

    feats: dict[int, torch.Tensor] = {}

    def make_hook(idx: int):
        def _hook(_m, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            feats[idx] = t.detach()

        return _hook

    hs = [
        obj.model.processor.proc[l10].register_forward_hook(make_hook(l10)),
        obj.model.processor.proc[l14].register_forward_hook(make_hook(l14)),
    ]
    try:
        with torch.no_grad():
            x = xin_full_btn_c.to(device)
            x = obj.pre_processors["data"](x.clone())
            x = x[..., input_idx]
            y = obj.model.forward({"data": x.unsqueeze(2)})["data"]
            # Optional denorm for physical teacher_pred
            y_phys = obj.post_processors["data"](y, in_place=False)
    finally:
        for h in hs:
            h.remove()

    if l10 not in feats or l14 not in feats:
        raise RuntimeError(f"Hooks missed layers {l10}/{l14}; got {list(feats)}")
    return y_phys.cpu(), feats[l10].float().cpu(), feats[l14].float().cpu()


def main() -> None:
    p = argparse.ArgumentParser(description="Build K1 teacher feature lapai_cache Zarr")
    p.add_argument("--era5", type=Path, default=Path("data/processed/lapai/era5_n96_smoke.zarr"))
    p.add_argument("--teacher", type=Path, default=Path("models/teacher_pruned.ckpt"))
    p.add_argument("--out", type=Path, default=Path("data/processed/lapai/teacher_k1_cache_smoke.zarr"))
    p.add_argument("--samples", type=int, default=8, help="Number of IC times to cache")
    p.add_argument("--stride", type=int, default=12, help="Step between IC indices in ERA5 time axis")
    p.add_argument("--start", type=int, default=0, help="First ERA5 time index for IC t0")
    p.add_argument("--lead_steps", type=int, default=1, help="Target offset in 6h ERA5 steps (1=6h)")
    p.add_argument("--l10", type=int, default=DEFAULT_L10_INDEX)
    p.add_argument("--l14", type=int, default=DEFAULT_L14_INDEX)
    p.add_argument("--lat", type=int, default=181)
    p.add_argument("--lon", type=int, default=360)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[cache] device={device} teacher={args.teacher} era5={args.era5}")
    print(f"[cache] layer_map L10->proc[{args.l10}] L14->proc[{args.l14}] (K1 num_layers=8)")

    z, variables = _load_era5(args.era5)
    vidx = _name_index(variables)
    lat_nodes = np.asarray(z["latitudes"], dtype=np.float64)
    lon_nodes = np.asarray(z["longitudes"], dtype=np.float64)
    # anemoi often stores radians — detect
    if np.nanmax(np.abs(lat_nodes)) <= (np.pi / 2 + 0.1):
        lat_nodes = np.rad2deg(lat_nodes)
        lon_nodes = np.rad2deg(lon_nodes)
    lon_nodes = np.mod(lon_nodes, 360.0)

    lat_t, lon_t = lat_lon_mesh(args.lat, args.lon)
    T_era = int(z["data"].shape[0])
    need = args.start + args.samples * args.stride + max(1, args.lead_steps) + 1
    if need > T_era:
        raise SystemExit(f"ERA5 too short: need index <{need}, have {T_era}")

    obj = load_teacher(args.teacher, device)
    n_layers = int(obj.model.processor.num_layers)
    if args.l10 >= n_layers or args.l14 >= n_layers:
        raise SystemExit(f"Layer indices out of range for num_layers={n_layers}")

    # Hidden mesh coordinates (radians → degrees)
    with torch.no_grad():
        coords_h = obj.model.node_attributes.get_coordinates("hidden").detach().cpu().numpy()
    lat_h = np.rad2deg(coords_h[:, 0]) if np.nanmax(np.abs(coords_h[:, 0])) <= (np.pi / 2 + 0.1) else coords_h[:, 0]
    lon_h = np.rad2deg(coords_h[:, 1]) if np.nanmax(np.abs(coords_h)) <= (2 * np.pi + 0.1) else coords_h[:, 1]
    lon_h = np.mod(lon_h, 360.0)

    di = obj.data_indices["data"]
    name_to_index = dict(di.name_to_index)
    output_idx = di.data.output.full.long()

    root = init_zarr_store(
        args.out,
        attrs={
            "note": "k1_teacher_feature_cache",
            "teacher_ckpt": str(args.teacher.resolve()),
            "era5": str(args.era5.resolve()),
            "layer_map_json": json.dumps(
                {
                    "feature_L10": args.l10,
                    "feature_L14": args.l14,
                    "num_layers": n_layers,
                    "mapping": "depth_proportional_placeholders_L10_L14",
                }
            ),
            "lead_steps_6h": args.lead_steps,
            "grid": f"{args.lat}x{args.lon}",
            "smoke_subset": True,
            "lapai_schema_version": SCHEMA_VERSION,
        },
        overwrite=args.overwrite or not args.out.exists(),
    )
    g = root.require_group("lapai_cache") if hasattr(root, "require_group") else (
        root["lapai_cache"] if "lapai_cache" in root else root.create_group("lapai_cache")
    )
    g.attrs["lapai_schema_version"] = SCHEMA_VERSION
    g.attrs["layer_map_json"] = root.attrs["layer_map_json"]

    written = 0
    for s in range(args.samples):
        t0 = args.start + s * args.stride
        t1 = t0 + 1
        t_tgt = t0 + args.lead_steps
        # ERA5 frames (C,N)
        f0 = np.asarray(z["data"][t0, :, 0, :], dtype=np.float32)
        f1 = np.asarray(z["data"][t1, :, 0, :], dtype=np.float32)
        ft = np.asarray(z["data"][t_tgt, :, 0, :], dtype=np.float32)

        state = _build_state_in(f0, vidx, lat_nodes, lon_nodes, lat_t, lon_t)
        era5 = _build_targets(ft, vidx, lat_nodes, lon_nodes, lat_t, lon_t)

        # (B=1,T=2,N,C)
        xin = torch.from_numpy(np.stack([f0, f1], axis=0)).permute(0, 2, 1).unsqueeze(0).contiguous()
        y, feat10, feat14 = run_teacher_step(obj, xin, args.l10, args.l14)

        # feat*: (N_hidden, 256) — squeeze batch if present
        def _as_nodes(ft: torch.Tensor) -> np.ndarray:
            a = ft.numpy()
            if a.ndim == 3:
                a = a.reshape(-1, a.shape[-1])
            return a

        f10g = _hidden_to_latlon_grid(_as_nodes(feat10), lat_h, lon_h, lat_t, lon_t)
        f14g = _hidden_to_latlon_grid(_as_nodes(feat14), lat_h, lon_h, lat_t, lon_t)
        tpred = _teacher_pred_from_output(
            y, name_to_index, output_idx, lat_nodes, lon_nodes, lat_t, lon_t
        )

        arrays = {
            "state_in": state[None, ...].astype(np.float32),
            "era5_target": era5[None, ...].astype(np.float32),
            "teacher_pred": tpred[None, ...].astype(np.float32),
            "teacher_feat_L10": f10g[None, ...].astype(np.float32),
            "teacher_feat_L14": f14g[None, ...].astype(np.float32),
            "init_id": np.asarray([t0], dtype=np.int64),
            "lead_hours": np.asarray([6 * args.lead_steps], dtype=np.int32),
        }
        for name, arr in arrays.items():
            zarr_append(g, name, arr, written)
        written += 1
        print(
            f"[cache] wrote sample {written}/{args.samples} t0={t0} "
            f"state={state.shape} feat10={f10g.shape} pred={tpred.shape}",
            flush=True,
        )

    print(f"[cache] done -> {args.out.resolve()} T={written}")
    # Verify open
    from lapai_inference.dataset import LapAIZarrDataset

    ds = LapAIZarrDataset(args.out)
    print(
        f"[cache] verify ok len={len(ds)} cin={ds.cin} cout={ds.cout} "
        f"d10={ds.d_feat10} d14={ds.d_feat14} grid={ds.lat}x{ds.lon}"
    )


if __name__ == "__main__":
    main()

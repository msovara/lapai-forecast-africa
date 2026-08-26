"""Track B MVP gate: student vs K1 teacher on distillation-cache targets.

PLAN §5.4 / skill envelope: ≤15% global RMSE degradation vs teacher.
This MVP scores the trained student on the ``lapai_cache`` Zarr used for
distillation (one-step 6h lead, channels ``tp / msl / 2t``). It is **not**
the full Week-9 skill table at leads {24,72,120,240}h with z500/t850.

Usage::

  CUDA_VISIBLE_DEVICES=1 MKL_INTERFACE_LAYER=GNU \\
    python -u evaluation/trackB_gate.py \\
      --ckpt models/student_global.ckpt \\
      --cache data/processed/lapai/teacher_k1_cache.zarr \\
      --device cuda
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluation.masks import AFRICA_LAT, AFRICA_LON, africa_hw_mask
from lapai_inference.cache_schema import cosine_latitude_weights, lat_lon_mesh
from lapai_inference.dataset import open_cache_readonly
from lapai_inference.model import LapAIStudentCNN, LapAIStudentConfig
from utils.losses_distillation import apply_soft_physical_constraints, normalize_channels

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Cache / builder channel order (training/build_teacher_feature_cache.py).
DEFAULT_VARIABLES = ("tp", "msl", "2t")
DEFAULT_LEAD_HOURS = 6
DEFAULT_MAX_DEGRAD_PCT = 15.0


def _area_rmse_per_channel(
    pred: torch.Tensor,
    target: torch.Tensor,
    lat_weights: torch.Tensor,
    spatial_mask: torch.Tensor | None = None,
) -> list[float]:
    """pred/target (N,C,H,W) or (C,H,W) → list of C cosine-lat RMSE values.

    Optional ``spatial_mask`` (H,W) restricts both numerator and denominator.
    """
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    out: list[float] = []
    w = lat_weights.view(1, 1, -1, 1).to(pred)
    if spatial_mask is not None:
        sm = spatial_mask
        if sm.ndim == 2:
            sm = sm.view(1, 1, sm.shape[0], sm.shape[1])
        w = w * sm.to(device=w.device, dtype=w.dtype)
    for c in range(pred.shape[1]):
        diff2 = (pred[:, c : c + 1] - target[:, c : c + 1]) ** 2
        num = (w * diff2).sum()
        den = w.expand_as(diff2).sum().clamp_min(1e-12)
        out.append(float(torch.sqrt(num / den).detach().cpu()))
    return out


def _domain_mask(
    lat: np.ndarray, lon: np.ndarray, domain: str
) -> torch.Tensor | None:
    """Optional (H,W) spatial mask; Africa uses wrap-aware box (lon 0..360 safe)."""
    if domain == "global":
        return None
    if domain == "africa":
        return torch.as_tensor(africa_hw_mask(lat, lon, AFRICA_LAT, AFRICA_LON), dtype=torch.float32)
    raise ValueError(f"Unknown domain {domain!r}")


@torch.no_grad()
def score_student_on_cache(
    *,
    ckpt: Path,
    cache: Path,
    device: torch.device,
    variables: tuple[str, ...] = DEFAULT_VARIABLES,
    batch_size: int = 4,
    domain: str = "global",
) -> dict[str, Any]:
    g = open_cache_readonly(cache)
    n = int(g["era5_target"].shape[0])
    cin = int(g["state_in"].shape[1])
    cout = int(g["era5_target"].shape[1])
    h = int(g["state_in"].shape[2])
    w = int(g["state_in"].shape[3])
    if cout != len(variables):
        raise ValueError(f"Cache Cout={cout} but variables={variables}")

    lead_arr = None
    if "lead_hours" in g:
        lead_arr = np.asarray(g["lead_hours"][:])
    init_arr = None
    if "init_id" in g:
        init_arr = np.asarray(g["init_id"][:])

    net, ckpt_meta = _load_student(ckpt, device)
    if int(ckpt_meta["cfg"]["out_channels"]) != cout:
        raise ValueError(
            f"Checkpoint out_channels={ckpt_meta['cfg']['out_channels']} != cache Cout={cout}"
        )

    lat, lon = lat_lon_mesh(h, w)
    w_lat_full = torch.as_tensor(cosine_latitude_weights(lat).squeeze(), dtype=torch.float32)
    spat = _domain_mask(lat, lon, domain)

    stud_preds: list[torch.Tensor] = []
    teach_preds: list[torch.Tensor] = []
    era5_tgts: list[torch.Tensor] = []
    per_sample: list[dict[str, Any]] = []

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        x = torch.as_tensor(np.asarray(g["state_in"][start:stop], dtype=np.float32), device=device)
        if ckpt_meta.get("normalize_inputs") and ckpt_meta.get("input_mean") is not None:
            im = torch.as_tensor(ckpt_meta["input_mean"], dtype=torch.float32, device=device)
            istd = torch.as_tensor(ckpt_meta["input_std"], dtype=torch.float32, device=device)
            x = normalize_channels(x, im, istd)
        era5 = torch.as_tensor(np.asarray(g["era5_target"][start:stop], dtype=np.float32))
        teacher = torch.as_tensor(np.asarray(g["teacher_pred"][start:stop], dtype=np.float32))
        out = net(x)
        pred = out["pred"]
        if bool(ckpt_meta.get("physical_constraints", True)):
            pred = apply_soft_physical_constraints(
                pred, tp_mode=str(ckpt_meta.get("tp_mode", "relu"))
            )
        pred = pred.detach().cpu()

        stud_preds.append(pred)
        teach_preds.append(teacher)
        era5_tgts.append(era5)

        for i, idx in enumerate(range(start, stop)):
            s_rmse = _area_rmse_per_channel(pred[i], era5[i], w_lat_full, spat)
            t_rmse = _area_rmse_per_channel(teacher[i], era5[i], w_lat_full, spat)
            row: dict[str, Any] = {
                "sample_index": int(idx),
                "lead_hours": int(lead_arr[idx]) if lead_arr is not None else DEFAULT_LEAD_HOURS,
                "init_id": int(init_arr[idx]) if init_arr is not None else None,
                "variables": {},
            }
            for v, sr, tr in zip(variables, s_rmse, t_rmse):
                if tr <= 0:
                    deg = 0.0 if sr <= 0 else float("inf")
                else:
                    deg = 100.0 * (sr - tr) / tr
                row["variables"][v] = {
                    "student_rmse_vs_era5": sr,
                    "teacher_rmse_vs_era5": tr,
                    "degradation_pct_vs_teacher": deg,
                }
            per_sample.append(row)

    stud_all = torch.cat(stud_preds, dim=0)
    teach_all = torch.cat(teach_preds, dim=0)
    era5_all = torch.cat(era5_tgts, dim=0)

    stud_agg = _area_rmse_per_channel(stud_all, era5_all, w_lat_full, spat)
    teach_agg = _area_rmse_per_channel(teach_all, era5_all, w_lat_full, spat)
    stud_vs_teach = _area_rmse_per_channel(stud_all, teach_all, w_lat_full, spat)

    aggregate: dict[str, Any] = {}
    for v, sr, tr, svt in zip(variables, stud_agg, teach_agg, stud_vs_teach):
        if tr <= 0:
            deg = 0.0 if sr <= 0 else float("inf")
        else:
            deg = 100.0 * (sr - tr) / tr
        aggregate[v] = {
            "student_rmse_vs_era5": sr,
            "teacher_rmse_vs_era5": tr,
            "student_rmse_vs_teacher": svt,
            "degradation_pct_vs_teacher": deg,
        }

    return {
        "protocol": "trackB_cache_mvp",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ckpt": ckpt_meta,
        "cache": str(cache),
        "n_samples": n,
        "cin": cin,
        "cout": cout,
        "grid": [h, w],
        "domain": domain,
        "variables": list(variables),
        "lead_hours_default": DEFAULT_LEAD_HOURS,
        "cache_attrs": dict(g.attrs) if hasattr(g, "attrs") else {},
        "aggregate": aggregate,
        "per_sample": per_sample,
    }


def _load_student(ckpt_path: Path, device: torch.device) -> tuple[LapAIStudentCNN, dict[str, Any]]:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = dict(blob.get("cfg") or {})
    # dataclass fields only
    allowed = {f.name for f in LapAIStudentConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    cfg = LapAIStudentConfig(**{k: v for k, v in cfg_dict.items() if k in allowed})
    net = LapAIStudentCNN(cfg)
    net.load_state_dict(blob["model"])
    net.to(device)
    net.eval()
    meta = {
        "teacher_ckpt": blob.get("teacher_ckpt"),
        "cfg": asdict(cfg),
        "ckpt": str(ckpt_path),
        "physical_constraints": bool(blob.get("physical_constraints", True)),
        "normalize_targets": bool(blob.get("normalize_targets", False)),
        "normalize_inputs": bool(blob.get("normalize_inputs", False)),
        "tp_mode": str(blob.get("tp_mode", "relu")),
        "input_mean": blob.get("input_mean"),
        "input_std": blob.get("input_std"),
    }
    return net, meta


def gate_from_scorecard(
    scorecard: dict[str, Any],
    *,
    max_rmse_degradation_pct: float = DEFAULT_MAX_DEGRAD_PCT,
    variables: list[str] | None = None,
) -> dict[str, Any]:
    """Pass iff each aggregate variable degradation ≤ max_pct vs teacher."""
    vars_ = variables or list(scorecard.get("variables") or DEFAULT_VARIABLES)
    checks: list[dict[str, Any]] = []
    agg = scorecard.get("aggregate") or {}
    for v in vars_:
        row = agg.get(v) or {}
        if not row:
            checks.append(
                {
                    "variable": v,
                    "lead_hours": scorecard.get("lead_hours_default", DEFAULT_LEAD_HOURS),
                    "passed": False,
                    "error": "missing aggregate row",
                }
            )
            continue
        deg = float(row["degradation_pct_vs_teacher"])
        passed = deg <= max_rmse_degradation_pct
        checks.append(
            {
                "variable": v,
                "lead_hours": scorecard.get("lead_hours_default", DEFAULT_LEAD_HOURS),
                "passed": passed,
                "student_rmse_vs_era5": row["student_rmse_vs_era5"],
                "teacher_rmse_vs_era5": row["teacher_rmse_vs_era5"],
                "student_rmse_vs_teacher": row.get("student_rmse_vs_teacher"),
                "degradation_pct": deg,
                "max_allowed_pct": max_rmse_degradation_pct,
            }
        )

    n_pass = sum(1 for c in checks if c.get("passed"))
    n_tot = len(checks)
    return {
        "step": "trackB_student",
        "protocol": scorecard.get("protocol"),
        "passed": bool(checks) and all(c.get("passed") for c in checks),
        "pass_count": n_pass,
        "total_checks": n_tot,
        "max_rmse_degradation_pct": max_rmse_degradation_pct,
        "variables": vars_,
        "leads_hours": [scorecard.get("lead_hours_default", DEFAULT_LEAD_HOURS)],
        "baseline": "K1 teacher_pred on same cache (models/teacher_pruned.ckpt)",
        "candidate": scorecard.get("ckpt", {}).get("ckpt"),
        "cache": scorecard.get("cache"),
        "domain": scorecard.get("domain"),
        "n_samples": scorecard.get("n_samples"),
        "checks": checks,
        "aggregate": agg,
        "caveats": [
            "MVP cache gate: one-step 6h lead only (not PLAN {24,72,120,240}h skill table).",
            "Channels are tp/msl/2t only (no u10/v10/z500/t850).",
            f"Cache T≈{scorecard.get('n_samples')} from 2020–2021 ERA5; same store used for train → not held-out.",
            "Full Jan-2023 Track-A-style init forecast + ERA5 GCS truth scoring still TODO.",
        ],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _json_default(o: Any) -> Any:
    if isinstance(o, torch.Tensor):
        t = o.detach().cpu()
        return t.item() if t.ndim == 0 else t.tolist()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def write_json(obj: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Track B student MVP gate (cache targets)")
    p.add_argument("--ckpt", type=Path, default=_REPO_ROOT / "models/student_global.ckpt")
    p.add_argument(
        "--cache",
        type=Path,
        default=_REPO_ROOT / "data/processed/lapai/teacher_k1_cache.zarr",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--domain", choices=("global", "africa"), default="global")
    p.add_argument("--max_degradation_pct", type=float, default=DEFAULT_MAX_DEGRAD_PCT)
    p.add_argument(
        "--scorecard_out",
        type=Path,
        default=_REPO_ROOT / "reports/TRACKB_STUDENT_SCORECARD.json",
    )
    p.add_argument(
        "--gate_out",
        type=Path,
        default=_REPO_ROOT / "reports/TRACKB_GATE.json",
    )
    p.add_argument(
        "--extra_cache",
        type=Path,
        default=None,
        help="Optional second cache (e.g. smoke) scored into scorecard['ood']",
    )
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"[trackB_gate] ckpt={args.ckpt} cache={args.cache} device={device} domain={args.domain}")

    scorecard = score_student_on_cache(
        ckpt=args.ckpt,
        cache=args.cache,
        device=device,
        batch_size=args.batch_size,
        domain=args.domain,
    )
    if args.extra_cache is not None and Path(args.extra_cache).exists():
        print(f"[trackB_gate] extra OOD cache={args.extra_cache}")
        scorecard["ood"] = score_student_on_cache(
            ckpt=args.ckpt,
            cache=args.extra_cache,
            device=device,
            batch_size=args.batch_size,
            domain=args.domain,
        )

    gate = gate_from_scorecard(scorecard, max_rmse_degradation_pct=args.max_degradation_pct)
    write_json(scorecard, args.scorecard_out)
    write_json(gate, args.gate_out)

    print(json.dumps({"passed": gate["passed"], "pass_count": gate["pass_count"],
                      "total_checks": gate["total_checks"], "checks": gate["checks"]}, indent=2))
    print(f"wrote {args.scorecard_out}")
    print(f"wrote {args.gate_out}")


if __name__ == "__main__":
    main()

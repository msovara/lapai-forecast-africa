#!/usr/bin/env python3
"""Track B held-out Jan-2023 skill: K1 multi-lead + student 6h one-step.

Protocol (honest MVP; see caveats in output JSON):

1. **K1 teacher free-running skill** on existing Jan-2023 weekly forecast NetCDFs
   (``data/processed/trackA_prune_k1_a1bplus/forecasts``) vs GCS ERA5 Africa truth
   at leads ``{6,24,72,120,240}`` h for vars available in those files
   (``t2m, tp, u10, v10``). This is the multi-lead reference table.

2. **Student held-out 6h** (optional ``--student_ckpt``): CDS ERA5 IC cache → 65-ch
   1° state → ``student_global_stable_v3`` predicts ``tp/msl/2t``; score ``tp`` and
   ``2t`` (as t2m) on the Africa eval box vs GCS and vs K1 +6h from the same
   forecast files. Student cannot free-run (Cout=3 surface heads only).

Not PLAN-complete: no z500/t850 student heads; no free-running multi-day student
rollout; msl has no GCS truth stem in this repo's Mvua protocol.

Usage::

  # Teacher multi-lead only (CPU / laptop OK if GCS ADC works):
  python -u evaluation/trackB_held_out_jan2023.py

  # + student 6h on Cassava GPU1:
  CUDA_VISIBLE_DEVICES=1 MKL_INTERFACE_LAYER=GNU \\
    python -u evaluation/trackB_held_out_jan2023.py \\
      --student_ckpt models/student_global_stable_v3.ckpt --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.gcs_era5_truth import (  # noqa: E402
    GCS_OPTS,
    _select_tp_at_valid_time,
    era5_gcs_path,
    normalize_lon,
    pick_era5_var,
    subset_africa,
    valid_time_index,
)
from evaluation.phase0_scorecard import (  # noqa: E402
    _resolve_fc_var,
    _score_point,
    forecast_step_and_valid_time,
)
from utils.cds_ic import AIFS_AFRICA_CACHE_DIR  # noqa: E402

DEFAULT_INITS = ("20230101", "20230108", "20230115", "20230122", "20230129")
# PLAN §5.4 leads plus 6h (student native / first forecast step).
DEFAULT_LEADS = (6, 24, 72, 120, 240)
TEACHER_VARS = ("t2m", "tp", "u10", "v10")
STUDENT_SCORE_VARS = ("tp", "t2m")  # student 2t ↔ t2m
LEVELS_13 = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
STATE_VARS = ["t", "u", "v", "q", "z"]


class TruthCache:
    """GCS ERA5 Africa slices with per-variable year cache (Track A scoring pattern)."""

    def __init__(self) -> None:
        self._ds: dict[str, xr.Dataset] = {}
        self._da: dict[str, xr.DataArray] = {}

    def open_var(self, var: str, year: int = 2023) -> xr.DataArray:
        key = f"{var}_{year}"
        if key in self._da:
            return self._da[key]
        path = era5_gcs_path(var, year)
        print(f"[truth] opening {path}", flush=True)
        ds = xr.open_zarr(path, storage_options=GCS_OPTS, consolidated=True)
        name = pick_era5_var(ds)
        da = ds[name]
        for drop in ("number", "surface", "expver"):
            if drop in da.dims and da.sizes.get(drop, 0) == 1:
                da = da.isel({drop: 0})
        self._ds[key] = ds
        self._da[key] = da
        return da

    def at(self, var: str, when: np.datetime64) -> xr.DataArray:
        year = int(str(when)[:4])
        da = self.open_var(var, year)
        if var == "tp" and "step" in da.dims:
            sl = _select_tp_at_valid_time(da, when)
        else:
            ds = self._ds[f"{var}_{year}"]
            time_dim = "time" if "time" in da.dims else da.dims[0]
            sl = da.isel({time_dim: valid_time_index(ds, when)})
            if "step" in sl.dims and sl.sizes.get("step", 1) == 1:
                sl = sl.isel(step=0)
        return subset_africa(normalize_lon(sl.squeeze(drop=True)))

    def close(self) -> None:
        for ds in self._ds.values():
            ds.close()


def _degradation_pct(student_rmse: float, teacher_rmse: float) -> float:
    if teacher_rmse <= 0:
        return 0.0 if student_rmse <= 0 else float("inf")
    return 100.0 * (student_rmse - teacher_rmse) / teacher_rmse


def score_teacher_multilead(
    forecast_dir: Path,
    *,
    leads: tuple[int, ...],
    variables: tuple[str, ...],
    cache: TruthCache,
) -> list[dict[str, Any]]:
    paths = sorted(forecast_dir.glob("*_00Z.nc"))
    if not paths:
        raise SystemExit(f"no forecasts in {forecast_dir}")
    results: list[dict[str, Any]] = []
    for path in paths:
        init = path.name.split("_")[0]
        fc = xr.open_dataset(path)
        try:
            for lead in leads:
                try:
                    step, vt = forecast_step_and_valid_time(fc, lead)
                except ValueError as exc:
                    results.append(
                        {
                            "init_date": init,
                            "lead_hours": lead,
                            "forecast": str(path),
                            "error": str(exc),
                            "variables": {},
                        }
                    )
                    continue
                row: dict[str, Any] = {
                    "model": "K1_teacher",
                    "init_date": init,
                    "lead_hours": lead,
                    "valid_time": str(vt),
                    "forecast": str(path),
                    "variables": {},
                }
                for var in variables:
                    fc_name = _resolve_fc_var(fc, var)
                    if fc_name is None:
                        row["variables"][var] = {"error": "missing in forecast"}
                        continue
                    pred = fc[fc_name].isel(time=step).assign_coords(name=var)
                    try:
                        truth = cache.at(var, vt)
                        scores = _score_point(pred, truth)
                        row["variables"][var] = {
                            k: float(v) for k, v in scores.items() if np.isfinite(v)
                        }
                    except Exception as exc:  # noqa: BLE001
                        row["variables"][var] = {"error": str(exc)}
                    print(
                        f"[K1] {init} +{lead}h {var} "
                        f"rmse={row['variables'][var].get('rmse')}",
                        flush=True,
                    )
                results.append(row)
        finally:
            fc.close()
    return results


def _n320_to_latlon1deg(
    values_n320: np.ndarray,
    n320_latlon: np.ndarray,
    lat_t: np.ndarray,
    lon_t: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour N320 → (H,W) 1° mesh (lon 0–360)."""
    from scipy.interpolate import NearestNDInterpolator

    lon_nodes = np.mod(n320_latlon[:, 1], 360.0)
    pts = np.column_stack([n320_latlon[:, 0], lon_nodes])
    lon_grid, lat_grid = np.meshgrid(lon_t, lat_t)
    query = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    interp = NearestNDInterpolator(pts, np.asarray(values_n320, dtype=np.float64))
    return interp(query).reshape(lat_t.size, lon_t.size).astype(np.float32)


def build_student_state_from_cds(
    init_yyyymmdd: str,
    *,
    cache_dir: Path,
    n320_latlon_path: Path,
    allow_download: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return state_in (65,H,W) for 00Z init from CDS IC cache (t−6h, t0 stacked fields)."""
    from lapai_inference.cache_schema import lat_lon_mesh
    from utils.cds_ic import retrieve_era5_fields

    if not n320_latlon_path.is_file():
        raise FileNotFoundError(
            f"Need N320 lat/lon table at {n320_latlon_path} "
            "(Cassava: data/processed/lapai/n320_latlons.npy)"
        )
    fields = retrieve_era5_fields(
        init_yyyymmdd,
        "0000",
        cache_dir=cache_dir,
        allow_download=allow_download,
    )
    n320_latlon = np.asarray(np.load(n320_latlon_path), dtype=np.float64)
    if n320_latlon.ndim != 2 or n320_latlon.shape[1] != 2:
        raise ValueError(f"bad n320_latlon shape {n320_latlon.shape}")
    lat_t, lon_t = lat_lon_mesh(181, 360)
    channels: list[np.ndarray] = []
    missing: list[str] = []
    for var in STATE_VARS:
        for lev in LEVELS_13:
            key = f"{var}_{lev}"
            if key not in fields:
                missing.append(key)
                continue
            # fields[key]: (2, n320) — use analysis at init (index 1)
            vec = np.asarray(fields[key][1], dtype=np.float32)
            channels.append(_n320_to_latlon1deg(vec, n320_latlon, lat_t, lon_t))
    if missing:
        raise KeyError(f"CDS IC missing fields: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    state = np.stack(channels, axis=0)
    meta = {
        "init_date": init_yyyymmdd,
        "init_time": "0000",
        "cache_dir": str(cache_dir),
        "n320_points": int(n320_latlon.shape[0]),
        "state_shape": list(state.shape),
    }
    return state, meta


def _student_pred_to_africa_da(
    field_hw: np.ndarray,
    *,
    name: str,
    lat_full: np.ndarray,
    lon_full: np.ndarray,
    template: xr.DataArray,
) -> xr.DataArray:
    """Map student 1° global field onto teacher Africa 0.25° grid via nearest reindex."""
    lon180 = ((lon_full + 180.0) % 360.0) - 180.0
    da = xr.DataArray(
        field_hw,
        dims=("latitude", "longitude"),
        coords={"latitude": lat_full, "longitude": lon180},
        name=name,
    )
    # Student lon mesh is 0–360 monotonic; after unwrap to −180..180 sort for reindex.
    da = da.assign_coords(longitude=lon180).sortby("longitude")
    return da.reindex(
        latitude=template.latitude,
        longitude=template.longitude,
        method="nearest",
    )


def score_student_6h(
    *,
    ckpt: Path,
    forecast_dir: Path,
    inits: tuple[str, ...],
    cache: TruthCache,
    device: Any,
    cds_cache: Path,
    n320_latlon_path: Path,
    allow_download: bool,
    student_forecast_dir: Path | None = None,
) -> list[dict[str, Any]]:
    import torch
    from evaluation.trackB_gate import _load_student
    from lapai_inference.cache_schema import lat_lon_mesh
    from utils.losses_distillation import apply_soft_physical_constraints

    net, ckpt_meta = _load_student(ckpt, device)
    lat_full, lon_full = lat_lon_mesh(181, 360)
    results: list[dict[str, Any]] = []
    if student_forecast_dir is not None:
        student_forecast_dir.mkdir(parents=True, exist_ok=True)

    for init in inits:
        fc_path = forecast_dir / f"{init}_00Z.nc"
        if not fc_path.is_file():
            results.append({"init_date": init, "error": f"missing forecast {fc_path.name}"})
            continue
        try:
            state, state_meta = build_student_state_from_cds(
                init,
                cache_dir=cds_cache,
                n320_latlon_path=n320_latlon_path,
                allow_download=allow_download,
            )
        except Exception as exc:  # noqa: BLE001
            results.append({"init_date": init, "error": f"CDS/state build failed: {exc}"})
            print(f"[student] {init} state fail: {exc}", flush=True)
            continue

        with torch.no_grad():
            x = torch.as_tensor(state[None, ...], dtype=torch.float32, device=device)
            out = net(x)["pred"]
            if bool(ckpt_meta.get("physical_constraints", True)):
                out = apply_soft_physical_constraints(out)
            pred = out[0].detach().cpu().numpy()  # (3,H,W) tp, msl, 2t

        fc = xr.open_dataset(fc_path)
        try:
            step6, vt6 = forecast_step_and_valid_time(fc, 6)
            row: dict[str, Any] = {
                "model": "student_v3",
                "init_date": init,
                "lead_hours": 6,
                "valid_time": str(vt6),
                "ckpt": str(ckpt),
                "state_meta": state_meta,
                "variables": {},
            }
            chan = {"tp": 0, "msl": 1, "t2m": 2}
            stud_fields: dict[str, xr.DataArray] = {}
            for var in STUDENT_SCORE_VARS:
                template_name = _resolve_fc_var(fc, var)
                if template_name is None:
                    row["variables"][var] = {"error": "no teacher template grid"}
                    continue
                template = fc[template_name].isel(time=step6)
                stud_da = _student_pred_to_africa_da(
                    pred[chan[var]],
                    name=var,
                    lat_full=lat_full,
                    lon_full=lon_full,
                    template=template,
                ).assign_coords(name=var)
                stud_fields[var] = stud_da

                # Always score student vs teacher on Africa grid (no GCS required).
                p = np.asarray(stud_da.values, dtype=np.float64)
                t = np.asarray(template.values, dtype=np.float64)
                lat = np.asarray(stud_da.latitude.values, dtype=np.float64)
                from lapai_inference.cache_schema import cosine_latitude_weights

                w = np.asarray(cosine_latitude_weights(lat), dtype=np.float64).reshape(-1, 1)
                mask = np.isfinite(p) & np.isfinite(t)
                svt = float(
                    np.sqrt(np.sum(w * (p - t) ** 2 * mask) / max(np.sum(w * mask), 1e-12))
                )
                cell: dict[str, Any] = {
                    "student_rmse_vs_teacher_field": svt,
                }

                try:
                    if getattr(cache, "_disabled", False):
                        raise RuntimeError("ERA5 truth scoring disabled (--no_era5)")
                    truth = cache.at(var, vt6)
                    stud_scores = _score_point(stud_da, truth)
                    teach_scores = _score_point(template.assign_coords(name=var), truth)
                    s_rmse = float(stud_scores["rmse"])
                    t_rmse = float(teach_scores["rmse"])
                    cell.update(
                        {
                            "student_rmse_vs_era5": s_rmse,
                            "student_acc": float(stud_scores.get("acc", float("nan"))),
                            "teacher_rmse_vs_era5": t_rmse,
                            "teacher_acc": float(teach_scores.get("acc", float("nan"))),
                            "degradation_pct_vs_teacher": _degradation_pct(s_rmse, t_rmse),
                        }
                    )
                    if var == "tp" and "pod_1mm" in stud_scores:
                        cell["student_pod_1mm"] = float(stud_scores["pod_1mm"])
                except Exception as exc:  # noqa: BLE001
                    cell["era5_score_error"] = str(exc)
                    print(f"[student] {init} +6h {var} ERA5 score skipped: {exc}", flush=True)

                row["variables"][var] = cell
                print(f"[student] {init} +6h {var} {cell}", flush=True)

            if student_forecast_dir is not None and stud_fields:
                # One-step NC matching Track A layout (time=1 at +6h) for local GCS rescore.
                out_nc = student_forecast_dir / f"{init}_00Z.nc"
                data_vars = {}
                for name, da in stud_fields.items():
                    clean = da.reset_coords(drop=True)
                    if "name" in clean.coords:
                        clean = clean.drop_vars("name")
                    clean = clean.expand_dims(time=[np.datetime64(vt6)])
                    clean.name = name
                    data_vars[name] = clean
                ds_out = xr.Dataset(data_vars)
                ds_out.attrs["title"] = "Track B student held-out Jan-2023 +6h"
                ds_out.attrs["reference_time"] = f"{init} 00:00:00"
                ds_out.attrs["checkpoint"] = str(ckpt)
                ds_out.to_netcdf(out_nc)
                row["forecast_nc"] = str(out_nc)
                print(f"[student] wrote {out_nc}", flush=True)

            results.append(row)
        finally:
            fc.close()
    return results


def _aggregate_teacher(results: list[dict[str, Any]], leads: tuple[int, ...], variables: tuple[str, ...]) -> dict[str, Any]:
    agg: dict[str, Any] = {}
    for lead in leads:
        agg[str(lead)] = {}
        for var in variables:
            rmses = [
                r["variables"][var]["rmse"]
                for r in results
                if r.get("lead_hours") == lead
                and "rmse" in (r.get("variables") or {}).get(var, {})
            ]
            if not rmses:
                agg[str(lead)][var] = {"n": 0}
            else:
                agg[str(lead)][var] = {
                    "n": len(rmses),
                    "rmse_mean": float(np.mean(rmses)),
                    "rmse_std": float(np.std(rmses)),
                }
    return agg


def _aggregate_student(results: list[dict[str, Any]]) -> dict[str, Any]:
    agg: dict[str, Any] = {}
    for var in STUDENT_SCORE_VARS:
        s_rmses, t_rmses, degs, svt = [], [], [], []
        for r in results:
            cell = (r.get("variables") or {}).get(var) or {}
            if "student_rmse_vs_teacher_field" in cell:
                svt.append(cell["student_rmse_vs_teacher_field"])
            if "student_rmse_vs_era5" not in cell:
                continue
            s_rmses.append(cell["student_rmse_vs_era5"])
            t_rmses.append(cell["teacher_rmse_vs_era5"])
            degs.append(cell["degradation_pct_vs_teacher"])
        row: dict[str, Any] = {"n_era5": len(s_rmses), "n_vs_teacher_field": len(svt)}
        if svt:
            row["student_rmse_vs_teacher_field_mean"] = float(np.mean(svt))
        if s_rmses:
            row.update(
                {
                    "student_rmse_mean": float(np.mean(s_rmses)),
                    "teacher_rmse_mean": float(np.mean(t_rmses)),
                    "degradation_pct_mean": float(np.mean(degs)),
                }
            )
        agg[var] = row
    return agg


def main() -> None:
    p = argparse.ArgumentParser(description="Track B Jan-2023 held-out skill vs K1")
    p.add_argument(
        "--forecast_dir",
        type=Path,
        default=_REPO_ROOT / "data/processed/trackA_prune_k1_a1bplus/forecasts",
    )
    p.add_argument("--leads", type=str, default=",".join(str(x) for x in DEFAULT_LEADS))
    p.add_argument(
        "--student_ckpt",
        type=Path,
        default=None,
        help="If set, also score student 6h from CDS IC cache",
    )
    p.add_argument("--device", default=None, help="cuda/cpu for student; default auto")
    p.add_argument("--cds_cache", type=Path, default=Path(AIFS_AFRICA_CACHE_DIR))
    p.add_argument(
        "--n320_latlon",
        type=Path,
        default=_REPO_ROOT / "data/processed/lapai/n320_latlons.npy",
    )
    p.add_argument("--allow_download", action="store_true")
    p.add_argument(
        "--student_forecast_dir",
        type=Path,
        default=_REPO_ROOT / "data/processed/trackB_student_heldout/forecasts",
        help="Write student Africa +6h NetCDFs for offline GCS rescore",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "reports/TRACKB_HELD_OUT_JAN2023.json",
    )
    p.add_argument("--skip_teacher", action="store_true", help="Only run student 6h block")
    p.add_argument(
        "--no_era5",
        action="store_true",
        help="Skip GCS ERA5 scoring for student (still writes NC + vs-teacher field RMSE)",
    )
    args = p.parse_args()

    leads = tuple(int(x) for x in args.leads.split(",") if x.strip())
    report: dict[str, Any] = {
        "protocol": "trackB_held_out_jan2023",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": "K1 teacher (models/teacher_pruned.ckpt) Jan-2023 weekly free-running forecasts",
        "candidate_student": str(args.student_ckpt) if args.student_ckpt else None,
        "forecast_dir": str(args.forecast_dir),
        "domain": "africa",
        "truth_source": "gcs://code4earth/era5",
        "leads_hours": list(leads),
        "teacher_variables": list(TEACHER_VARS),
        "student_variables_scored": list(STUDENT_SCORE_VARS),
        "caveats": [
            "Student MVP outputs tp/msl/2t only — no z500/t850 heads; PLAN Week-9 skill table incomplete.",
            "Student has no free-running multi-day rollout (65→3 channel heads); multi-lead table is K1-only.",
            "Student held-out comparison is analysis-forced 6h one-step from CDS ERA5 ICs (Jan 2023 weekly 00Z).",
            "msl not scored held-out (no GCS truth stem in Mvua protocol; teacher forecast NCs lack msl).",
            "MVP cache gate accepted with documented tp soft-fail (21.2% > 15%); this protocol is off-cache Jan-2023.",
        ],
        "teacher_results": [],
        "student_results": [],
        "aggregate": {},
    }

    cache = TruthCache()
    if args.no_era5:
        cache._disabled = True  # type: ignore[attr-defined]
    try:
        if not args.skip_teacher:
            print(f"[heldout] scoring K1 multi-lead from {args.forecast_dir}", flush=True)
            report["teacher_results"] = score_teacher_multilead(
                args.forecast_dir, leads=leads, variables=TEACHER_VARS, cache=cache
            )
            report["aggregate"]["teacher"] = _aggregate_teacher(
                report["teacher_results"], leads, TEACHER_VARS
            )

        if args.student_ckpt is not None:
            import torch

            print(f"[heldout] scoring student 6h ckpt={args.student_ckpt}", flush=True)
            dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
            device = torch.device(dev)
            report["student_results"] = score_student_6h(
                ckpt=args.student_ckpt,
                forecast_dir=args.forecast_dir,
                inits=DEFAULT_INITS,
                cache=cache,
                device=device,
                cds_cache=args.cds_cache,
                n320_latlon_path=args.n320_latlon,
                allow_download=args.allow_download,
                student_forecast_dir=args.student_forecast_dir,
            )
            report["aggregate"]["student_6h"] = _aggregate_student(report["student_results"])
    finally:
        cache.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    if report["aggregate"]:
        print(json.dumps(report["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()

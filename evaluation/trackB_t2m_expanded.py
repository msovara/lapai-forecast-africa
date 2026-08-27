#!/usr/bin/env python3
"""Track B production t2m evaluation — analysis-forced multi-season (v5 freeze).

Protocol (locked):
  * Checkpoint: ``models/student_global_stable_v5.ckpt`` (no architecture change)
  * Analysis-forced only: for lead L, IC at init+(L−6)h → one student +6h step
  * Leads: +6, +12, +18, +24 h
  * Primary variable: **t2m** (channel ``2t``). tp is out-of-scope (dry-collapse note only)
  * IC + surface truth: public ARCO ERA5 (no CDS)
  * K1 comparison where teacher forecast NCs exist; student-vs-ERA5 always

Does **not** require a K1 NetCDF per init (unlike Jan-2023 held-out). Africa scoring
grid comes from ARCO truth; when a K1 file exists, both are scored with the same
``_score_t2m`` methodology vs ERA5.

Usage (Cassava GPU1)::

  CUDA_VISIBLE_DEVICES=1 MKL_INTERFACE_LAYER=GNU \\
    python -u evaluation/trackB_t2m_expanded.py \\
      --ckpt models/student_global_stable_v5.ckpt \\
      --device cuda --ic_source arco \\
      --step_days 5 --leads 6,12,18,24
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "dummy")

from evaluation.gcs_era5_truth import (  # noqa: E402
    align_truth_to_pred,
    normalize_lon,
    subset_africa,
)
from evaluation.trackB_held_out_jan2023 import (  # noqa: E402
    _degradation_pct,
    _ic_datetime_for_lead,
    _student_pred_to_africa_da,
    build_student_state,
)
from lapai_inference.cache_schema import cosine_latitude_weights, lat_lon_mesh  # noqa: E402
from utils.era5_ondisk_ic import (  # noqa: E402
    ARCO_DEFAULT,
    DEFAULT_STATE_CACHE,
    prefetch_arco_student_states,
)

DEFAULT_LEADS = (6, 12, 18, 24)
SEASON_OF_MONTH = {
    12: "DJF",
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
}
ARCO_T2M = "2m_temperature"
ARCO_TP = "total_precipitation"
GCS_OPTS = {"token": "anon"}


def expand_inits_2023(
    *,
    step_days: int = 5,
    months: tuple[int, ...] | None = None,
) -> tuple[str, ...]:
    """00Z initialisation dates in 2023 every ``step_days`` (optionally month-filtered)."""
    d = date(2023, 1, 1)
    end = date(2023, 12, 31)
    out: list[str] = []
    while d <= end:
        if months is None or d.month in months:
            out.append(d.strftime("%Y%m%d"))
        d += timedelta(days=step_days)
    return tuple(out)


def season_of(init_yyyymmdd: str) -> str:
    return SEASON_OF_MONTH[int(init_yyyymmdd[4:6])]


class ArcoSurfaceTruth:
    """Public ARCO ERA5 surface fields (Africa box) for scoring."""

    def __init__(self, path: str = ARCO_DEFAULT) -> None:
        print(f"[truth] opening {path}", flush=True)
        self.path = path
        self.ds = xr.open_zarr(path, storage_options=GCS_OPTS, consolidated=True)
        self._cache: dict[str, xr.DataArray] = {}

    def at(self, var: str, when: np.datetime64) -> xr.DataArray:
        name = ARCO_T2M if var == "t2m" else ARCO_TP
        key = f"{var}_{np.datetime64(when, 'h')}"
        if key in self._cache:
            return self._cache[key]
        da = self.ds[name]
        for drop in ("level", "number", "surface", "expver"):
            if drop in da.dims and da.sizes.get(drop, 0) == 1:
                da = da.isel({drop: 0})
        sl = da.sel(time=np.datetime64(when, "ns"), method="nearest")
        sl = sl.load().squeeze(drop=True)
        out = subset_africa(normalize_lon(sl))
        self._cache[key] = out
        return out

    def close(self) -> None:
        self.ds.close()


def _score_t2m(pred: xr.DataArray, truth: xr.DataArray) -> dict[str, float]:
    """Cosine-latitude RMSE / ACC / bias — shared student & K1 methodology."""
    truth_a = align_truth_to_pred(truth, pred)
    p = np.asarray(pred.values, dtype=np.float64).squeeze()
    t = np.asarray(truth_a.values, dtype=np.float64).squeeze()
    if p.ndim != 2 or t.ndim != 2:
        raise ValueError(f"expected 2-D fields, got pred={p.shape} truth={t.shape}")
    lat = np.asarray(pred.latitude.values, dtype=np.float64).squeeze()
    w = np.asarray(cosine_latitude_weights(lat), dtype=np.float64).squeeze()
    w2d = w[:, None] * np.ones(p.shape[1], dtype=np.float64)
    mask = np.isfinite(p) & np.isfinite(t)
    if not mask.any():
        return {"rmse": float("nan"), "acc": float("nan"), "bias": float("nan")}

    err = p - t
    rmse = float(np.sqrt(np.sum(w2d[mask] * err[mask] ** 2) / np.sum(w2d[mask])))
    bias = float(np.sum(w2d[mask] * err[mask]) / np.sum(w2d[mask]))

    clim = np.nanmean(t, axis=0, keepdims=True)
    pred_anom = p - clim
    ref_anom = t - clim
    num = np.sum(w2d[mask] * pred_anom[mask] * ref_anom[mask])
    den = np.sqrt(
        np.sum(w2d[mask] * pred_anom[mask] ** 2)
        * np.sum(w2d[mask] * ref_anom[mask] ** 2)
    )
    acc = float(num / den) if den > 0 else float("nan")
    return {"rmse": rmse, "acc": acc, "bias": bias}


def _k1_step_and_field(
    fc: xr.Dataset, lead_hours: int, var: str = "t2m"
) -> tuple[np.datetime64, xr.DataArray] | None:
    """Return (valid_time, Africa field) from a K1 forecast NC, or None if missing."""
    from evaluation.phase0_scorecard import _resolve_fc_var, forecast_step_and_valid_time

    name = _resolve_fc_var(fc, var)
    if name is None:
        return None
    try:
        step, vt = forecast_step_and_valid_time(fc, lead_hours)
    except ValueError:
        return None
    da = fc[name].isel(time=step).assign_coords(name=var)
    return np.datetime64(vt), da


def run_expanded(
    *,
    ckpt: Path,
    inits: tuple[str, ...],
    leads: tuple[int, ...],
    device: Any,
    ic_source: str,
    arco_path: str,
    state_cache_dir: Path,
    teacher_fc_dir: Path,
    student_forecast_dir: Path | None,
    figures_dir: Path,
    cds_cache: Path,
    n320_latlon: Path,
) -> dict[str, Any]:
    import torch
    from evaluation.trackB_gate import _load_student
    from utils.losses_distillation import apply_soft_physical_constraints, normalize_channels

    truth = ArcoSurfaceTruth(arco_path)
    net, ckpt_meta = _load_student(ckpt, device)
    lat_full, lon_full = lat_lon_mesh(181, 360)

    # Prefetch ARCO ICs once.
    specs: list[tuple[str, str]] = []
    for init in inits:
        for lead in leads:
            specs.append(_ic_datetime_for_lead(init, lead))
    uniq = sorted(set(specs))
    print(f"[ic] prefetching {len(uniq)} unique ARCO ICs for {len(inits)} inits", flush=True)
    ic_map = prefetch_arco_student_states(
        uniq, arco_path=arco_path, state_cache_dir=state_cache_dir
    )

    if student_forecast_dir is not None:
        student_forecast_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Spatial accumulators on Africa truth grid (0.25°) for +6h / +24h.
    spatial: dict[int, dict[str, Any]] = {}
    for lead in (6, 24):
        if lead in leads:
            spatial[lead] = {"n": 0, "sum_err": None, "sum_err2": None, "lat": None, "lon": None}

    student_rows: list[dict[str, Any]] = []
    k1_rows: list[dict[str, Any]] = []
    k1_coverage: list[str] = []
    tp_note_samples: list[dict[str, Any]] = []

    for init in inits:
        season = season_of(init)
        teach_path = teacher_fc_dir / f"{init}_00Z.nc"
        teach_fc = xr.open_dataset(teach_path) if teach_path.is_file() else None
        if teach_fc is not None:
            k1_coverage.append(init)

        try:
            for lead in leads:
                ic_date, ic_hhmm = _ic_datetime_for_lead(init, lead)
                key = (ic_date, ic_hhmm)
                try:
                    if key in ic_map:
                        state, state_meta = ic_map[key]
                    else:
                        state, state_meta = build_student_state(
                            ic_date,
                            ic_source=ic_source,
                            cache_dir=cds_cache,
                            n320_latlon_path=n320_latlon,
                            allow_download=False,
                            init_hhmm=ic_hhmm,
                            arco_path=arco_path,
                            state_cache_dir=state_cache_dir,
                        )
                        ic_map[key] = (state, state_meta)
                except Exception as exc:  # noqa: BLE001
                    student_rows.append(
                        {
                            "init_date": init,
                            "season": season,
                            "lead_hours": lead,
                            "error": f"IC failed: {exc}",
                        }
                    )
                    print(f"[student] {init} +{lead}h IC fail: {exc}", flush=True)
                    continue

                with torch.no_grad():
                    x = torch.as_tensor(state[None, ...], dtype=torch.float32, device=device)
                    if ckpt_meta.get("normalize_inputs") and ckpt_meta.get("input_mean") is not None:
                        im = torch.as_tensor(
                            ckpt_meta["input_mean"], dtype=torch.float32, device=device
                        )
                        istd = torch.as_tensor(
                            ckpt_meta["input_std"], dtype=torch.float32, device=device
                        )
                        x = normalize_channels(x, im, istd)
                    out = net(x)["pred"]
                    if bool(ckpt_meta.get("physical_constraints", True)):
                        out = apply_soft_physical_constraints(
                            out, tp_mode=str(ckpt_meta.get("tp_mode", "relu"))
                        )
                    pred = out[0].detach().cpu().numpy()  # (3,) tp,msl,2t

                init_dt = datetime.strptime(init + "0000", "%Y%m%d%H%M")
                vt = np.datetime64(init_dt + timedelta(hours=lead))
                era5_t2m = truth.at("t2m", vt)
                stud_t2m = _student_pred_to_africa_da(
                    pred[2],
                    name="t2m",
                    lat_full=lat_full,
                    lon_full=lon_full,
                    template=era5_t2m,
                ).assign_coords(name="t2m")

                s_scores = _score_t2m(stud_t2m, era5_t2m)
                cell: dict[str, Any] = {
                    "student_rmse_vs_era5": s_scores["rmse"],
                    "student_acc": s_scores["acc"],
                    "student_bias": s_scores["bias"],
                }

                # Optional K1 comparison (same scoring vs ERA5).
                if teach_fc is not None:
                    got = _k1_step_and_field(teach_fc, lead, "t2m")
                    if got is not None:
                        _vt_k1, teach_da = got
                        t_scores = _score_t2m(teach_da.assign_coords(name="t2m"), era5_t2m)
                        cell["teacher_rmse_vs_era5"] = t_scores["rmse"]
                        cell["teacher_acc"] = t_scores["acc"]
                        cell["teacher_bias"] = t_scores["bias"]
                        cell["degradation_pct_vs_teacher"] = _degradation_pct(
                            s_scores["rmse"], t_scores["rmse"]
                        )
                        # Field RMSE student vs teacher on teacher grid.
                        stud_on_k1 = _student_pred_to_africa_da(
                            pred[2],
                            name="t2m",
                            lat_full=lat_full,
                            lon_full=lon_full,
                            template=teach_da,
                        )
                        p = np.asarray(stud_on_k1.values, dtype=np.float64)
                        t = np.asarray(teach_da.values, dtype=np.float64)
                        lat = np.asarray(stud_on_k1.latitude.values, dtype=np.float64)
                        w = np.asarray(cosine_latitude_weights(lat), dtype=np.float64).reshape(-1, 1)
                        mask = np.isfinite(p) & np.isfinite(t)
                        cell["student_rmse_vs_teacher_field"] = float(
                            np.sqrt(np.sum(w * (p - t) ** 2 * mask) / max(np.sum(w * mask), 1e-12))
                        )
                        k1_rows.append(
                            {
                                "model": "K1_teacher",
                                "init_date": init,
                                "season": season,
                                "lead_hours": lead,
                                "valid_time": str(vt),
                                "variables": {
                                    "t2m": {
                                        "rmse": t_scores["rmse"],
                                        "acc": t_scores["acc"],
                                        "bias": t_scores["bias"],
                                    }
                                },
                            }
                        )

                row = {
                    "model": "student",
                    "init_date": init,
                    "season": season,
                    "lead_hours": lead,
                    "valid_time": str(vt),
                    "ckpt": str(ckpt),
                    "protocol": "analysis_forced_6h_step",
                    "ic_date": ic_date,
                    "ic_time": ic_hhmm,
                    "state_meta": {
                        k: state_meta[k]
                        for k in ("ic_source", "arco_time_selected", "state_shape")
                        if k in state_meta
                    },
                    "variables": {"t2m": cell},
                    "k1_available": teach_fc is not None and "teacher_rmse_vs_era5" in cell,
                }
                student_rows.append(row)
                print(
                    f"[student] {init} {season} +{lead}h t2m "
                    f"rmse={cell['student_rmse_vs_era5']:.3f} "
                    f"acc={cell['student_acc']:.3f} bias={cell['student_bias']:.3f}"
                    + (
                        f" deg={cell['degradation_pct_vs_teacher']:+.1f}%"
                        if "degradation_pct_vs_teacher" in cell
                        else " (no K1)"
                    ),
                    flush=True,
                )

                # Spatial maps (+6 / +24).
                if lead in spatial:
                    err = np.asarray(stud_t2m.values, dtype=np.float64) - np.asarray(
                        align_truth_to_pred(era5_t2m, stud_t2m).values, dtype=np.float64
                    )
                    accu = spatial[lead]
                    if accu["sum_err"] is None:
                        accu["sum_err"] = np.zeros_like(err)
                        accu["sum_err2"] = np.zeros_like(err)
                        accu["lat"] = np.asarray(stud_t2m.latitude.values)
                        accu["lon"] = np.asarray(stud_t2m.longitude.values)
                    finite = np.isfinite(err)
                    accu["sum_err"] = np.where(finite, accu["sum_err"] + err, accu["sum_err"])
                    accu["sum_err2"] = np.where(
                        finite, accu["sum_err2"] + err**2, accu["sum_err2"]
                    )
                    accu["n"] += 1

                # Sparse tp dry-collapse note (first init of each season at +6h only).
                if lead == 6 and season not in {r.get("season") for r in tp_note_samples}:
                    try:
                        era5_tp = truth.at("tp", vt)
                        stud_tp = _student_pred_to_africa_da(
                            pred[0],
                            name="tp",
                            lat_full=lat_full,
                            lon_full=lon_full,
                            template=era5_tp,
                        )
                        tp_max = float(np.nanmax(stud_tp.values))
                        tp_note_samples.append(
                            {
                                "init_date": init,
                                "season": season,
                                "lead_hours": 6,
                                "student_tp_max_m": tp_max,
                                "note": "tp out-of-scope; all-dry collapse if max≈0",
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        tp_note_samples.append(
                            {"init_date": init, "season": season, "error": str(exc)}
                        )

                if student_forecast_dir is not None:
                    out_nc = student_forecast_dir / f"{init}_00Z_L{lead:03d}h.nc"
                    ds_out = xr.Dataset(
                        {
                            "t2m": stud_t2m.reset_coords(drop=True).expand_dims(
                                time=[vt]
                            )
                        }
                    )
                    ds_out.attrs.update(
                        {
                            "title": f"Track B v5 analysis-forced t2m +{lead}h",
                            "reference_time": f"{init} 00:00:00",
                            "ic_time": f"{ic_date} {ic_hhmm}",
                            "checkpoint": str(ckpt),
                            "protocol": "analysis_forced_6h_step",
                        }
                    )
                    ds_out.to_netcdf(out_nc)
                    row["forecast_nc"] = str(out_nc)
        finally:
            if teach_fc is not None:
                teach_fc.close()

    figure_paths = _write_spatial_figures(spatial, figures_dir)
    aggregate = _aggregate(student_rows, k1_rows, leads=leads)

    report: dict[str, Any] = {
        "protocol": "trackB_t2m_expanded_analysis_forced",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_student": str(ckpt),
        "domain": "africa",
        "truth_source": arco_path,
        "ic_source": ic_source,
        "student_leads_hours": list(leads),
        "n_inits": len(inits),
        "inits": list(inits),
        "seasons_present": sorted({season_of(i) for i in inits}),
        "k1_coverage_inits": k1_coverage,
        "k1_coverage_n": len(k1_coverage),
        "k1_coverage_note": (
            "K1 NetCDFs only available for listed inits (typically Jan-2023 weekly). "
            "Student vs ERA5 covers all inits; degradation vs K1 only where listed."
        ),
        "variable_gate": "t2m",
        "tp_out_of_scope": True,
        "tp_dry_collapse_samples": tp_note_samples,
        "caveats": [
            "Analysis-forced multi-lead only (not free-run). Cout=3 Case A.",
            "Primary metric is t2m; tp out-of-scope (documented dry collapse).",
            "Student IC and surface truth from public ARCO ERA5 (no CDS).",
            "K1 comparison only where teacher forecast NCs exist.",
            "Same scoring methodology (cosine-lat RMSE/ACC/bias vs ERA5) for student and K1.",
        ],
        "student_results": student_rows,
        "k1_results": k1_rows,
        "aggregate": aggregate,
        "figures": figure_paths,
        "verdict": _verdict(aggregate, k1_coverage),
    }
    truth.close()
    return report


def _write_spatial_figures(
    spatial: dict[int, dict[str, Any]], figures_dir: Path
) -> dict[str, str]:
    paths: dict[str, str] = {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[figures] matplotlib unavailable: {exc}", flush=True)
        return paths

    for lead, accu in spatial.items():
        if not accu["n"] or accu["sum_err"] is None:
            continue
        n = float(accu["n"])
        mean_bias = accu["sum_err"] / n
        mean_rmse = np.sqrt(accu["sum_err2"] / n)
        lat = accu["lat"]
        lon = accu["lon"]

        for kind, field, cmap, label in (
            ("bias", mean_bias, "RdBu_r", "Mean bias (K)"),
            ("rmse", mean_rmse, "viridis", "RMSE (K)"),
        ):
            fig, ax = plt.subplots(figsize=(8, 5.5), dpi=140)
            vmax = float(np.nanpercentile(np.abs(field), 98)) if kind == "bias" else float(
                np.nanpercentile(field, 98)
            )
            vmin = -vmax if kind == "bias" else 0.0
            im = ax.pcolormesh(
                lon, lat, field, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax
            )
            ax.set_title(f"Student v5 t2m {kind} Africa +{lead}h (n={int(n)} inits)")
            ax.set_xlabel("longitude")
            ax.set_ylabel("latitude")
            fig.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)
            fig.tight_layout()
            out = figures_dir / f"trackb_t2m_v5_{kind}_L{lead:03d}h.png"
            fig.savefig(out)
            plt.close(fig)
            paths[f"{kind}_L{lead}h"] = str(out)
            print(f"[figures] wrote {out}", flush=True)

        # Also dump numeric grids for reproducibility (small).
        npz = figures_dir / f"trackb_t2m_v5_spatial_L{lead:03d}h.npz"
        np.savez_compressed(
            npz,
            lat=lat,
            lon=lon,
            mean_bias=mean_bias,
            mean_rmse=mean_rmse,
            n=np.array([n]),
        )
        paths[f"spatial_npz_L{lead}h"] = str(npz)
    return paths


def _mean(xs: list[float]) -> float | None:
    return float(np.mean(xs)) if xs else None


def _aggregate(
    student_rows: list[dict[str, Any]],
    k1_rows: list[dict[str, Any]],
    *,
    leads: tuple[int, ...],
) -> dict[str, Any]:
    by_lead: dict[str, Any] = {}
    by_season_lead: dict[str, Any] = defaultdict(dict)

    for lead in leads:
        s_rmse, s_acc, s_bias = [], [], []
        t_rmse, t_acc, t_bias, degs = [], [], [], []
        for r in student_rows:
            if r.get("lead_hours") != lead:
                continue
            cell = (r.get("variables") or {}).get("t2m") or {}
            if "student_rmse_vs_era5" not in cell:
                continue
            s_rmse.append(cell["student_rmse_vs_era5"])
            s_acc.append(cell["student_acc"])
            s_bias.append(cell["student_bias"])
            if "teacher_rmse_vs_era5" in cell:
                t_rmse.append(cell["teacher_rmse_vs_era5"])
                t_acc.append(cell["teacher_acc"])
                t_bias.append(cell["teacher_bias"])
                degs.append(cell["degradation_pct_vs_teacher"])
        by_lead[str(lead)] = {
            "n_student": len(s_rmse),
            "n_vs_k1": len(degs),
            "student_rmse_mean": _mean(s_rmse),
            "student_acc_mean": _mean(s_acc),
            "student_bias_mean": _mean(s_bias),
            "teacher_rmse_mean": _mean(t_rmse),
            "teacher_acc_mean": _mean(t_acc),
            "teacher_bias_mean": _mean(t_bias),
            "degradation_pct_mean": _mean(degs),
        }

    seasons = sorted({r.get("season") for r in student_rows if r.get("season")})
    for season in seasons:
        for lead in leads:
            s_rmse, s_acc, s_bias, degs = [], [], [], []
            for r in student_rows:
                if r.get("season") != season or r.get("lead_hours") != lead:
                    continue
                cell = (r.get("variables") or {}).get("t2m") or {}
                if "student_rmse_vs_era5" not in cell:
                    continue
                s_rmse.append(cell["student_rmse_vs_era5"])
                s_acc.append(cell["student_acc"])
                s_bias.append(cell["student_bias"])
                if "degradation_pct_vs_teacher" in cell:
                    degs.append(cell["degradation_pct_vs_teacher"])
            by_season_lead[season][str(lead)] = {
                "n_student": len(s_rmse),
                "n_vs_k1": len(degs),
                "student_rmse_mean": _mean(s_rmse),
                "student_acc_mean": _mean(s_acc),
                "student_bias_mean": _mean(s_bias),
                "degradation_pct_mean": _mean(degs),
            }

    return {"by_lead": by_lead, "by_season_lead": dict(by_season_lead)}


def _verdict(aggregate: dict[str, Any], k1_coverage: list[str]) -> dict[str, Any]:
    by_lead = aggregate.get("by_lead") or {}
    l6 = by_lead.get("6") or {}
    l24 = by_lead.get("24") or {}
    seasonal = aggregate.get("by_season_lead") or {}

    # Q1: +6h strong outside January?
    jan_like = []
    non_jan = []
    for season, leads in seasonal.items():
        row = leads.get("6") or {}
        if row.get("student_rmse_mean") is None:
            continue
        # DJF includes Jan; treat by calendar season rather than month.
        (jan_like if season == "DJF" else non_jan).append(row["student_rmse_mean"])

    rmse6 = l6.get("student_rmse_mean")
    rmse24 = l24.get("student_rmse_mean")
    growth = None
    if rmse6 and rmse24 and rmse6 > 0:
        growth = 100.0 * (rmse24 - rmse6) / rmse6

    # Persistence of +24h degradation vs K1 (only where K1 exists).
    deg6 = l6.get("degradation_pct_mean")
    deg24 = l24.get("degradation_pct_mean")
    persist_vs_k1 = None
    if deg24 is not None:
        persist_vs_k1 = bool(deg24 > 50.0)  # large gap persists if >>15% skill envelope

    # Across seasons: does student +24h RMSE stay much worse than +6h?
    season_growth = {}
    for season, leads in seasonal.items():
        a = (leads.get("6") or {}).get("student_rmse_mean")
        b = (leads.get("24") or {}).get("student_rmse_mean")
        if a and b and a > 0:
            season_growth[season] = 100.0 * (b - a) / a
    persist_across_seasons = (
        bool(season_growth) and all(g > 40.0 for g in season_growth.values())
    )

    return {
        "plus6h_strong_outside_djf": (
            bool(non_jan)
            and rmse6 is not None
            and float(np.mean(non_jan)) < 2.0
            and float(np.mean(non_jan)) < 1.5 * (float(np.mean(jan_like)) if jan_like else rmse6)
        ),
        "plus6h_rmse_overall": rmse6,
        "plus6h_rmse_non_djf_mean": float(np.mean(non_jan)) if non_jan else None,
        "skill_growth_6_to_24_pct": growth,
        "plus24h_degradation_vs_k1_pct": deg24,
        "plus6h_degradation_vs_k1_pct": deg6,
        "plus24h_degradation_vs_k1_persists": persist_vs_k1,
        "plus24h_rmse_growth_persists_across_seasons": persist_across_seasons,
        "season_rmse_growth_6_to_24_pct": season_growth,
        "k1_coverage_n": len(k1_coverage),
        "summary": (
            f"+6h t2m RMSE={rmse6}; +24h RMSE={rmse24} "
            f"(+{growth:.0f}% vs +6h). "
            f"K1 deg +6h={deg6}; +24h={deg24} (K1 n={len(k1_coverage)}). "
            f"24h growth persists across seasons={persist_across_seasons}."
            if rmse6 is not None and rmse24 is not None and growth is not None
            else "Incomplete aggregates."
        ),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    agg = report["aggregate"]
    by_lead = agg["by_lead"]
    seasonal = agg["by_season_lead"]
    v = report["verdict"]
    lines = [
        "# Track B — Expanded t2m analysis-forced evaluation (v5 freeze)",
        "",
        f"**Timestamp (UTC):** {report['timestamp_utc']}",
        f"**Checkpoint:** `{report['candidate_student']}`",
        f"**N inits:** {report['n_inits']} (seasons: {', '.join(report['seasons_present'])})",
        f"**Leads:** {report['student_leads_hours']}",
        f"**K1 coverage:** {report['k1_coverage_n']} inits — {report['k1_coverage_inits']}",
        "",
        "## Protocol",
        "",
        "- Analysis-forced only (Case A / Cout=3): IC at `init+(L−6)h` → one +6h step.",
        "- IC + truth: public ARCO ERA5 (no CDS).",
        "- Gate variable: **t2m**. tp out-of-scope (dry-collapse).",
        "- Student and K1 scored with the same cosine-latitude RMSE / ACC / bias vs ERA5.",
        "",
        "## Lead-time table (t2m, all inits)",
        "",
        "| lead | n | student RMSE | student ACC | student bias | K1 RMSE | K1 ACC | deg vs K1 |",
        "|-----:|--:|-------------:|------------:|-------------:|--------:|-------:|----------:|",
    ]
    for lead in report["student_leads_hours"]:
        row = by_lead.get(str(lead)) or {}
        lines.append(
            "| {lead} h | {n} | {srmse} | {sacc} | {sbias} | {trmse} | {tacc} | {deg} |".format(
                lead=lead,
                n=row.get("n_student", 0),
                srmse=_fmt(row.get("student_rmse_mean")),
                sacc=_fmt(row.get("student_acc_mean"), 3),
                sbias=_fmt(row.get("student_bias_mean")),
                trmse=_fmt(row.get("teacher_rmse_mean")),
                tacc=_fmt(row.get("teacher_acc_mean"), 3),
                deg=_fmt(row.get("degradation_pct_mean"), 1, suffix="%"),
            )
        )

    lines += ["", "## Seasonal breakdown (student vs ERA5)", ""]
    for season in sorted(seasonal):
        lines.append(f"### {season}")
        lines.append("")
        lines.append("| lead | n | RMSE | ACC | bias | deg vs K1 |")
        lines.append("|-----:|--:|-----:|----:|-----:|----------:|")
        for lead in report["student_leads_hours"]:
            row = seasonal[season].get(str(lead)) or {}
            lines.append(
                "| {lead} h | {n} | {rmse} | {acc} | {bias} | {deg} |".format(
                    lead=lead,
                    n=row.get("n_student", 0),
                    rmse=_fmt(row.get("student_rmse_mean")),
                    acc=_fmt(row.get("student_acc_mean"), 3),
                    bias=_fmt(row.get("student_bias_mean")),
                    deg=_fmt(row.get("degradation_pct_mean"), 1, suffix="%"),
                )
            )
        lines.append("")

    lines += [
        "## Critical questions",
        "",
        f"1. **Does +6h t2m stay strong outside January/DJF?** "
        f"{'Yes' if v.get('plus6h_strong_outside_djf') else 'No / mixed'} — "
        f"non-DJF mean RMSE={_fmt(v.get('plus6h_rmse_non_djf_mean'))} "
        f"(overall +6h={_fmt(v.get('plus6h_rmse_overall'))}).",
        f"2. **How fast does skill deteriorate through 24h?** "
        f"RMSE growth +6→+24h = {_fmt(v.get('skill_growth_6_to_24_pct'), 0, suffix='%')}. "
        f"Seasonal growth: {v.get('season_rmse_growth_6_to_24_pct')}.",
        f"3. **Does +24h degradation vs K1 persist?** "
        f"{'Yes' if v.get('plus24h_degradation_vs_k1_persists') else 'No / insufficient K1'} "
        f"(+24h deg={_fmt(v.get('plus24h_degradation_vs_k1_pct'), 1, suffix='%')}; "
        f"K1 n={v.get('k1_coverage_n')}). "
        f"Across-season +24h RMSE growth persistence: "
        f"{v.get('plus24h_rmse_growth_persists_across_seasons')}.",
        "",
        f"**Verdict summary:** {v.get('summary')}",
        "",
        "## tp (out of scope)",
        "",
        "tp remains out-of-scope for Track B promotion. Samples:",
        "```json",
        json.dumps(report.get("tp_dry_collapse_samples"), indent=2),
        "```",
        "",
        "## Figures",
        "",
    ]
    for k, p in (report.get("figures") or {}).items():
        lines.append(f"- `{k}`: `{p}`")
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "CUDA_VISIBLE_DEVICES=1 MKL_INTERFACE_LAYER=GNU \\",
        "  python -u evaluation/trackB_t2m_expanded.py \\",
        "    --ckpt models/student_global_stable_v5.ckpt --device cuda \\",
        "    --ic_source arco --step_days 5 --leads 6,12,18,24",
        "```",
        "",
        "See also `scripts/run_trackB_t2m_expanded_cassava.sh` and README Track B section.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}", flush=True)


def _fmt(x: Any, digits: int = 3, suffix: str = "") -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "—"


def main() -> int:
    from utils.cds_ic import AIFS_AFRICA_CACHE_DIR

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=_REPO_ROOT / "models/student_global_stable_v5.ckpt")
    p.add_argument("--device", default=None)
    p.add_argument("--ic_source", choices=("auto", "cds", "arco"), default="arco")
    p.add_argument("--arco_path", default=ARCO_DEFAULT)
    p.add_argument("--state_cache_dir", type=Path, default=_REPO_ROOT / DEFAULT_STATE_CACHE)
    p.add_argument("--step_days", type=int, default=5, help="Init spacing in days across 2023")
    p.add_argument(
        "--months",
        default="",
        help="Optional comma months 1-12 (default: all). Example: 1,4,7,10",
    )
    p.add_argument("--leads", default="6,12,18,24")
    p.add_argument(
        "--inits",
        default="",
        help="Optional explicit YYYYMMDD list (overrides --step_days/--months)",
    )
    p.add_argument(
        "--teacher_fc_dir",
        type=Path,
        default=_REPO_ROOT / "data/processed/trackA_prune_k1_a1bplus/forecasts",
    )
    p.add_argument(
        "--student_forecast_dir",
        type=Path,
        default=_REPO_ROOT / "data/processed/trackB_t2m_expanded/forecasts",
    )
    p.add_argument("--no_write_nc", action="store_true")
    p.add_argument("--figures_dir", type=Path, default=_REPO_ROOT / "reports/figures")
    p.add_argument("--out_json", type=Path, default=_REPO_ROOT / "reports/TRACKB_T2M_EXPANDED.json")
    p.add_argument("--out_md", type=Path, default=_REPO_ROOT / "reports/TRACKB_T2M_EXPANDED.md")
    p.add_argument("--cds_cache", type=Path, default=Path(AIFS_AFRICA_CACHE_DIR))
    p.add_argument(
        "--n320_latlon",
        type=Path,
        default=_REPO_ROOT / "data/processed/lapai/n320_latlons.npy",
    )
    args = p.parse_args()

    leads = tuple(int(x) for x in args.leads.split(",") if x.strip())
    if args.inits.strip():
        inits = tuple(x.strip() for x in args.inits.split(",") if x.strip())
    else:
        months = (
            tuple(int(x) for x in args.months.split(",") if x.strip())
            if args.months.strip()
            else None
        )
        inits = expand_inits_2023(step_days=args.step_days, months=months)

    import torch

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev)
    print(
        f"[expanded] n_inits={len(inits)} leads={leads} device={device} "
        f"ckpt={args.ckpt} ic_source={args.ic_source}",
        flush=True,
    )

    report = run_expanded(
        ckpt=args.ckpt,
        inits=inits,
        leads=leads,
        device=device,
        ic_source=args.ic_source,
        arco_path=args.arco_path,
        state_cache_dir=args.state_cache_dir,
        teacher_fc_dir=args.teacher_fc_dir,
        student_forecast_dir=None if args.no_write_nc else args.student_forecast_dir,
        figures_dir=args.figures_dir,
        cds_cache=args.cds_cache,
        n320_latlon=args.n320_latlon,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out_json}", flush=True)
    write_markdown(report, args.out_md)
    print(json.dumps(report["aggregate"], indent=2), flush=True)
    print(json.dumps(report["verdict"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

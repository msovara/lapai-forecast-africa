"""Phase 0 baseline scorecard: multi-init, multi-lead scoring vs GCS or local ERA5 truth."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from evaluation.gcs_era5_truth import align_truth_to_pred, open_era5_at_valid_time
from evaluation.run_scorecard import _read_eval_config, _resolve_domain
from lapai_inference.cache_schema import cosine_latitude_weights

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Forecast file short names -> scorecard names.
FC_ALIASES = {
    "2t": "t2m",
    "t2m": "t2m",
    "tp": "tp",
    "10u": "u10",
    "u10": "u10",
    "10v": "v10",
    "v10": "v10",
}


def load_phase0_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or (_REPO_ROOT / "configs" / "phase0_baseline.yaml")
    return _read_eval_config(cfg_path)


def _resolve_fc_var(ds: xr.Dataset, eval_var: str) -> str | None:
    for fc_name, alias in FC_ALIASES.items():
        if alias == eval_var and fc_name in ds:
            return fc_name
    return eval_var if eval_var in ds else None


def forecast_step_and_valid_time(fc: xr.Dataset, lead_hours: int) -> tuple[int, np.datetime64]:
    step = lead_hours // 6 - 1
    if step < 0 or step >= fc.sizes["time"]:
        raise ValueError(
            f"lead_hours {lead_hours} invalid for time size {fc.sizes['time']} "
            f"(need {(fc.sizes['time'] + 1) * 6}h rollout)"
        )
    return step, np.datetime64(fc.time.isel(time=step).values)


def _eval_africa_box() -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Lat/lon box from configs/eval.yaml africa region (source of truth)."""
    cfg = _read_eval_config(_REPO_ROOT / "configs" / "eval.yaml")
    resolved = _resolve_domain(cfg, "africa")
    if resolved is None:
        return None
    return resolved[1], resolved[2]


def _score_point(
    pred: xr.DataArray,
    truth: xr.DataArray,
) -> dict[str, float]:
    # Crop wider forecast files (e.g. A2 -20:70) onto the eval.yaml box (A1 -20:55)
    # so A1 vs A2 scoring is on the same 321 x 301 grid. Existing nc files are not rewritten.
    box = _eval_africa_box()
    if box is not None:
        lat_range, lon_range = box
        if "longitude" in pred.dims:
            pred = _subset_eval_domain(pred, lat_range, lon_range)
        if "longitude" in truth.dims:
            truth = _subset_eval_domain(truth, lat_range, lon_range)
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
        return {"rmse": float("nan"), "acc": float("nan")}

    err2 = (p - t) ** 2
    rmse = float(np.sqrt(np.sum(w2d[mask] * err2[mask]) / np.sum(w2d[mask])))

    clim = np.nanmean(t, axis=0, keepdims=True)
    pred_anom = p - clim
    ref_anom = t - clim
    num = np.sum(w2d[mask] * pred_anom[mask] * ref_anom[mask])
    den = np.sqrt(
        np.sum(w2d[mask] * pred_anom[mask] ** 2)
        * np.sum(w2d[mask] * ref_anom[mask] ** 2)
    )
    acc = float(num / den) if den > 0 else float("nan")
    out = {"rmse": rmse, "acc": acc}
    var_name = str(getattr(pred, "name", "") or pred.attrs.get("name", ""))
    if var_name == "tp":
        threshold = 0.001
        out["pod_1mm"] = float(np.sum((p >= threshold) & (t >= threshold) & mask) / np.sum((t >= threshold) & mask))
    return out


def score_forecast_gcs(
    forecast_path: Path,
    *,
    lead_hours: int,
    variables: list[str],
    gcs_bucket: str = "gs://code4earth/era5",
    init_date: str | None = None,
) -> dict[str, Any]:
    fc = xr.open_dataset(forecast_path)
    try:
        step, valid_time = forecast_step_and_valid_time(fc, lead_hours)
        row: dict[str, Any] = {
            "forecast": str(forecast_path),
            "init_date": init_date,
            "lead_hours": lead_hours,
            "valid_time": str(valid_time),
            "variables": {},
        }
        for eval_var in variables:
            fc_name = _resolve_fc_var(fc, eval_var)
            if fc_name is None:
                row["variables"][eval_var] = {"error": f"missing in forecast file"}
                continue
            pred = fc[fc_name].isel(time=step)
            pred = pred.assign_coords(name=eval_var)
            try:
                truth = open_era5_at_valid_time(eval_var, valid_time, bucket=gcs_bucket)
                row["variables"][eval_var] = _score_point(pred, truth)
            except Exception as exc:  # noqa: BLE001
                row["variables"][eval_var] = {"error": str(exc)}
        return row
    finally:
        fc.close()


def _subset_eval_domain(da: xr.DataArray, lat_range: tuple[float, float], lon_range: tuple[float, float]) -> xr.DataArray:
    if da.latitude[0] > da.latitude[-1]:
        lat_slice = slice(lat_range[1], lat_range[0])
    else:
        lat_slice = slice(lat_range[0], lat_range[1])
    return da.sel(latitude=lat_slice, longitude=slice(lon_range[0], lon_range[1]))


def score_forecast_local(
    forecast_path: Path,
    truth_path: Path,
    *,
    lead_hours: int,
    variables: list[str],
    domain: str = "africa",
    eval_config: Path | None = None,
    init_date: str | None = None,
) -> dict[str, Any]:
    cfg = _read_eval_config(eval_config or (_REPO_ROOT / "configs" / "eval.yaml"))
    resolved = _resolve_domain(cfg, domain)
    domain_box = (resolved[1], resolved[2]) if resolved else None

    fc = xr.open_dataset(forecast_path)
    try:
        step, valid_time = forecast_step_and_valid_time(fc, lead_hours)
        row: dict[str, Any] = {
            "forecast": str(forecast_path),
            "init_date": init_date,
            "lead_hours": lead_hours,
            "valid_time": str(valid_time),
            "variables": {},
        }
        if truth_path.suffix == ".zarr" or (truth_path.is_dir() and (truth_path / ".zmetadata").exists()):
            truth_ds = xr.open_zarr(truth_path)
        else:
            truth_ds = xr.open_dataset(truth_path)
        try:
            for eval_var in variables:
                fc_name = _resolve_fc_var(fc, eval_var)
                if fc_name is None:
                    row["variables"][eval_var] = {"error": "missing in forecast file"}
                    continue
                if eval_var not in truth_ds and fc_name not in truth_ds:
                    row["variables"][eval_var] = {"error": "missing in truth store"}
                    continue
                try:
                    pred = fc[fc_name].isel(time=step)
                    truth_var = eval_var if eval_var in truth_ds else fc_name
                    truth = truth_ds[truth_var]
                    if "time" in truth.dims:
                        tidx = int(
                            np.argmin(
                                np.abs(truth["time"].values.astype("datetime64[ns]") - valid_time)
                            )
                        )
                        truth = truth.isel(time=tidx)
                    if domain_box is not None:
                        pred = _subset_eval_domain(pred, domain_box[0], domain_box[1])
                        truth = _subset_eval_domain(truth, domain_box[0], domain_box[1])
                    pred = pred.assign_coords(name=eval_var)
                    row["variables"][eval_var] = _score_point(pred, truth)
                except Exception as exc:  # noqa: BLE001
                    row["variables"][eval_var] = {"error": str(exc)}
        finally:
            truth_ds.close()
        return row
    finally:
        fc.close()


def build_phase0_scorecard(
    forecast_paths: list[Path],
    *,
    config: dict[str, Any] | None = None,
    truth_source: str | None = None,
) -> dict[str, Any]:
    cfg = config or load_phase0_config()
    eval_cfg = _read_eval_config(_REPO_ROOT / "configs" / "eval.yaml")
    variables = list(cfg.get("variables") or eval_cfg.get("variables") or ["t2m", "tp", "u10", "v10"])
    leads = [int(x) for x in (cfg.get("leads_hours") or eval_cfg.get("leads_hours") or [24, 48, 72])]
    truth_block = cfg.get("truth") or {}
    source = truth_source or truth_block.get("source", "gcs")
    gcs_bucket = truth_block.get("gcs_bucket", "gs://code4earth/era5")
    local_zarr = truth_block.get("local_zarr")
    domain = cfg.get("domain", "africa")

    init_map = {p.stem.split("_")[0]: p for p in forecast_paths}
    results: list[dict[str, Any]] = []

    for path in sorted(forecast_paths):
        init_guess = path.name.split("_")[0]
        with xr.open_dataset(path) as fc_probe:
            max_lead = fc_probe.sizes.get("time", 0) * 6
        for lead in leads:
            if lead > max_lead:
                results.append(
                    {
                        "forecast": str(path),
                        "init_date": init_guess,
                        "lead_hours": lead,
                        "skipped": True,
                        "reason": f"forecast has {max_lead}h rollout; need {lead}h",
                    }
                )
                continue
            if source == "local":
                if not local_zarr:
                    raise ValueError("truth.source=local requires truth.local_zarr in phase0 config")
                truth_path = _REPO_ROOT / local_zarr if not Path(local_zarr).is_absolute() else Path(local_zarr)
                row = score_forecast_local(
                    path,
                    truth_path,
                    lead_hours=lead,
                    variables=variables,
                    domain=domain,
                    init_date=init_guess,
                )
            else:
                row = score_forecast_gcs(
                    path,
                    lead_hours=lead,
                    variables=variables,
                    gcs_bucket=gcs_bucket,
                    init_date=init_guess,
                )
            results.append(row)

    return {
        "phase": 0,
        "status": "baseline_teacher",
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "teacher_config": cfg.get("teacher_config"),
        "checkpoint": cfg.get("checkpoint"),
        "domain": domain,
        "variables": variables,
        "leads_hours": leads,
        "truth_source": source,
        "inits_scored": sorted(init_map.keys()),
        "forecasts": [str(p) for p in sorted(forecast_paths)],
        "results": results,
    }


def write_phase0_scorecard(scorecard: dict[str, Any], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    payload = json.dumps(_sanitize(scorecard), indent=2)
    out_path.write_text(payload + "\n", encoding="utf-8")
    return out_path


def discover_forecasts(forecast_dir: Path, pattern: str = "*_00Z.nc") -> list[Path]:
    forecast_dir = Path(forecast_dir)
    if not forecast_dir.is_dir():
        return []
    return sorted(forecast_dir.glob(pattern))

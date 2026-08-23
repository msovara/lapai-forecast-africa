"""Score GraphCast Africa GCS forecasts vs ERA5 (shared LapAI Africa contract)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from evaluation.gcs_era5_truth import open_era5_at_valid_time
from evaluation.phase0_scorecard import _score_point
from evaluation.run_scorecard import _read_eval_config, _resolve_domain
from teachers.graphcast.gcs_forecasts import (
    GCS_BUCKET_PREFIX,
    open_graphcast_forecast,
    select_init_lead,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def load_graphcast_score_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or (_REPO_ROOT / "configs" / "graphcast_africa_baseline.yaml")
    return _read_eval_config(cfg_path)


def _parse_init(init: str) -> np.datetime64:
    """Accept YYYYMMDD or YYYY-MM-DD → datetime64 at 00Z."""
    s = init.strip().replace("-", "")
    if len(s) != 8 or not s.isdigit():
        raise ValueError(f"init must be YYYYMMDD, got {init!r}")
    return np.datetime64(f"{s[:4]}-{s[4:6]}-{s[6:8]}T00:00:00", "ns")


def _valid_time(init: np.datetime64, lead_hours: int) -> np.datetime64:
    return init + np.timedelta64(int(lead_hours), "h")


def score_graphcast_point(
    *,
    var: str,
    year: int,
    init: str,
    lead_hours: int,
    forecast_bucket: str = GCS_BUCKET_PREFIX,
    truth_bucket: str = "gs://code4earth/era5",
    da_cache: dict[tuple[str, int], xr.DataArray] | None = None,
) -> dict[str, Any]:
    """Score one variable / init / lead against GCS ERA5 truth."""
    key = (var, year)
    cache = da_cache if da_cache is not None else {}
    if key not in cache:
        cache[key] = open_graphcast_forecast(var, year, bucket_prefix=forecast_bucket)
    da = cache[key]
    init64 = _parse_init(init)
    try:
        pred = select_init_lead(da, init=init64, lead_hours=float(lead_hours))
    except ValueError as exc:
        return {"error": str(exc)}
    pred = pred.assign_coords(name=var)
    # Ensure standard dim names for align_truth_to_pred
    if "latitude" not in pred.dims and "lat" in pred.dims:
        pred = pred.rename({"lat": "latitude"})
    if "longitude" not in pred.dims and "lon" in pred.dims:
        pred = pred.rename({"lon": "longitude"})
    valid = _valid_time(init64, lead_hours)
    try:
        truth = open_era5_at_valid_time(var, valid, bucket=truth_bucket)
        return _score_point(pred, truth)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def build_graphcast_scorecard(
    cfg: dict[str, Any],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    inits = list(cfg.get("inits") or [])
    leads = [int(x) for x in (cfg.get("leads_hours") or [24, 48])]
    variables = list(cfg.get("variables") or ["t2m", "u10", "v10"])
    forecast_bucket = cfg.get("forecast_bucket") or GCS_BUCKET_PREFIX
    truth_bucket = (cfg.get("truth") or {}).get("gcs_bucket") or "gs://code4earth/era5"
    eval_cfg = _read_eval_config(_REPO_ROOT / "configs" / "eval.yaml")
    resolved = _resolve_domain(eval_cfg, "africa")
    domain_box = (
        {"lat": list(resolved[1]), "lon": list(resolved[2])}
        if resolved
        else {"lat": [-40.0, 40.0], "lon": [-20.0, 55.0]}
    )

    card: dict[str, Any] = {
        "pathway": "graphcast_africa",
        "status": "baseline_teacher",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "forecast_bucket": forecast_bucket,
        "domain": "africa",
        "domain_box": domain_box,
        "variables": variables,
        "leads_hours": leads,
        "truth_source": "gcs",
        "truth_bucket": truth_bucket,
        "inits_scored": inits,
        "results": [],
        "notes": cfg.get("notes"),
    }
    if dry_run:
        card["status"] = "dry_run"
        return card

    da_cache: dict[tuple[str, int], xr.DataArray] = {}
    for init in inits:
        year = int(str(init).replace("-", "")[:4])
        for lead in leads:
            row: dict[str, Any] = {
                "init_date": str(init).replace("-", "")[:8],
                "lead_hours": lead,
                "valid_time": str(_valid_time(_parse_init(init), lead)),
                "variables": {},
            }
            for var in variables:
                row["variables"][var] = score_graphcast_point(
                    var=var,
                    year=year,
                    init=init,
                    lead_hours=lead,
                    forecast_bucket=forecast_bucket,
                    truth_bucket=truth_bucket,
                    da_cache=da_cache,
                )
            card["results"].append(row)
    return card


def write_graphcast_scorecard(card: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")

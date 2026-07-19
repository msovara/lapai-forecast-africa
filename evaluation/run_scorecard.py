#!/usr/bin/env python3
"""Multi-variable skill scorecard vs ERA5 for the Mvua evaluation protocol.

Scores the variables in ``configs/eval.yaml`` (default: t2m, tp, u10, v10) at one or
more temporal resolutions (6h, daily) against ERA5 ground truth, emitting a JSON
scorecard (and optional Markdown table). Precipitation (tp) is accumulated when
aggregating to daily; other variables are averaged.

Usage:
  python -m evaluation.run_scorecard --pred forecast.nc --truth era5.nc
  python -m evaluation.run_scorecard --pred fc.zarr --truth era5.zarr \\
      --variables t2m,tp --temporal 6h,daily --out scorecard.json --markdown
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Daily aggregation reducer per variable; precipitation accumulates, everything else averages.
_DAILY_REDUCER = {"tp": "sum"}
_DEFAULT_REDUCER = "mean"
_TIME_DIM_CANDIDATES = ("time", "valid_time", "forecast_time", "step")
_DEFAULT_VARIABLES = ["t2m", "tp", "u10", "v10"]
_DEFAULT_TEMPORAL = ["6h"]


def _read_eval_config(path: Path) -> dict:
    """Prefer PyYAML; fall back to a tiny reader for flow-style lists / scalars."""
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except Exception:  # noqa: BLE001 - PyYAML missing or parse hiccup; use minimal reader
        out: dict = {}
        for line in text.splitlines():
            stripped = line.split("#", 1)[0].rstrip()
            if not stripped or stripped[0] in " -" or ":" not in stripped:
                continue
            key, val = stripped.split(":", 1)
            key = key.strip()
            val = val.strip()
            if val.startswith("[") and val.endswith("]"):
                out[key] = [x.strip().strip("\"'") for x in val[1:-1].split(",") if x.strip()]
            elif val:
                out[key] = val.strip("\"'")
        return out


def _split_csv(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [x.strip() for x in raw.split(",") if x.strip()]


def _find_time_dim(da) -> str | None:  # type: ignore[no-untyped-def]
    return next((c for c in _TIME_DIM_CANDIDATES if c in da.dims), None)


def _aggregate_daily(da, var: str):  # type: ignore[no-untyped-def]
    tdim = _find_time_dim(da)
    if tdim is None:
        raise ValueError(f"no time-like dim in {tuple(da.dims)} for daily aggregation")
    grouped = da.resample({tdim: "1D"})
    reducer = _DAILY_REDUCER.get(var, _DEFAULT_REDUCER)
    return grouped.sum() if reducer == "sum" else grouped.mean()


def _resolve_domain(cfg: dict, override: str | None) -> tuple[str, tuple, tuple] | None:
    """Return (region_name, lat_range, lon_range) for cropping, or None for full grid."""
    dom = cfg.get("domain") or {}
    name = override or dom.get("default")
    if not name:
        return None
    regions = dom.get("regions") or {}
    region = regions.get(name)
    if region is None:
        # 'global' is a no-op even without an explicit config entry.
        if name == "global":
            return None
        raise SystemExit(f"unknown domain {name!r}; configured: {sorted(regions)}")
    lat = region.get("lat")
    lon = region.get("lon")
    if not (lat and lon):
        raise SystemExit(f"domain {name!r} missing lat/lon ranges")
    if name == "global":
        return None
    return name, (float(lat[0]), float(lat[1])), (float(lon[0]), float(lon[1]))


def _crop_domain(da, lat_range: tuple, lon_range: tuple):  # type: ignore[no-untyped-def]
    """Crop to a lat/lon box, tolerant of 0..360 vs -180..180 longitude conventions."""
    from evaluation.eval_skill import _resolve_lat_lon_names

    dims = tuple(str(d) for d in da.dims)
    lat_n, lon_n = _resolve_lat_lon_names(dims)
    if lat_n is None or lon_n is None:
        return da
    lat = da[lat_n]
    lon180 = ((da[lon_n] + 180) % 360) - 180
    mask = (
        (lat >= lat_range[0])
        & (lat <= lat_range[1])
        & (lon180 >= lon_range[0])
        & (lon180 <= lon_range[1])
    )
    return da.where(mask, drop=True)


def _to_tensors(ap, at):  # type: ignore[no-untyped-def]
    """Aligned DataArrays -> (pred[T,1,H,W], truth[T,1,H,W], lat_weights[H])."""
    import torch
    import xarray as xr

    from evaluation.eval_skill import _resolve_lat_lon_names
    from lapai_inference.cache_schema import cosine_latitude_weights

    ap, at = xr.align(ap, at, join="inner")
    dims = tuple(str(d) for d in ap.dims)
    lat_n, lon_n = _resolve_lat_lon_names(dims)
    if lat_n is None or lon_n is None:
        raise ValueError(f"could not infer lat/lon from {dims}; rename to latitude/longitude")
    other = [d for d in ap.dims if d not in (lat_n, lon_n)]
    order = (*other, lat_n, lon_n)
    ap = ap.transpose(*order)
    at = at.transpose(*order)

    lat_vals = np.asarray(ap.coords[lat_n].data).astype(np.float64).squeeze()
    if lat_vals.ndim != 1:
        raise ValueError("latitude coordinate must be 1-D after transpose")
    w = torch.from_numpy(cosine_latitude_weights(lat_vals).squeeze().astype(np.float32))

    pv = torch.from_numpy(np.ascontiguousarray(ap.values.astype(np.float32)))
    tv = torch.from_numpy(np.ascontiguousarray(at.values.astype(np.float32)))
    if not pv.shape[:-2]:
        pv = pv.unsqueeze(0)
        tv = tv.unsqueeze(0)
    else:
        pv = pv.reshape(-1, pv.shape[-2], pv.shape[-1])
        tv = tv.reshape(-1, tv.shape[-2], tv.shape[-1])
    return pv.unsqueeze(1), tv.unsqueeze(1), w


def _score_variable(
    pred_path: Path,
    truth_path: Path,
    var: str,
    temporal: list[str],
    isel_kw: dict,
    domain_box: tuple | None = None,
) -> list[dict]:
    from evaluation.eval_skill import (
        _dataarray_from_dataset,
        _open_xarray,
        _parse_isel_arg,
        compute_skill_metrics,
    )

    rows: list[dict] = []
    dsp = _open_xarray(pred_path)
    dst = _open_xarray(truth_path)
    try:
        ap = _dataarray_from_dataset(dsp, var)
        at = _dataarray_from_dataset(dst, var)
        if isel_kw:
            ap = ap.isel(**isel_kw)
            at = at.isel(**isel_kw)
        if domain_box is not None:
            ap = _crop_domain(ap, domain_box[0], domain_box[1])
            at = _crop_domain(at, domain_box[0], domain_box[1])
        for res in temporal:
            if res == "daily":
                ap_r = _aggregate_daily(ap, var)
                at_r = _aggregate_daily(at, var)
            else:  # "6h" or native: use as-is
                ap_r, at_r = ap, at
            pv, tv, w = _to_tensors(ap_r, at_r)
            metrics = compute_skill_metrics(pv, tv, w)
            rows.append(
                {
                    "variable": var,
                    "temporal": res,
                    "n_times": int(pv.shape[0]),
                    "rmse": metrics["rmse"],
                    "acc": metrics["acc"],
                }
            )
    finally:
        dsp.close()
        dst.close()
    return rows


def _markdown_table(rows: list[dict]) -> str:
    head = "| variable | temporal | n_times | RMSE | ACC |\n|---|---|---|---|---|"
    body = "\n".join(
        f"| {r['variable']} | {r['temporal']} | {r.get('n_times', '-')} | "
        f"{r['rmse']:.4g} | {r['acc']:.4g} |"
        if "rmse" in r
        else f"| {r['variable']} | {r['temporal']} | - | ERROR: {r.get('error', '?')} | - |"
        for r in rows
    )
    return f"{head}\n{body}\n"


def main() -> int:
    from evaluation.eval_skill import _parse_isel_arg

    p = argparse.ArgumentParser(description="Mvua multi-variable skill scorecard vs ERA5")
    p.add_argument("--pred", type=Path, required=True, help="Forecast NetCDF or Zarr store")
    p.add_argument("--truth", type=Path, required=True, help="ERA5 ground-truth NetCDF or Zarr store")
    p.add_argument(
        "--config",
        type=Path,
        default=_REPO_ROOT / "configs" / "eval.yaml",
        help="Eval protocol YAML (default: configs/eval.yaml)",
    )
    p.add_argument("--variables", default=None, help="Comma list overriding config (e.g. t2m,tp,u10,v10)")
    p.add_argument("--temporal", default=None, help="Comma list overriding config (e.g. 6h,daily)")
    p.add_argument(
        "--domain",
        default=None,
        help="Evaluation domain from config (e.g. africa, global). Default: config domain.default",
    )
    p.add_argument("--isel", default=None, help="xarray.isel applied to both, e.g. step=-1")
    p.add_argument("--out", type=Path, default=None, help="Write JSON scorecard here (default: stdout)")
    p.add_argument("--markdown", action="store_true", help="Also print a Markdown table to stderr")
    args = p.parse_args()

    cfg = _read_eval_config(args.config)
    variables = _split_csv(args.variables) or cfg.get("variables") or _DEFAULT_VARIABLES
    temporal = _split_csv(args.temporal) or cfg.get("temporal_resolutions") or _DEFAULT_TEMPORAL
    isel_kw = _parse_isel_arg(args.isel)

    resolved = _resolve_domain(cfg, args.domain)
    domain_name = resolved[0] if resolved else "global"
    domain_box = (resolved[1], resolved[2]) if resolved else None

    print(
        f"# scorecard: ground_truth={cfg.get('ground_truth', 'era5')} "
        f"variables={variables} temporal={temporal} domain={domain_name}",
        file=sys.stderr,
    )

    rows: list[dict] = []
    for var in variables:
        try:
            rows.extend(_score_variable(args.pred, args.truth, var, temporal, isel_kw, domain_box))
        except Exception as exc:  # noqa: BLE001 - record per-variable failure, keep scoring others
            print(f"[scorecard] {var}: {exc}", file=sys.stderr)
            for res in temporal:
                rows.append({"variable": var, "temporal": res, "error": str(exc)})

    scorecard = {
        "ground_truth": cfg.get("ground_truth", "era5"),
        "test_period": cfg.get("test_period"),
        "domain": domain_name,
        "results": rows,
    }
    payload = json.dumps(scorecard, indent=2)
    if args.out is not None:
        args.out.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(payload)

    if args.markdown:
        print(_markdown_table(rows), file=sys.stderr)

    return 0 if any("rmse" in r for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

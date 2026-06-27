"""Global skill metrics vs ERA5 analysis (RMSE, ACC hooks)."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import torch

from evaluation.masks import anomaly_correlation_coefficient, area_rmse
from lapai_inference.cache_schema import cosine_latitude_weights, lat_lon_mesh


def compute_skill_metrics(pred: torch.Tensor, era5: torch.Tensor, w: torch.Tensor) -> dict[str, float]:
    """Cosine‑latitude weighted RMSE and ACC (pred anomalies vs ERA5 anomalies)."""
    rmse = area_rmse(pred, era5, w)
    clim = era5.mean(dim=0, keepdim=True)
    acc = anomaly_correlation_coefficient(pred - clim, era5 - clim, w)
    return {"rmse": float(rmse), "acc": float(acc)}


def load_torch(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    pred = blob["pred"]
    era5 = blob["era5"]
    return pred, era5


def _resolve_lat_lon_names(dims: tuple[str, ...]) -> tuple[str | None, str | None]:
    lat_candidates = {"latitude", "lat", "y"}
    lon_candidates = {"longitude", "lon", "x"}
    lat = next((d for d in dims if d in lat_candidates), None)
    lon = next((d for d in dims if d in lon_candidates), None)
    return lat, lon


def _parse_isel_arg(raw: str | None) -> dict:
    """Parse `time=0,step=-1`-style indexer for xarray.isel."""
    if not raw or not raw.strip():
        return {}
    out: dict = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid --isel fragment {part!r} (want key=value)")
        k, v = part.split("=", 1)
        key = k.strip()
        expr = v.strip()
        try:
            out[key] = int(expr)
        except ValueError:
            out[key] = ast.literal_eval(expr)
    return out


def _open_xarray(path: Path):  # type: ignore[no-untyped-def]
    try:
        import xarray as xr
    except ImportError as exc:
        raise SystemExit(
            "NetCDF / Zarr mode needs xarray. Install: pip install -e '.[data]' "
            "or conda/pip install xarray netcdf4"
        ) from exc

    p = path.resolve()
    if p.suffix in {".zarr"} or (p.is_dir() and (p / ".zmetadata").exists()):
        return xr.open_zarr(str(p))
    return xr.open_dataset(str(p))


_VAR_ALIASES: dict[str, tuple[str, ...]] = {
    "t2m": ("t2m", "2t"),
    "tp": ("tp",),
    "u10": ("u10", "10u"),
    "v10": ("v10", "10v"),
}


def _dataarray_from_dataset(ds, var: str):  # type: ignore[no-untyped-def]
    if isinstance(var, str) and var in ds:
        return ds[var]
    for alt in _VAR_ALIASES.get(var, ()):
        if alt in ds:
            return ds[alt]
    raise KeyError(f"Variable {var!r} not in dataset")


def load_xarray_pair(
    pred_path: Path,
    truth_path: Path,
    var: str,
    isel_kw: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load grids from two NetCDF/Zarr datasets; return (pred, truth, lat_cos_w) tensors."""
    import xarray as xr

    dsp = _open_xarray(pred_path)
    dst = _open_xarray(truth_path)
    try:
        ap = _dataarray_from_dataset(dsp, var)
        at = _dataarray_from_dataset(dst, var)
        if isel_kw:
            ap = ap.isel(**isel_kw)
            at = at.isel(**isel_kw)
        ap, at = xr.align(ap, at, join="inner")
        dim_names = tuple(str(d) for d in ap.dims)
        lat_n, lon_n = _resolve_lat_lon_names(dim_names)
        if lat_n is None or lon_n is None:
            raise ValueError(
                f"Could not infer lat/lon dims from {dim_names!r}; "
                "rename to latitude/longitude or lat/lon."
            )
        # Order as (..., lat, lon) for mesh weights
        other = [d for d in ap.dims if d not in (lat_n, lon_n)]
        order = (*other, lat_n, lon_n)
        ap = ap.transpose(*order)
        at = at.transpose(*order)
        lat_vals = np.asarray(ap.coords[lat_n].data).astype(np.float64).squeeze()
        if lat_vals.ndim != 1:
            raise ValueError("Latitude coordinate must be 1-D after isel.")
        wnp = cosine_latitude_weights(lat_vals).squeeze().astype(np.float32)

        pv = torch.from_numpy(np.ascontiguousarray(ap.values.astype(np.float32)))
        tv = torch.from_numpy(np.ascontiguousarray(at.values.astype(np.float32)))

        extra = pv.shape[:-2]
        if not extra:
            pv = pv.unsqueeze(0)
            tv = tv.unsqueeze(0)
        else:
            pv = pv.reshape(-1, pv.shape[-2], pv.shape[-1])
            tv = tv.reshape(-1, tv.shape[-2], tv.shape[-1])
        pv = pv.unsqueeze(1)
        tv = tv.unsqueeze(1)
        w = torch.from_numpy(wnp)
        return pv, tv, w
    finally:
        dsp.close()
        dst.close()


def main() -> None:
    p = argparse.ArgumentParser(
        description="RMSE / ACC vs ERA5 (torch blob or aligned NetCDF/Zarr pair)"
    )
    p.add_argument(
        "--blob",
        type=Path,
        default=None,
        help="torch save dict with keys pred, era5 [B,C,H,W]",
    )
    p.add_argument(
        "--pred-netcdf",
        type=Path,
        default=None,
        help="Forecast file (NetCDF or zarr store path)",
    )
    p.add_argument(
        "--truth-netcdf",
        type=Path,
        default=None,
        help="Reference file (ERA5 analysis, etc.)",
    )
    p.add_argument(
        "--var",
        type=str,
        default=None,
        help="Variable name in both datasets (required for NetCDF/Zarr mode)",
    )
    p.add_argument(
        "--isel",
        type=str,
        default=None,
        help="Comma-separated xarray.isel, e.g. time=0,step=7 (same applied to pred and truth)",
    )
    args = p.parse_args()

    netcdf_mode = args.pred_netcdf is not None and args.truth_netcdf is not None
    blob_mode = args.blob is not None

    if blob_mode == netcdf_mode:
        raise SystemExit("Provide exactly one of: --blob OR (--pred-netcdf and --truth-netcdf)")

    if netcdf_mode:
        if args.var is None:
            raise SystemExit("--var is required for NetCDF/Zarr mode")
        isel_kw = _parse_isel_arg(args.isel)
        pred, era5, w = load_xarray_pair(
            args.pred_netcdf,
            args.truth_netcdf,
            args.var,
            isel_kw,
        )
    else:
        pred, era5 = load_torch(args.blob)
        lat, _ = lat_lon_mesh(pred.shape[-2], pred.shape[-1])
        w = torch.from_numpy(cosine_latitude_weights(lat).squeeze()).float()

    print(json.dumps(compute_skill_metrics(pred, era5, w)))


if __name__ == "__main__":
    main()

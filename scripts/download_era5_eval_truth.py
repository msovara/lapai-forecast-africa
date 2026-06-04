#!/usr/bin/env python3
"""Download ERA5 ground truth for the Mvua evaluation protocol (CDS -> GRIB -> Zarr).

Aligns with configs/eval.yaml: test period 2023-2025, variables t2m/tp/u10/v10,
Africa bounding box by default (full domain available via --domain global).

Prerequisites:
  - CDS account: https://cds.climate.copernicus.eu/
  - ~/.cdsapirc with url + key (see runbook section 8a)
  - pip install -e '.[cds]'   # cdsapi
  - For --convert: pip install cfgrib + eccodes (on Lengau see docs/LENGAU.md nogrib note)

Examples:
  python scripts/download_era5_eval_truth.py --dry-run
  python scripts/download_era5_eval_truth.py --years 2023
  python scripts/download_era5_eval_truth.py --years 2023,2024,2025
  python scripts/download_era5_eval_truth.py --years 2023 --convert
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluation.run_scorecard import _read_eval_config, _resolve_domain  # noqa: E402

_DEFAULT_DOWNLOAD_CFG = _REPO / "configs" / "era5_eval_download.yaml"
_DEFAULT_EVAL_CFG = _REPO / "configs" / "eval.yaml"

# CDS area order: North, West, South, East (degrees)
_CDS_VAR_MAP = {
    "t2m": "2m_temperature",
    "tp": "total_precipitation",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
}
_GRIB_RENAME = {"2t": "t2m", "tp": "tp", "10u": "u10", "10v": "v10"}


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return _read_eval_config(path)


def _parse_years(raw: str | None, eval_cfg: dict) -> list[int]:
    if raw:
        return [int(y.strip()) for y in raw.split(",") if y.strip()]
    period = eval_cfg.get("test_period") or {}
    start = int(str(period.get("start", "2023-01-01"))[:4])
    end = int(str(period.get("end", "2025-12-31"))[:4])
    return list(range(start, end + 1))


def _cds_area_nws_e(domain_box: tuple | None) -> list[float]:
    """lat/lon ranges -> CDS [N, W, S, E]."""
    if domain_box is None:
        return [90.0, -180.0, -90.0, 180.0]
    (lat0, lat1), (lon0, lon1) = domain_box
    return [float(lat1), float(lon0), float(lat0), float(lon1)]


def build_cds_request(
    year: int,
    variables: list[str],
    area: list[float],
    dl_cfg: dict,
) -> dict:
    var_map = dl_cfg.get("variable_map") or _CDS_VAR_MAP
    cds_vars = [var_map[v] for v in variables if v in var_map]
    if not cds_vars:
        raise SystemExit(f"No CDS mapping for variables {variables}")
    cds = dl_cfg.get("cds") or {}
    return {
        "product_type": cds.get("product_type", "reanalysis"),
        "variable": cds_vars,
        "year": str(year),
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": cds.get("times", ["00:00", "06:00", "12:00", "18:00"]),
        "area": area,
        "format": cds.get("format", "grib"),
    }


def download_year(
    client,
    year: int,
    variables: list[str],
    area: list[float],
    out_path: Path,
    dl_cfg: dict,
    dry_run: bool,
) -> None:
    req = build_cds_request(year, variables, area, dl_cfg)
    dataset = (dl_cfg.get("cds") or {}).get("dataset", "reanalysis-era5-single-levels")
    print(f"# CDS {dataset} year={year} -> {out_path}", file=sys.stderr)
    if dry_run:
        print(req, file=sys.stderr)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    client.retrieve(dataset, req, str(out_path))
    print(f"wrote {out_path}", file=sys.stderr)


def grib_to_zarr(grib_paths: list[Path], out_zarr: Path, rename: dict[str, str] | None = None) -> None:
    """Merge yearly GRIB files into one scorecard-ready Zarr (latitude, longitude, time; t2m,tp,u10,v10)."""
    try:
        import xarray as xr
    except ImportError as exc:
        raise SystemExit("convert needs xarray: pip install -e '.[data]'") from exc

    rename = rename or _GRIB_RENAME
    paths = sorted(p for p in grib_paths if p.is_file())
    if not paths:
        raise SystemExit(f"No GRIB files under {grib_paths!r}")

    parts: list = []
    for p in paths:
        try:
            opened = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""})
        except Exception:
            # Multi-type GRIB hypercubes: open each and merge
            import cfgrib  # type: ignore

            datasets = cfgrib.open_datasets(p, backend_kwargs={"indexpath": ""})
            opened = xr.merge(datasets, compat="override", join="outer")
        parts.append(opened)

    ds = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
    ds = ds.sortby("time")

    # Normalise dim names for evaluation/run_scorecard.py
    ren_dim: dict[str, str] = {}
    if "latitude" not in ds.dims and "lat" in ds.dims:
        ren_dim["lat"] = "latitude"
    if "longitude" not in ds.dims and "lon" in ds.dims:
        ren_dim["lon"] = "longitude"
    if ren_dim:
        ds = ds.rename(ren_dim)

    # Rename variables (cfgrib shortName or long name)
    var_renames: dict[str, str] = {}
    for v in list(ds.data_vars):
        sn = ds[v].attrs.get("GRIB_shortName", v)
        if sn in rename:
            var_renames[v] = rename[sn]
        elif v in rename:
            var_renames[v] = rename[v]
    if var_renames:
        ds = ds.rename(var_renames)

    keep = [v for v in rename.values() if v in ds]
    if not keep:
        raise SystemExit(f"None of {list(rename.values())} found after open; vars={list(ds.data_vars)}")
    ds = ds[keep]

    out_zarr.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(out_zarr, mode="w")
    print(f"wrote {out_zarr.resolve()}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser(description="Download ERA5 eval ground truth (CDS, Africa default)")
    p.add_argument("--eval-config", type=Path, default=_DEFAULT_EVAL_CFG)
    p.add_argument("--download-config", type=Path, default=_DEFAULT_DOWNLOAD_CFG)
    p.add_argument("--domain", default=None, help="africa (default) or global")
    p.add_argument("--years", default=None, help="Comma years, default from eval test_period")
    p.add_argument("--variables", default=None, help="Comma list, default from eval.yaml")
    p.add_argument("--grib-dir", type=Path, default=None, help="Override GRIB output directory")
    p.add_argument("--zarr-out", type=Path, default=None, help="Override merged Zarr path")
    p.add_argument("--dry-run", action="store_true", help="Print CDS requests only")
    p.add_argument("--convert", action="store_true", help="Merge GRIB -> Zarr after download")
    p.add_argument("--convert-only", action="store_true", help="Skip download; convert existing GRIB")
    args = p.parse_args()

    eval_cfg = _read_eval_config(args.eval_config)
    dl_cfg = _load_yaml(args.download_config)

    variables = (
        [x.strip() for x in args.variables.split(",") if x.strip()]
        if args.variables
        else list(eval_cfg.get("variables") or ["t2m", "tp", "u10", "v10"])
    )
    years = _parse_years(args.years, eval_cfg)
    resolved = _resolve_domain(eval_cfg, args.domain)
    domain_name = resolved[0] if resolved else "global"
    area = _cds_area_nws_e((resolved[1], resolved[2]) if resolved else None)

    out_block = dl_cfg.get("output") or {}
    grib_dir = args.grib_dir or (_REPO / out_block.get("grib_dir", "data/raw/lapai/era5/eval/grib"))
    zarr_out = args.zarr_out or (_REPO / out_block.get("zarr_path", "data/processed/lapai/era5_eval_truth_africa.zarr"))

    print(
        f"# era5 eval truth: domain={domain_name} area(N,W,S,E)={area} years={years} vars={variables}",
        file=sys.stderr,
    )

    grib_paths: list[Path] = []
    for year in years:
        out_file = grib_dir / f"era5_eval_{domain_name}_{year}.grib"
        grib_paths.append(out_file)

    if not args.convert_only:
        if args.dry_run:
            for year, out_file in zip(years, grib_paths, strict=True):
                download_year(None, year, variables, area, out_file, dl_cfg, dry_run=True)
        else:
            try:
                import cdsapi
            except ImportError as exc:
                raise SystemExit("CDS download needs cdsapi: pip install -e '.[cds]'") from exc
            client = cdsapi.Client()
            for year, out_file in zip(years, grib_paths, strict=True):
                download_year(client, year, variables, area, out_file, dl_cfg, dry_run=False)

    if args.convert or args.convert_only:
        grib_rename = dl_cfg.get("grib_rename") or _GRIB_RENAME
        grib_to_zarr(grib_paths, zarr_out, rename={str(k): str(v) for k, v in grib_rename.items()})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

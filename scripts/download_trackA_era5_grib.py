#!/usr/bin/env python3
"""Download ERA5 GRIB for Track A Zarr build (fallback when ECMWF O96 subset is unavailable).

Preferred path: scripts/build_trackA_era5_zarr.py --smoke (ECMWF public O96 Zarr subset).

This script pulls 6-hourly single-level fields from CDS into:
  data/raw/lapai/era5/n320/era5_sfc_YYYYMM.grib

Requires: pip install cdsapi, ~/.cdsapirc configured.

Examples:
  python scripts/download_trackA_era5_grib.py --dry-run
  python scripts/download_trackA_era5_grib.py --smoke          # 2018 Q1
  python scripts/download_trackA_era5_grib.py --start 2020-01 --end 2021-12
"""

from __future__ import annotations

import argparse
import calendar
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Track A recipe surface fields (short names; CDS uses long names below).
TRACKA_CDS_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    "surface_pressure",
    "total_precipitation",
]

SMOKE_START = (2018, 1)
SMOKE_END = (2018, 3)
DEFAULT_OUT = _REPO / "data/raw/lapai/era5/n320"


def _month_range(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    y, m = start
    ey, em = end
    out: list[tuple[int, int]] = []
    while (y, m) <= (ey, em):
        out.append((y, m))
        m += 1
        if m > 12:
            y += 1
            m = 1
    return out


def _download_month(client, year: int, month: int, out_dir: Path, *, dry_run: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"era5_sfc_{year}{month:02d}.grib"
    if dest.is_file():
        print(f"  skip (exists): {dest}")
        return dest

    ndays = calendar.monthrange(year, month)[1]
    days = [f"{d:02d}" for d in range(1, ndays + 1)]
    request = {
        "product_type": "reanalysis",
        "variable": TRACKA_CDS_VARIABLES,
        "year": str(year),
        "month": f"{month:02d}",
        "day": days,
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "grid": [0.25, 0.25],
        "format": "grib",
    }
    print(f"  CDS -> {dest}")
    if dry_run:
        return dest

    tmp = dest.with_suffix(".grib.tmp")
    client.retrieve("reanalysis-era5-single-levels", request, str(tmp))
    tmp.replace(dest)
    return dest


def main() -> int:
    p = argparse.ArgumentParser(description="Download ERA5 GRIB for Track A (CDS fallback)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--smoke", action="store_true", help="2018-01 .. 2018-03")
    p.add_argument("--start", default=None, help="Start YYYY-MM")
    p.add_argument("--end", default=None, help="End YYYY-MM")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.smoke:
        start = SMOKE_START
        end = SMOKE_END
    else:
        if not args.start or not args.end:
            print("Pass --smoke or both --start YYYY-MM and --end YYYY-MM", file=sys.stderr)
            return 1
        sy, sm = (int(x) for x in args.start.split("-"))
        ey, em = (int(x) for x in args.end.split("-"))
        start, end = (sy, sm), (ey, em)

    months = _month_range(start, end)
    print(f"# download Track A GRIB: months={len(months)} out_dir={args.out_dir}")

    client = None
    if not args.dry_run:
        import cdsapi

        client = cdsapi.Client()

    for y, m in months:
        print(f"\n## {y}-{m:02d}")
        _download_month(client, y, m, args.out_dir, dry_run=args.dry_run)

    if not args.dry_run:
        print(f"\nOK: GRIB under {args.out_dir}")
        print("Next:")
        print("  python scripts/build_trackA_era5_zarr.py --from-grib --smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

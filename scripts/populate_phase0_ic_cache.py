#!/usr/bin/env python3
"""Download ERA5 CDS initial-condition GRIBs for Phase 0 inits (run on laptop with internet).

Uses the same cache keys as utils/cds_ic.py / aifs-africa. After this completes,
sync the cache directory to Lengau (offline cluster reads cache only).

Examples:
  python scripts/populate_phase0_ic_cache.py --dry-run
  python scripts/populate_phase0_ic_cache.py
  python scripts/populate_phase0_ic_cache.py --cache-dir ~/.cache/aifs-africa/era5
  python scripts/populate_phase0_ic_cache.py --inits 20230101,20230108
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluation.phase0_scorecard import load_phase0_config  # noqa: E402
from utils.cds_ic import AIFS_AFRICA_CACHE_DIR, ensure_era5_ic_grib_cache  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Populate CDS ERA5 IC cache for Phase 0")
    p.add_argument("--config", type=Path, default=_REPO / "configs" / "phase0_baseline.yaml")
    p.add_argument("--inits", default=None, help="Comma YYYYMMDD list (default: config inits)")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=f"GRIB cache root (default: config cds_cache_dir or {AIFS_AFRICA_CACHE_DIR})",
    )
    p.add_argument("--init-time", default=None, help="Init HHMM (default: config init_time)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = load_phase0_config(args.config)
    init_time = args.init_time or cfg.get("init_time", "0000")

    inits = (
        [x.strip() for x in args.inits.split(",") if x.strip()]
        if args.inits
        else list(cfg.get("inits") or [])
    )
    cache_dir = args.cache_dir or Path(
        os.path.expanduser(
            str(cfg.get("cds_cache_dir") or cfg.get("ic", {}).get("cache_dir") or AIFS_AFRICA_CACHE_DIR)
        )
    )

    print(f"# populate IC cache: inits={inits} cache_dir={cache_dir}")
    for init in inits:
        print(f"\n## init {init} {init_time}Z (+ t-6h)")
        if args.dry_run:
            print(f"   would retrieve ERA5 CDS -> {cache_dir}")
            continue
        ensure_era5_ic_grib_cache(init, init_time, cache_dir=cache_dir, allow_download=True)

    if not args.dry_run:
        print(f"\nOK: cache ready under {cache_dir}")
        print("Sync to Lengau (example):")
        print(f"  scp -r {cache_dir} msovara@lengau.chpc.ac.za:~/.cache/aifs-africa/era5/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

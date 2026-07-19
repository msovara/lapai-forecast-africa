#!/usr/bin/env python3
"""Build Track A training Zarr (era5_n96) for anemoi coarsen fine-tune.

Recommended path (laptop + internet): subset ECMWF public O96 ERA5 — no CDS GRIB
regrid needed. After build, rsync the Zarr to Lengau and set anemoi.dataset in
configs/trackA_coarsen.yaml.

Examples:
  pip install "anemoi-datasets>=0.5.22"

  # Smoke (2018 Q1, ~few GB) — unblocks Track A pipeline test
  python scripts/build_trackA_era5_zarr.py --smoke

  # Two-year subset (better fine-tune than smoke; expect several hours over HTTP)
  python scripts/build_trackA_era5_zarr.py --recipe recipes/recipe_era5_o96_from_ecmwf_2020_2021.yaml

  # Monitor a background build:
  #   Get-Content reports/zarr_build_stderr.log -Wait -Tail 5
  #   (data chunks appear only after the load task finishes; init ~1 min, load hours)

  # Note: anemoi-datasets 0.5.x --threads can fail with TypeError; use default (no --threads).

  # Full ECMWF O96 archive (~0.5 TB) — resume-safe
  python scripts/build_trackA_era5_zarr.py --full-copy --target data/processed/lapai/era5_o96_full.zarr

  # Fallback: build from local CDS GRIB under data/raw/lapai/era5/n320/
  python scripts/download_trackA_era5_grib.py --smoke
  python scripts/build_trackA_era5_zarr.py --from-grib --smoke
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
ECMWF_O96_URL = "https://data.ecmwf.int/anemoi-datasets/era5-o96-1979-2023-6h-v8.zarr"

SMOKE_RECIPE = _REPO / "recipes" / "recipe_era5_o96_from_ecmwf_smoke.yaml"
GRIB_SMOKE_RECIPE = _REPO / "recipes" / "recipe_era5_n96_smoke.yaml"
GRIB_FULL_RECIPE = _REPO / "recipes" / "recipe_era5_n96.yaml"


def _anemoi_datasets_exe() -> list[str]:
    exe = shutil.which("anemoi-datasets")
    if exe:
        return [exe]
    return [sys.executable, "-m", "anemoi.datasets"]


def _run(cmd: list[str], *, dry_run: bool) -> int:
    print("exec:", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd)


def _output_from_recipe(recipe: Path) -> Path | None:
    try:
        import yaml
    except ImportError:
        return None
    data = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    out = (data or {}).get("output") or {}
    rel = out.get("path")
    if not rel:
        return None
    p = Path(rel)
    return p if p.is_absolute() else _REPO / p


def _validate_zarr(path: Path) -> None:
    import xarray as xr

    ds = xr.open_zarr(path)
    n_time = ds.sizes.get("time") or ds.sizes.get("date") or "?"
    n_var = len(ds.data_vars)
    print(f"OK: zarr {path} — dims={dict(ds.sizes)} data_vars={n_var} (time-like={n_time})")


def _print_lengau_rsync(zarr_path: Path) -> None:
    rel = zarr_path.relative_to(_REPO) if zarr_path.is_relative_to(_REPO) else zarr_path
    print("\n# Sync to Lengau, then set configs/trackA_coarsen.yaml:")
    print(f"#   anemoi.dataset: {rel.as_posix()}")
    print("#")
    print(f"rsync -avP {zarr_path}/ \\")
    print(f"  msovara@lengau.chpc.ac.za:~/repos/lapai-forecast/{rel.as_posix()}/")
    print("#")
    print("cd ~/repos/lapai-forecast && qsub pbs/trackA.pbs")


def main() -> int:
    p = argparse.ArgumentParser(description="Build Track A era5_n96 training Zarr")
    p.add_argument("--recipe", type=Path, help="anemoi-datasets recipe YAML")
    p.add_argument("--output", type=Path, help="Override output Zarr path")
    p.add_argument("--smoke", action="store_true", help="Use 2018 Q1 ECMWF subset recipe")
    p.add_argument(
        "--from-grib",
        action="store_true",
        help="Use local GRIB recipe (run download_trackA_era5_grib.py first)",
    )
    p.add_argument(
        "--full-copy",
        action="store_true",
        help=f"Copy full ECMWF O96 archive from {ECMWF_O96_URL}",
    )
    p.add_argument(
        "--target",
        type=Path,
        default=_REPO / "data/processed/lapai/era5_o96_full.zarr",
        help="Destination for --full-copy",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume", action="store_true", help="Resume --full-copy")
    p.add_argument("--transfers", type=int, default=4, help="Parallel HTTP transfers for --full-copy")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.full_copy:
        cmd = [
            *_anemoi_datasets_exe(),
            "copy",
            "--url",
            ECMWF_O96_URL,
            "--target",
            str(args.target),
            "--transfers",
            str(args.transfers),
        ]
        if args.resume:
            cmd.append("--resume")
        rc = _run(cmd, dry_run=args.dry_run)
        if rc == 0 and not args.dry_run:
            _validate_zarr(args.target)
            _print_lengau_rsync(args.target)
        return rc

    if args.from_grib:
        recipe = GRIB_SMOKE_RECIPE if args.smoke else GRIB_FULL_RECIPE
    elif args.smoke:
        recipe = SMOKE_RECIPE
    elif args.recipe:
        recipe = args.recipe
    else:
        recipe = SMOKE_RECIPE

    recipe = recipe if recipe.is_absolute() else _REPO / recipe
    if not recipe.is_file():
        print(f"recipe not found: {recipe}", file=sys.stderr)
        return 1

    out = args.output or _output_from_recipe(recipe)
    if out is None:
        print("could not determine output path; pass --output", file=sys.stderr)
        return 1
    out = out if out.is_absolute() else _REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [*_anemoi_datasets_exe(), "create"]
    if args.overwrite:
        cmd.append("--overwrite")
    cmd.extend([str(recipe), str(out)])

    rc = _run(cmd, dry_run=args.dry_run)
    if rc != 0:
        return rc
    if args.dry_run:
        _print_lengau_rsync(out)
        return 0

    _validate_zarr(out)
    _print_lengau_rsync(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

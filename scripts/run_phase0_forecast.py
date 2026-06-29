#!/usr/bin/env python3
"""Run one Phase 0 baseline forecast (C4E n320_gt6 -> eval NetCDF on Africa grid).

GPU required. Writes t2m/tp/u10/v10 on regular lat/lon for evaluation.run_scorecard.

Examples:
  python scripts/run_phase0_forecast.py --init 20230101 --dry-run
  python scripts/run_phase0_forecast.py --init 20230101 --lead-time 240
  python scripts/run_phase0_forecast.py --init 20230101 --output data/processed/phase0/forecasts/20230101_00Z.nc
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.phase0_scorecard import load_phase0_config  # noqa: E402
from utils.eval_forecast_io import snapshot_forecast_state, write_eval_regridded_netcdf  # noqa: E402

# Reuse the open-data inference stack from the notebook-aligned driver.
from utils.aifs_fields import STATIC_FORCING_VARS  # noqa: E402
from utils.cds_ic import AIFS_AFRICA_CACHE_DIR, build_cds_input_state  # noqa: E402
from scripts.run_n320_gt6_opendata_forecast import (  # noqa: E402
    _configure_anemoi_inference_without_triton,
    _configure_eccodes,
    _configure_earthkit_caches,
    _make_open_data_runner,
    _patch_earthkit_regrid_windows_urls,
    _print_state_summary,
    _require_cuda,
    _resolve_checkpoint,
    attach_grid_coords,
    build_input_state,
)


def _parse_init(init: str, init_time: str) -> datetime.datetime:
    return datetime.datetime.strptime(f"{init}{init_time}", "%Y%m%d%H%M")


def main() -> int:
    cfg = load_phase0_config()

    p = argparse.ArgumentParser(description="Phase 0 baseline forecast (eval-ready NetCDF)")
    p.add_argument("--config", type=Path, default=_REPO_ROOT / "configs" / "phase0_baseline.yaml")
    p.add_argument("--init", required=True, help="Init date YYYYMMDD")
    p.add_argument("--init-time", default=cfg.get("init_time", "0000"))
    p.add_argument("--lead-time", type=int, default=int(cfg.get("lead_time_hours", 240)))
    p.add_argument("--teacher-config", default=cfg.get("teacher_config", "configs/teacher_n320_gt6.yaml"))
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument(
        "--ic-source",
        choices=("cds", "opendata"),
        default=cfg.get("ic_source", "cds"),
        help="Initial conditions: CDS cache (offline on Lengau) or ECMWF open-data",
    )
    p.add_argument("--source", default=cfg.get("open_data_source", "ecmwf"))
    p.add_argument(
        "--cds-cache-dir",
        type=Path,
        default=None,
        help=f"CDS GRIB cache (default: config cds_cache_dir or {AIFS_AFRICA_CACHE_DIR})",
    )
    p.add_argument(
        "--cds-offline",
        action="store_true",
        default=os.environ.get("LAPAI_CDS_OFFLINE", "").lower() in ("1", "true", "yes"),
        help="Fail if CDS cache miss (set on Lengau; populate cache on laptop first)",
    )
    p.add_argument("--num-chunks", type=int, default=int(cfg.get("num_chunks", 16)))
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=_REPO_ROOT / cfg.get("cache_dir", "data/cache/earthkit"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: data/processed/phase0/forecasts/YYYYMMDD_00Z.nc",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.config != _REPO_ROOT / "configs" / "phase0_baseline.yaml":
        cfg = load_phase0_config(args.config)

    init_dt = _parse_init(args.init, args.init_time)
    out = args.output or (
        _REPO_ROOT
        / cfg.get("forecast_dir", "data/processed/phase0/forecasts")
        / f"{args.init}_00Z.nc"
    )
    if not str(out).endswith(".nc"):
        out = Path(str(out) + ".nc")

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    _configure_anemoi_inference_without_triton()

    import torch

    print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
    _require_cuda()

    ckpt_args = argparse.Namespace(
        checkpoint=args.checkpoint,
        teacher_config=args.teacher_config,
    )
    ckpt = _resolve_checkpoint(ckpt_args)
    print("Checkpoint:", ckpt)
    print("Init:", init_dt, "Lead:", args.lead_time, "h ->", out)

    if args.dry_run:
        return 0

    _configure_eccodes()
    _configure_earthkit_caches(args.cache_dir.resolve())
    _patch_earthkit_regrid_windows_urls()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["ANEMOI_INFERENCE_NUM_CHUNKS"] = str(args.num_chunks)

    cds_cache = args.cds_cache_dir or Path(
        os.path.expanduser(
            str(cfg.get("cds_cache_dir") or cfg.get("ic", {}).get("cache_dir") or AIFS_AFRICA_CACHE_DIR)
        )
    )
    if args.ic_source == "cds":
        print("IC source: CDS cache ->", cds_cache, "offline=", args.cds_offline)
        input_state = build_cds_input_state(
            init_dt,
            cache_dir=cds_cache,
            allow_download=not args.cds_offline,
        )
    else:
        print("IC source: ECMWF open-data ->", args.source)
        input_state = build_input_state(init_dt, args.source)
    _print_state_summary(input_state, label="input")
    static_forcings = {k: input_state["fields"][k] for k in STATIC_FORCING_VARS}

    runner = _make_open_data_runner(str(ckpt), static_forcings)
    dataset_name = next(iter(runner.tensor_handlers))
    handler = runner.tensor_handlers[dataset_name]
    latitudes = handler.metadata.latitudes
    longitudes = handler.metadata.longitudes
    input_state = attach_grid_coords(input_state, latitudes, longitudes)

    states = []
    for state in runner.run(input_states=input_state, lead_time=args.lead_time):
        state = attach_grid_coords(state, latitudes, longitudes)
        states.append(snapshot_forecast_state(state))
        _print_state_summary(state, label="forecast")

    netcdf_path = write_eval_regridded_netcdf(
        states,
        out.resolve(),
        reference_date=init_dt,
        cache_root=args.cache_dir.resolve(),
        domain=str(cfg.get("domain", "africa")),
    )
    print(f"Phase 0 eval NetCDF: {netcdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

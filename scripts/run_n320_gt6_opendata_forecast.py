#!/usr/bin/env python3
"""Generate a forecast with the C4E n320_gt6 checkpoint (notebook-aligned).

Ports the Mvula / AIFS Single v2 open-data workflow:
  ECMWF open data (0.25 deg lat/lon) -> earthkit-regrid -> N320 -> SimpleRunner.

Prerequisites (GPU required):
  conda activate lapai-anemoi
  pip install ecmwf-opendata earthkit-regrid "earthkit-data<1"
  # anemoi-inference, anemoi-models, torch+cuda already in lapai-anemoi

Notebook pins (Colab/HPCF) differ slightly from public PyPI on Windows:
  anemoi-inference==0.11.0 (PyPI latest; notebook lists 0.11.4)
  torch>=2.5+cu121 is fine for local smoke; notebook uses 2.7 on Colab.

Examples:
  python scripts/run_n320_gt6_opendata_forecast.py --dry-run
  python scripts/run_n320_gt6_opendata_forecast.py --lead-time 12 --output data/forecasts/n320_gt6/latest.nc --plot
  python scripts/run_n320_gt6_opendata_forecast.py --checkpoint models/teacher_n320_gt6/inference.ckpt
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import os
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.n320_forecast_io import (  # noqa: E402
    attach_grid_coords,
    default_plot_path,
    plot_temperature_celsius,
    write_forecast_netcdf,
)
from utils.teacher_yaml import (  # noqa: E402
    resolve_teacher_config_path,
    teacher_checkpoint_resolved,
)

PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "lsm", "z", "slor", "sdor"]
PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
STATIC_FORCING_VARS = ("lsm", "sdor", "slor", "z")


def _configure_eccodes() -> None:
    """Ensure conda-forge ecCodes DLLs are visible to pip eccodes/gribapi on Windows."""
    prefix = Path(sys.prefix)
    lib_bin = prefix / "Library" / "bin"
    if lib_bin.is_dir():
        os.environ["PATH"] = str(lib_bin) + os.pathsep + os.environ.get("PATH", "")
    defs = prefix / "Library" / "share" / "eccodes" / "definitions"
    samples = prefix / "Library" / "share" / "eccodes" / "samples"
    if defs.is_dir():
        os.environ.setdefault("ECCODES_DEFINITION_PATH", str(defs))
    if samples.is_dir():
        os.environ.setdefault("ECCODES_SAMPLES_PATH", str(samples))


def _configure_earthkit_caches(cache_root: Path) -> None:
    """Use repo-local caches; avoid filling ~/.cache when C: is >95% full."""
    data_dir = cache_root / "data"
    regrid_dir = cache_root / "regrid"
    data_dir.mkdir(parents=True, exist_ok=True)
    regrid_dir.mkdir(parents=True, exist_ok=True)

    import earthkit.data as ekd
    from earthkit.regrid.utils import caching as regrid_caching

    ekd.config.set(
        {
            "cache-policy": "user",
            "user-cache-directory": str(data_dir),
            # Default 95% trips on busy Windows laptops; allow more headroom here.
            "maximum-cache-disk-usage": 99,
        }
    )
    regrid_caching.SETTINGS.update(
        {
            "cache-policy": "user",
            "user-cache-directory": str(regrid_dir),
            "maximum-cache-disk-usage": 99,
        }
    )
    print(f"earthkit-data cache: {data_dir}")
    print(f"earthkit-regrid cache: {regrid_dir}")


def _patch_earthkit_regrid_windows_urls() -> None:
    """earthkit-regrid builds download URLs with os.path.join (backslash on Windows)."""
    if os.name != "nt":
        return

    import logging

    import earthkit.regrid.db as regrid_db
    from earthkit.regrid.utils.download import download_and_cache

    log = logging.getLogger("earthkit.regrid.db")

    def matrix_path_fixed(self, name: str) -> str:
        base = self._url.rstrip("/")
        rel = name.replace("\\", "/").lstrip("/")
        url = f"{base}/{rel}"
        try:
            return download_and_cache(
                url,
                owner="url",
                verify=True,
                force=None,
                chunk_size=1024 * 1024,
                http_headers=None,
                update_if_out_of_date=False,
                maximum_retries=5,
                retry_after=10,
            )
        except Exception:
            log.error("Could not download matrix file=%s", url)
            raise

    regrid_db.UrlAccessor.matrix_path = matrix_path_fixed  # type: ignore[method-assign]


def _configure_anemoi_inference_without_triton() -> None:
    """Windows and other hosts without Triton: unpickle checkpoint, run GraphTransformer via pyg."""
    os.environ.setdefault("ANEMOI_INFERENCE_GRAPHTRANSFORMER_ATTENTION_BACKEND", "pyg")
    if importlib.util.find_spec("triton") is not None:
        return

    def _stub_module(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.util.spec_from_loader(name, loader=None)
        return mod

    triton = _stub_module("triton")
    triton_language = _stub_module("triton.language")
    triton_language.constexpr = int  # type: ignore[attr-defined]

    def _jit(fn=None, **_kwargs):
        if fn is None:
            return lambda f: f
        return fn

    triton.jit = _jit  # type: ignore[attr-defined]
    triton.language = triton_language
    sys.modules["triton"] = triton
    sys.modules["triton.language"] = triton_language


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not available. This forecast script requires a GPU "
            "(notebook: Ampere or newer; tested on L4/A100)."
        )
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    print(f"GPU: {name} (compute {major}.{minor})")


def _resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        ck = Path(args.checkpoint).resolve()
    else:
        cfg = resolve_teacher_config_path(_REPO_ROOT, args.teacher_config)
        ck = teacher_checkpoint_resolved(_REPO_ROOT, cfg)
        if ck is None:
            raise SystemExit("Could not resolve checkpoint; pass --checkpoint")
        ck = ck.resolve()
    if not ck.is_file():
        raise SystemExit(f"Missing checkpoint: {ck}")
    return ck


def get_open_data(
    date: datetime.datetime,
    param,
    levelist=None,
    source: str = "ecmwf",
    **kwargs,
):
    import earthkit.data as ekd
    import earthkit.regrid as ekr

    levelist = levelist or []
    fields: dict[str, list] = defaultdict(list)
    for dt in [date - datetime.timedelta(hours=6), date]:
        data = ekd.from_source(
            "ecmwf-open-data",
            date=dt,
            param=param,
            levelist=levelist,
            source=source,
            **kwargs,
        )
        for f in data:  # type: ignore[union-attr]
            arr = f.to_numpy()
            if arr.shape != (721, 1440):
                raise ValueError(f"Unexpected open-data shape {arr.shape} for {f.metadata('param')}")
            values = np.roll(arr, -f.shape[1] // 2, axis=1)
            values = ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": "N320"})
            name = (
                f"{f.metadata('param')}_{f.metadata('levelist')}"
                if levelist
                else f.metadata("param")
            )
            fields[name].append(values)

    out = {}
    for k, v in fields.items():
        stacked = np.stack(v)
        if not levelist and stacked.shape[0] != 2:
            raise ValueError(
                f"Surface field `{k}` has shape {stacked.shape}, expected (2, n320). "
                "Try levtype='sfc' or check for duplicate GRIB messages."
            )
        out[k] = stacked
    return out


def _print_state_summary(state: dict, label: str = "state") -> None:
    fields = state.get("fields", state)
    date = state.get("date")
    prefix = f"{label}: " if label else ""
    if date is not None:
        print(f"{prefix}date={date} fields={len(fields)}")
    else:
        print(f"{prefix}fields={len(fields)}")
    for name in sorted(fields)[:6]:
        arr = fields[name]
        print(f"    {name:<6} shape={arr.shape} min={arr.min():.6g} max={arr.max():.6g}")


def build_input_state(date: datetime.datetime, source: str) -> dict:
    fields: dict = {}
    print("Downloading surface fields from ECMWF open data …")
    fields.update(get_open_data(date, param=PARAM_SFC, levtype="sfc", source=source))
    missing_sfc = set(PARAM_SFC) - set(fields)
    if missing_sfc:
        raise RuntimeError(f"Missing surface parameters: {sorted(missing_sfc)}")

    print("Downloading pressure-level fields …")
    fields.update(get_open_data(date, param=PARAM_PL, levelist=LEVELS, source=source))
    pressure_names = [f"{p}_{lev}" for p in PARAM_PL for lev in LEVELS]
    missing_pl = set(pressure_names) - set(fields)
    if missing_pl:
        raise RuntimeError(f"Missing pressure-level parameters: {sorted(missing_pl)}")

    for level in LEVELS:
        gh_key = f"gh_{level}"
        if gh_key in fields:
            fields[f"z_{level}"] = fields.pop(gh_key) * 9.80665

    input_state = dict(date=date, fields=fields)
    _print_state_summary(input_state, label="input")
    return input_state


def _make_open_data_runner(checkpoint: str, static_forcing_fields: dict):
    """SimpleRunner with static surface forcings carried through multi-step rollout."""
    from anemoi.inference.config.run import RunConfiguration
    from anemoi.inference.forcings import Forcings
    from anemoi.inference.runner import Runner
    from anemoi.inference.runner import RunnerClasses
    from anemoi.inference.runners.simple import SimpleTensorHandler

    class _StateFieldForcings(Forcings):
        def __init__(self, context, variables, mask, field_store):
            super().__init__(context)
            self.variables = variables
            self.mask = mask
            self.field_store = field_store
            self.kinds = dict(retrieved=True, constant_in_time=True)

        def load_forcings_array(self, dates, current_state):
            del current_state
            slices = []
            for name in self.variables:
                arr = np.asarray(self.field_store[name], dtype=np.float32)
                if arr.ndim == 1:
                    per_date = np.stack([arr] * len(dates), axis=0)
                else:
                    per_date = arr[-len(dates) :]
                slices.append(per_date)
            return np.stack(slices, axis=0)

    class _OpenDataTensorHandler(SimpleTensorHandler):
        def create_dynamic_coupled_forcings(self, variables, mask):
            if not variables:
                return []
            return [_StateFieldForcings(self, variables, mask, static_forcing_fields)]

    class _OpenDataRunner(Runner):
        def __init__(self):
            config = RunConfiguration(checkpoint=checkpoint, input="empty", output="none")
            super().__init__(config, classes=RunnerClasses(tensor_handler=_OpenDataTensorHandler))

        def execute(self):
            raise NotImplementedError("Use run()")

        def run(self, *, input_states, **kwargs):
            multi_metadata = self.checkpoint.multi_dataset_metadata
            legacy = False
            if len(multi_metadata) == 1:
                dataset_name = next(iter(multi_metadata))
                if dataset_name not in input_states:
                    legacy = True
                    input_states = {dataset_name: input_states}
            for states in Runner.run(self, input_states=input_states, **kwargs):
                if legacy:
                    yield next(iter(states.values()))
                else:
                    yield states

    return _OpenDataRunner()


def main() -> int:
    p = argparse.ArgumentParser(description="C4E n320_gt6 forecast via ECMWF open data")
    p.add_argument(
        "--teacher-config",
        default="configs/teacher_n320_gt6.yaml",
        help="Teacher YAML (default: configs/teacher_n320_gt6.yaml)",
    )
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--lead-time", type=int, default=12, help="Forecast length in hours")
    p.add_argument(
        "--source",
        default="ecmwf",
        choices=("ecmwf", "azure", "aws", "google"),
        help="ECMWF open-data mirror",
    )
    p.add_argument(
        "--date",
        default=None,
        help="Initial condition ISO datetime (default: latest from open-data API)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan only (CUDA check + checkpoint path); no download or inference",
    )
    p.add_argument(
        "--num-chunks",
        type=int,
        default=16,
        help="ANEMOI_INFERENCE_NUM_CHUNKS (memory tuning)",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "cache" / "earthkit",
        help="Root for earthkit-data + earthkit-regrid caches (default: data/cache/earthkit)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "data" / "forecasts" / "n320_gt6" / "latest.nc",
        help="Write forecast NetCDF to this path (parent dirs created)",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="After inference, write 2t Africa map PNG alongside --output NetCDF",
    )
    p.add_argument(
        "--plot-var",
        default="2t",
        help="Variable for --plot (default: 2t)",
    )
    args = p.parse_args()

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    _configure_anemoi_inference_without_triton()

    import torch

    print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
    _require_cuda()

    ckpt = _resolve_checkpoint(args)
    print("Checkpoint:", ckpt)

    if args.dry_run:
        print(f"Would fetch open-data ICs (source={args.source}) and run lead_time={args.lead_time}h")
        return 0

    try:
        from ecmwf.opendata import Client as OpendataClient
    except ImportError as exc:
        print(
            "Missing open-data stack. Install:\n"
            "  pip install -e '.[forecast]'",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    _configure_eccodes()
    _configure_earthkit_caches(args.cache_dir.resolve())
    _patch_earthkit_regrid_windows_urls()

    if args.date:
        date = datetime.datetime.fromisoformat(args.date.replace("Z", "+00:00"))
    else:
        date = OpendataClient(args.source).latest()
    print("Initial date:", date)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["ANEMOI_INFERENCE_NUM_CHUNKS"] = str(args.num_chunks)

    input_state = build_input_state(date, args.source)
    static_forcings = {k: input_state["fields"][k] for k in STATIC_FORCING_VARS}

    print("Loading model …")
    runner = _make_open_data_runner(str(ckpt), static_forcings)
    dataset_name = next(iter(runner.tensor_handlers))
    handler = runner.tensor_handlers[dataset_name]
    latitudes = np.asarray(handler.metadata.latitudes, dtype=np.float32)
    longitudes = np.asarray(handler.metadata.longitudes, dtype=np.float32)
    input_state = attach_grid_coords(input_state, latitudes, longitudes)

    print(f"Running {args.lead_time}h forecast …")
    states = []
    for state in runner.run(input_states=input_state, lead_time=args.lead_time):
        state = attach_grid_coords(state, latitudes, longitudes)
        states.append(state)
        _print_state_summary(state, label="forecast")

    print(f"OK: {len(states)} state(s) produced")

    netcdf_path = write_forecast_netcdf(
        states,
        args.output.resolve(),
        reference_date=date,
        latitudes=latitudes,
        longitudes=longitudes,
        static_fields={"lsm": static_forcings["lsm"]},
    )
    print(f"NetCDF: {netcdf_path}")

    if args.plot:
        last = states[-1]
        fields = last["fields"]
        if args.plot_var not in fields:
            raise SystemExit(f"--plot-var {args.plot_var} not in forecast fields")
        png_path = default_plot_path(netcdf_path, args.plot_var)
        valid = last["date"]
        step_h = (
            int((valid - date).total_seconds() // 3600)
            if isinstance(valid, datetime.datetime)
            else args.lead_time
        )
        title = f"n320_gt6 2m temperature (+{step_h}h from {date:%Y-%m-%d %HZ})"
        if args.plot_var == "2t":
            plot_temperature_celsius(
                latitudes,
                longitudes,
                fields[args.plot_var],
                title=title,
                output=png_path,
                lsm=static_forcings.get("lsm"),
            )
        else:
            from utils.n320_forecast_io import plot_unstructured_map

            plot_unstructured_map(
                latitudes,
                longitudes,
                fields[args.plot_var],
                title=title,
                output=png_path,
            )
        print(f"Plot: {png_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

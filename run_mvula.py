#!/usr/bin/env python3
"""Mvula end-user entry point (Path A Apptainer + Path B conda/venv).

Subcommands:
  info       Freeze / Case A summary and checkpoint presence
  bench      CPU laptop-style timing (scripts/bench_mvula_laptop_v5.py)
  dashboard  Streamlit status UI (streamlit_status.py)
  help       Dual-path usage

Checkpoint is never bundled: place student_global_stable_v5.ckpt under models/
(or bind-mount that directory into the container).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

FREEZE_TAG = "trackb-v5-c4e"
CKPT_NAME = "student_global_stable_v5.ckpt"
N_PARAMS_M = 2.17  # published v5 student (see MVULA_LAPTOP_BENCHMARK)


def _repo() -> Path:
    env = os.environ.get("LAPAI_REPO", "").strip()
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parent


def _ckpt(repo: Path) -> Path:
    return repo / "models" / CKPT_NAME


def _print_scope_banner(ckpt: Path | None = None, *, present: bool | None = None) -> None:
    """Make supported vs unsupported scope impossible to miss."""
    size_line = "  (place under models/ or bind-mount)"
    if ckpt is not None and present:
        size_mib = ckpt.stat().st_size / (1024 * 1024)
        size_line = f"  {size_mib:.1f} MiB on disk"
    elif ckpt is not None and present is False:
        size_line = "  MISSING - copy or bind-mount models/"

    print(
        f"""Mvula v5
AIFS-derived lightweight CNN for African short-range t2m

Model:
  {CKPT_NAME}
  ~{N_PARAMS_M:.2f}M parameters
{size_line}

Hardware:
  CPU supported
  GPU not required

Validated use:
  Analysis-forced African t2m
  +6 h primary result (freeze {FREEZE_TAG})

Not supported:
  Autonomous free-running forecast
  10-day forecast
  Full 65-channel state reconstruction
"""
    )


def cmd_info(repo: Path) -> int:
    ckpt = _ckpt(repo)
    present = ckpt.is_file()
    report = repo / "reports" / "FINAL_REPORT.md"
    bench = repo / "reports" / "MVULA_LAPTOP_BENCHMARK.md"
    _print_scope_banner(ckpt, present=present)
    print("Paths:")
    print(f"  repo:         {repo}")
    print(f"  checkpoint:   {ckpt}")
    print(f"  final report: {'yes' if report.is_file() else 'missing'}  {report}")
    print(f"  laptop bench: {'yes' if bench.is_file() else 'missing'}  {bench}")
    print()
    print("Next:")
    print("  python run_mvula.py bench       # needs ckpt")
    print("  python run_mvula.py dashboard   # Streamlit status UI")
    print("  See containers/README.md for Apptainer Path A")
    print("  Full narrative: reports/FINAL_REPORT.md")
    if not present:
        return 1
    return 0


def cmd_bench(repo: Path) -> int:
    ckpt = _ckpt(repo)
    if not ckpt.is_file():
        _print_scope_banner(ckpt, present=False)
        print(
            f"missing checkpoint: {ckpt}\n"
            "Place student_global_stable_v5.ckpt under models/ "
            "(or bind-mount models/ into the container).",
            file=sys.stderr,
        )
        return 1
    _print_scope_banner(ckpt, present=True)
    script = repo / "scripts" / "bench_mvula_laptop_v5.py"
    if not script.is_file():
        print(f"missing bench script: {script}", file=sys.stderr)
        return 1
    env = os.environ.copy()
    env["LAPAI_REPO"] = str(repo)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    return subprocess.call([sys.executable, "-u", str(script)], cwd=str(repo), env=env)


def cmd_dashboard(repo: Path, extra: list[str]) -> int:
    app = repo / "streamlit_status.py"
    if not app.is_file():
        print(f"missing dashboard: {app}", file=sys.stderr)
        return 1
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        *extra,
    ]
    return subprocess.call(cmd, cwd=str(repo))


def cmd_help() -> int:
    _print_scope_banner()
    print(
        f"""Usage:
  python run_mvula.py info
  python run_mvula.py bench
  python run_mvula.py dashboard [-- streamlit args...]
  python run_mvula.py help

Which path?
  Laptop user              Path B - Conda/Python
  Researcher               Path B or A
  CHPC / Lengau            Path A - Apptainer
  Reproducibility / paper  Path A - pinned container
  Developer                Path B
  Streamlit demonstration  Either

Path A - Apptainer / Singularity (reproducibility):
  apptainer build mvula-v5.sif containers/Apptainer.def
  apptainer run -B "$PWD/models:/opt/lapai-forecast/models" mvula-v5.sif info
  See containers/README.md

Path B - conda / venv (laptop accessibility):
  conda env create -f environment-mvula-enduser.yml
  conda activate mvula-enduser
  pip install -e ".[dev,data,ort]"
  pip install -r requirements_streamlit.txt
  python run_mvula.py dashboard

Reports: reports/FINAL_REPORT.md
"""
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="run_mvula",
        description="Mvula end-user entry (info / bench / dashboard)",
        add_help=False,
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="help",
        choices=("info", "bench", "dashboard", "help"),
    )
    parser.add_argument("rest", nargs=argparse.REMAINDER, help="Extra args for dashboard")
    args = parser.parse_args(argv)

    if args.command == "help":
        return cmd_help()

    repo = _repo()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    os.chdir(repo)

    if args.command == "info":
        return cmd_info(repo)
    if args.command == "bench":
        return cmd_bench(repo)
    if args.command == "dashboard":
        extra = list(args.rest)
        if extra and extra[0] == "--":
            extra = extra[1:]
        return cmd_dashboard(repo, extra)
    return cmd_help()


if __name__ == "__main__":
    raise SystemExit(main())

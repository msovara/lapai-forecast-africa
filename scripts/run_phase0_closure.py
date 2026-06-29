#!/usr/bin/env python3
"""Close Phase 0: batch-score baseline forecasts and write PHASE0_BASELINE_SCORECARD.json.

Modes:
  --dry-run          Print planned inits/leads; no I/O
  --forecast-only    Generate missing forecasts (GPU on Lengau)
  --score-only       Score existing NetCDFs only (works on laptop with GCS ADC)
  (default)          Forecast missing inits, then score all

Examples:
  python scripts/run_phase0_closure.py --dry-run
  python scripts/run_phase0_closure.py --score-only
  python scripts/run_phase0_closure.py --score-only --forecast-dir ../aifs-africa/output
  python scripts/run_phase0_closure.py --truth local
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.phase0_scorecard import (  # noqa: E402
    build_phase0_scorecard,
    discover_forecasts,
    load_phase0_config,
    write_phase0_scorecard,
)


def _forecast_filename(init: str) -> str:
    return f"{init}_00Z.nc"


def _forecast_path(cfg: dict, init: str) -> Path:
    return _REPO_ROOT / cfg.get("forecast_dir", "data/processed/phase0/forecasts") / _forecast_filename(init)


def _run_forecast(init: str, cfg: dict, config_path: Path) -> None:
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "run_phase0_forecast.py"),
        "--config",
        str(config_path),
        "--init",
        init,
    ]
    print(f"# forecast: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=_REPO_ROOT)


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 0 baseline closure (forecast + scorecard)")
    p.add_argument("--config", type=Path, default=_REPO_ROOT / "configs" / "phase0_baseline.yaml")
    p.add_argument("--forecast-dir", type=Path, default=None, help="Override forecast_dir from config")
    p.add_argument("--out", type=Path, default=None, help="Override scorecard_out from config")
    p.add_argument("--truth", choices=("gcs", "local"), default=None, help="Override truth.source")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--forecast-only", action="store_true")
    p.add_argument("--score-only", action="store_true")
    p.add_argument("--inits", default=None, help="Comma list overriding config inits")
    p.add_argument("--leads", default=None, help="Comma lead hours overriding config (e.g. 24,48)")
    args = p.parse_args()

    cfg = load_phase0_config(args.config)
    inits = [x.strip() for x in args.inits.split(",") if x.strip()] if args.inits else list(cfg.get("inits") or [])
    if args.leads:
        cfg = dict(cfg)
        cfg["leads_hours"] = [int(x.strip()) for x in args.leads.split(",") if x.strip()]
    forecast_dir = args.forecast_dir or (_REPO_ROOT / cfg.get("forecast_dir", "data/processed/phase0/forecasts"))
    out_path = args.out or (_REPO_ROOT / cfg.get("scorecard_out", "reports/PHASE0_BASELINE_SCORECARD.json"))
    truth_source = args.truth or (cfg.get("truth") or {}).get("source", "gcs")
    leads = cfg.get("leads_hours", [])
    variables = cfg.get("variables", [])

    print(f"# Phase 0 closure: inits={inits} leads={leads} vars={variables} truth={truth_source}")
    print(f"# forecast_dir={forecast_dir}")
    print(f"# scorecard_out={out_path}")

    if args.dry_run:
        for init in inits:
            fp = forecast_dir / _forecast_filename(init)
            exists = fp.is_file()
            print(f"  init {init}: forecast {'EXISTS' if exists else 'MISSING'} -> {fp}")
        return 0

    if not args.score_only:
        for init in inits:
            fp = forecast_dir / _forecast_filename(init)
            if fp.is_file():
                print(f"# skip existing forecast {fp}")
                continue
            _run_forecast(init, cfg, args.config)
        if args.forecast_only:
            print("# forecast-only complete")
            return 0

    pattern = str(cfg.get("forecast_pattern", "*_00Z.nc"))
    paths = discover_forecasts(forecast_dir, pattern=pattern)
    if not paths:
        # Fall back to explicit init list paths.
        paths = [_forecast_path(cfg, init) for init in inits]
        paths = [p for p in paths if p.is_file()]

    if not paths:
        print("ERROR: no forecast NetCDF files found to score.", file=sys.stderr)
        return 1

    print(f"# scoring {len(paths)} forecast file(s)")
    scorecard = build_phase0_scorecard(paths, config=cfg, truth_source=truth_source)
    write_phase0_scorecard(scorecard, out_path)
    print(f"wrote {out_path}")

    # Summary table to stdout.
    print("\n| init | lead | var | RMSE | ACC |")
    print("|---|---|---|---|---|")
    for row in scorecard["results"]:
        if row.get("skipped"):
            print(f"| {row.get('init_date', '?')} | +{row.get('lead_hours')}h | - | SKIPPED | {row.get('reason', '')} |")
            continue
        init = row.get("init_date", "?")
        lead = row.get("lead_hours", "?")
        for var, stats in row.get("variables", {}).items():
            if "rmse" in stats:
                print(f"| {init} | +{lead}h | {var} | {stats['rmse']:.4g} | {stats.get('acc', float('nan')):.4g} |")
            else:
                print(f"| {init} | +{lead}h | {var} | ERROR | - |")

    manifest = out_path.with_name("PHASE0_MANIFEST.json")
    manifest.write_text(
        json.dumps(
            {
                "phase": 0,
                "forecasts": [str(p) for p in paths],
                "scorecard": str(out_path),
                "inits": inits,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

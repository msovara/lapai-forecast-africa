#!/usr/bin/env python3
"""Score GraphCast Africa GCS forecasts vs ERA5 (LapAI pathway B).

Examples:
  python scripts/score_graphcast_africa.py --dry-run
  python scripts/score_graphcast_africa.py
  python scripts/score_graphcast_africa.py --inits 20220115 --leads 24,48 --vars t2m,u10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.graphcast_africa_scorecard import (  # noqa: E402
    build_graphcast_scorecard,
    load_graphcast_score_config,
    write_graphcast_scorecard,
)


def main() -> int:
    p = argparse.ArgumentParser(description="GraphCast Africa → ERA5 scorecard")
    p.add_argument(
        "--config",
        type=Path,
        default=_REPO_ROOT / "configs" / "graphcast_africa_baseline.yaml",
    )
    p.add_argument("--out", type=Path, default=None, help="Override scorecard_out")
    p.add_argument("--inits", default=None, help="Comma YYYYMMDD list")
    p.add_argument("--leads", default=None, help="Comma lead hours, e.g. 24,48")
    p.add_argument("--vars", default=None, help="Comma variables, e.g. t2m,u10,v10")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = dict(load_graphcast_score_config(args.config))
    if args.inits:
        cfg["inits"] = [x.strip() for x in args.inits.split(",") if x.strip()]
    if args.leads:
        cfg["leads_hours"] = [int(x.strip()) for x in args.leads.split(",") if x.strip()]
    if args.vars:
        cfg["variables"] = [x.strip() for x in args.vars.split(",") if x.strip()]

    out = args.out or (_REPO_ROOT / cfg.get("scorecard_out", "reports/GRAPHCAST_AFRICA_SCORECARD.json"))

    print(
        f"# GraphCast Africa scorecard: inits={cfg.get('inits')} "
        f"leads={cfg.get('leads_hours')} vars={cfg.get('variables')}"
    )
    print(f"# forecast_bucket={cfg.get('forecast_bucket')}")
    print(f"# scorecard_out={out}")

    card = build_graphcast_scorecard(cfg, dry_run=args.dry_run)
    if args.dry_run:
        print(json_preview(card))
        return 0

    write_graphcast_scorecard(card, Path(out))
    n_ok = sum(
        1
        for row in card["results"]
        for v in row["variables"].values()
        if "rmse" in v and v.get("rmse") == v.get("rmse")  # not NaN
    )
    n_err = sum(1 for row in card["results"] for v in row["variables"].values() if "error" in v)
    print(f"# wrote {out}  ok_metrics={n_ok} errors={n_err}")
    return 0 if n_err == 0 or n_ok > 0 else 1


def json_preview(card: dict) -> str:
    import json

    slim = {k: card[k] for k in card if k != "results"}
    slim["n_planned_rows"] = len(card.get("inits_scored") or []) * len(card.get("leads_hours") or [])
    return json.dumps(slim, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())

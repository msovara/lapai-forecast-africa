#!/usr/bin/env python3
"""Summarize A1 gate t2m+24h and tp nulls."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
g = json.loads((root / "reports/TRACKA_A1_GATE.json").read_text())
print("t2m+24h:")
for c in g["checks"]:
    if c.get("variable") == "t2m" and c.get("lead_hours") == 24:
        print(
            f"  {c['init_date']}: base={c['baseline_rmse']:.3f} "
            f"cand={c['candidate_rmse']:.3f} deg={c['degradation_pct']:.1f}% pass={c['passed']}"
        )
s = json.loads((root / "reports/TRACKA_COARSEN_SCORECARD.json").read_text())
tp_null = all(r["variables"]["tp"].get("rmse") is None for r in s["results"])
print("tp all null in Track A scorecard:", tp_null)
p = json.loads((root / "reports/PHASE0_BASELINE_SCORECARD.json").read_text())
tp_ok = sum(1 for r in p["results"] if r["variables"].get("tp", {}).get("rmse") is not None)
print(f"Phase0 tp rows with rmse: {tp_ok}/{len(p['results'])}")

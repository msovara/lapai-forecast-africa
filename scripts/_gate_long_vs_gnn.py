#!/usr/bin/env python3
from pathlib import Path
from evaluation.trackA_gate import run_coarsen_gate, write_gate_report
from evaluation.run_scorecard import _read_eval_config

CFG = _read_eval_config(Path("configs/trackA_prune.yaml"))
r = run_coarsen_gate(
    Path("reports/TRACKA_PRUNE_LONG_SCORECARD.json"),
    baseline_scorecard=Path("reports/TRACKA_COARSEN_SCORECARD.json"),
    config=CFG,
)
r["step"] = "prune_long_vs_gnn_a1"
write_gate_report(r, Path("reports/TRACKA_A2_GATE_LONG_VS_GNN_A1.json"))
n = sum(1 for c in r["checks"] if c.get("passed"))
print(f"vs GNN A1 passed={r['passed']} ({n}/{len(r['checks'])})")
fails = [c for c in r["checks"] if not c.get("passed")]
for c in fails:
    print(
        f"  FAIL {c.get('init_date')} +{c.get('lead_hours')}h {c.get('variable')}: "
        f"deg={c.get('degradation_pct')}"
    )
print("n_fail", len(fails))

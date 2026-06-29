"""Track A acceptance gate: compare candidate scorecard vs Phase 0 baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.run_scorecard import _read_eval_config

_REPO_ROOT = Path(__file__).resolve().parents[1]

# PLAN §4.1 gate variables (scorecard names where available).
DEFAULT_GATE_VARIABLES = ("t2m", "u10", "v10")
DEFAULT_GATE_LEADS = (24, 48)


def load_scorecard(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _index_results(scorecard: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in scorecard.get("results", []):
        init = row.get("init_date")
        lead = int(row["lead_hours"])
        out[(init, lead)] = row.get("variables") or {}
    return out


def compare_scorecards(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    variables: list[str],
    leads_hours: list[int],
    max_rmse_degradation_pct: float = 5.0,
) -> dict[str, Any]:
    """Return gate report; passed=True iff all checks pass."""
    base_idx = _index_results(baseline)
    cand_idx = _index_results(candidate)
    checks: list[dict[str, Any]] = []

    for (init, lead), base_vars in sorted(base_idx.items()):
        if lead not in leads_hours:
            continue
        cand_vars = cand_idx.get((init, lead))
        if cand_vars is None:
            checks.append(
                {
                    "init_date": init,
                    "lead_hours": lead,
                    "variable": None,
                    "passed": False,
                    "error": "missing candidate row",
                }
            )
            continue
        for var in variables:
            b = base_vars.get(var) or {}
            c = cand_vars.get(var) or {}
            if "error" in b:
                checks.append(
                    {
                        "init_date": init,
                        "lead_hours": lead,
                        "variable": var,
                        "passed": False,
                        "error": f"baseline error: {b['error']}",
                    }
                )
                continue
            if "error" in c:
                checks.append(
                    {
                        "init_date": init,
                        "lead_hours": lead,
                        "variable": var,
                        "passed": False,
                        "error": f"candidate error: {c['error']}",
                    }
                )
                continue
            base_rmse = float(b["rmse"])
            cand_rmse = float(c["rmse"])
            if base_rmse <= 0:
                rel_pct = 0.0 if cand_rmse <= 0 else float("inf")
            else:
                rel_pct = 100.0 * (cand_rmse - base_rmse) / base_rmse
            passed = rel_pct <= max_rmse_degradation_pct
            checks.append(
                {
                    "init_date": init,
                    "lead_hours": lead,
                    "variable": var,
                    "passed": passed,
                    "baseline_rmse": base_rmse,
                    "candidate_rmse": cand_rmse,
                    "degradation_pct": rel_pct,
                    "max_allowed_pct": max_rmse_degradation_pct,
                }
            )

    passed = all(c.get("passed") for c in checks) if checks else False
    return {
        "step": "coarsen",
        "passed": passed,
        "baseline_scorecard": baseline.get("forecasts"),
        "candidate_scorecard": candidate.get("forecasts"),
        "variables": variables,
        "leads_hours": leads_hours,
        "max_rmse_degradation_pct": max_rmse_degradation_pct,
        "checks": checks,
    }


def load_trackA_coarsen_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or (_REPO_ROOT / "configs" / "trackA_coarsen.yaml")
    return _read_eval_config(cfg_path)


def run_coarsen_gate(
    candidate_scorecard: Path,
    *,
    baseline_scorecard: Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config or load_trackA_coarsen_config()
    gate = cfg.get("gate") or {}
    baseline_path = baseline_scorecard or (_REPO_ROOT / gate.get(
        "baseline_scorecard", "reports/PHASE0_BASELINE_SCORECARD.json"
    ))
    if not Path(baseline_path).is_absolute():
        baseline_path = _REPO_ROOT / baseline_path
    baseline = load_scorecard(Path(baseline_path))
    candidate = load_scorecard(Path(candidate_scorecard))
    variables = list(gate.get("variables") or DEFAULT_GATE_VARIABLES)
    leads = [int(x) for x in (gate.get("leads_hours") or DEFAULT_GATE_LEADS)]
    max_pct = float(gate.get("max_rmse_degradation_pct", 5.0))
    return compare_scorecards(
        baseline,
        candidate,
        variables=variables,
        leads_hours=leads,
        max_rmse_degradation_pct=max_pct,
    )


def write_gate_report(report: dict[str, Any], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out_path

#!/usr/bin/env bash
# A2 forecast (opendata IC) → score (GCS) → gate vs A1 coarsen scorecard.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-GNU}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export KMP_DUPLICATE_LIB_OK=TRUE

source /local/Mthetho/miniforge3/etc/profile.d/conda.sh
conda activate /local/Mthetho/envs/lapai-anemoi
cd /local/Mthetho/lapai-forecast

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES} $(date -Is)"
test -e models/teacher_pruned.ckpt

if [[ ! -f data/processed/lapai/n320_latlons.npy ]]; then
  echo "[A2 fc] extract N320 lat/lons"
  python -u scripts/extract_grid_latlons.py \
    --ckpt models/teacher_n320_gt6/inference.ckpt \
    --out data/processed/lapai/n320_latlons.npy
fi

mkdir -p data/processed/trackA_prune/forecasts
INITS=(20230101 20230108 20230115 20230122 20230129)

for init in "${INITS[@]}"; do
  out="data/processed/trackA_prune/forecasts/${init}_00Z.nc"
  if [[ -f "${out}" ]]; then
    echo "[A2 fc] skip existing ${out}"
    continue
  fi
  echo "[A2 fc] forecasting ${init}"
  python -u scripts/run_phase0_forecast.py \
    --config configs/trackA_prune.yaml \
    --init "${init}" \
    --teacher-config configs/teacher_n320_gt6.yaml \
    --checkpoint models/teacher_pruned.ckpt \
    --ic-source cds \
    --cds-cache-dir /home/ubuntu/.cache/aifs-africa/era5 \
    --cds-offline \
    --output "${out}"
done

echo "[A2 fc] scoring → TRACKA_PRUNE_SCORECARD.json"
python -u - <<'PY'
from pathlib import Path
import json
from datetime import datetime, timezone
import numpy as np
import xarray as xr

from evaluation.gcs_era5_truth import (
    GCS_OPTS, align_truth_to_pred, era5_gcs_path, normalize_lon,
    pick_era5_var, subset_africa, valid_time_index, _select_tp_at_valid_time,
)
from evaluation.phase0_scorecard import (
    _resolve_fc_var, _score_point, forecast_step_and_valid_time, write_phase0_scorecard,
)
from evaluation.trackA_gate import run_coarsen_gate, write_gate_report
from evaluation.run_scorecard import _read_eval_config

ROOT = Path(".").resolve()
FORECAST_DIR = ROOT / "data/processed/trackA_prune/forecasts"
SCORECARD = ROOT / "reports/TRACKA_PRUNE_SCORECARD.json"
GATE_OUT = ROOT / "reports/TRACKA_A2_GATE.json"
CFG = _read_eval_config(ROOT / "configs/trackA_prune.yaml")
VARS = list((CFG.get("gate") or {}).get("variables") or ["t2m", "u10", "v10", "tp"])
LEADS = list((CFG.get("gate") or {}).get("leads_hours") or [24, 48])

class TruthCache:
    def __init__(self):
        self._ds = {}
        self._da = {}
    def open_var(self, var, year=2023):
        key = f"{var}_{year}"
        if key in self._da:
            return self._da[key]
        path = era5_gcs_path(var, year)
        print(f"opening {path}", flush=True)
        ds = xr.open_zarr(path, storage_options=GCS_OPTS, consolidated=True)
        name = pick_era5_var(ds)
        da = ds[name]
        for drop in ("number", "surface", "expver"):
            if drop in da.dims and da.sizes[drop] == 1:
                da = da.isel({drop: 0})
        self._ds[key] = ds
        self._da[key] = da
        return da
    def at(self, var, when):
        year = int(str(when)[:4])
        da = self.open_var(var, year)
        if var == "tp" and "step" in da.dims:
            sl = _select_tp_at_valid_time(da, when)
        else:
            ds = self._ds[f"{var}_{year}"]
            time_dim = "time" if "time" in da.dims else da.dims[0]
            sl = da.isel({time_dim: valid_time_index(ds, when)})
            if "step" in sl.dims and sl.sizes.get("step", 1) == 1:
                sl = sl.isel(step=0)
        return subset_africa(normalize_lon(sl.squeeze(drop=True)))
    def close(self):
        for ds in self._ds.values():
            ds.close()

paths = sorted(FORECAST_DIR.glob("*_00Z.nc"))
if not paths:
    raise SystemExit(f"no forecasts in {FORECAST_DIR}")
cache = TruthCache()
results = []
try:
    for path in paths:
        init = path.name.split("_")[0]
        fc = xr.open_dataset(path)
        try:
            for lead in LEADS:
                step, vt = forecast_step_and_valid_time(fc, lead)
                row = {"forecast": str(path), "init_date": init, "lead_hours": lead,
                       "valid_time": str(vt), "variables": {}}
                for var in VARS:
                    fc_name = _resolve_fc_var(fc, var)
                    if fc_name is None:
                        row["variables"][var] = {"error": "missing in forecast"}
                        continue
                    pred = fc[fc_name].isel(time=step).assign_coords(name=var)
                    try:
                        truth = cache.at(var, vt)
                        scores = _score_point(pred, truth)
                        row["variables"][var] = {
                            k: (None if isinstance(v, float) and not np.isfinite(v) else float(v))
                            for k, v in scores.items()
                        }
                    except Exception as exc:
                        row["variables"][var] = {"error": str(exc)}
                    print(f"{init} +{lead}h {var} rmse={row['variables'][var].get('rmse')}", flush=True)
                results.append(row)
        finally:
            fc.close()
finally:
    cache.close()

card = {
    "phase": "trackA_a2_prune_round1",
    "status": "candidate",
    "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "checkpoint": "models/teacher_pruned.ckpt",
    "domain": "africa",
    "variables": VARS,
    "leads_hours": LEADS,
    "truth_source": "gcs",
    "inits_scored": [p.name.split("_")[0] for p in paths],
    "forecasts": [str(p) for p in paths],
    "results": results,
}
write_phase0_scorecard(card, SCORECARD)
print(f"wrote {SCORECARD}", flush=True)

report = run_coarsen_gate(SCORECARD, config=CFG)
report["step"] = "prune"
write_gate_report(report, GATE_OUT)
n_ok = sum(1 for c in report["checks"] if c.get("passed"))
print(f"GATE passed={report['passed']} ({n_ok}/{len(report['checks'])}) -> {GATE_OUT}", flush=True)
for c in report["checks"]:
    if "error" in c:
        print(f"  FAIL {c}", flush=True)
        continue
    print(
        f"  {c['init_date']} +{c['lead_hours']}h {c['variable']}: "
        f"deg={c.get('degradation_pct'):.2f}% pass={c.get('passed')} "
        f"base={c.get('baseline_rmse'):.4g} cand={c.get('candidate_rmse'):.4g}",
        flush=True,
    )
raise SystemExit(0 if report["passed"] else 1)
PY

echo "[A2 fc] done $(date -Is)"

#!/bin/bash
# Day 2 smoke/subset AF t2m eval for v5_t2mRMSE vs v6 accept criteria.
# GPU1 only. Does not overwrite TRACKB_T2M_EXPANDED.*.
set -euo pipefail
cd /local/Mthetho/lapai-forecast
export MKL_INTERFACE_LAYER=GNU
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
source /local/Mthetho/miniforge3/etc/profile.d/conda.sh
conda activate /local/Mthetho/envs/lapai-credit

LOG=/local/Mthetho/logs/trackB_v6_t2m_rmse_eval.log
CKPT=models/student_global_stable_v5_t2mRMSE.ckpt
OUT_JSON=reports/TRACKB_T2M_T2MRMSE_SMOKE.json
OUT_MD=reports/TRACKB_T2M_T2MRMSE_SMOKE.md
FIG_DIR=reports/figures/t2mrmse_smoke

# Subset: ~5 inits per season (00Z), leads 6/12/24
INITS=20230101,20230115,20230129,20230210,20230220,20230401,20230415,20230429,20230510,20230520,20230702,20230716,20230730,20230810,20230820,20231002,20231016,20231030,20231110,20231120

mkdir -p /local/Mthetho/logs reports/figures "$FIG_DIR" /tmp/lapai_no_k1_fc

if [[ ! -f "$CKPT" ]]; then
  echo "[v6-eval] missing $CKPT" | tee "$LOG"
  exit 1
fi

echo "[v6-eval] start $(date -Iseconds)" | tee "$LOG"
echo "[v6-eval] ckpt=$CKPT inits=$INITS -> $OUT_JSON" | tee -a "$LOG"
pip install -e . --no-deps -q
pip install -q gcsfs

python -u evaluation/trackB_t2m_expanded.py \
  --ckpt "$CKPT" \
  --device cuda \
  --ic_source arco \
  --leads 6,12,24 \
  --inits "$INITS" \
  --no_write_nc \
  --teacher_fc_dir /tmp/lapai_no_k1_fc \
  --figures_dir "$FIG_DIR" \
  --out_json "$OUT_JSON" \
  --out_md "$OUT_MD" \
  2>&1 | tee -a "$LOG"

echo "[v6-eval] done $(date -Iseconds)" | tee -a "$LOG"

python - <<'PY' 2>&1 | tee -a "$LOG"
import json
from pathlib import Path
smoke = json.load(open("reports/TRACKB_T2M_T2MRMSE_SMOKE.json"))
base = json.load(open("reports/TRACKB_T2M_EXPANDED.json"))
# pull lead aggregates from student_results
def agg(blob):
    from collections import defaultdict
    import statistics as st
    by = defaultdict(list)
    for r in blob.get("student_results", []):
        t = r["variables"]["t2m"]
        by[r["lead_hours"]].append((float(t["student_rmse_vs_era5"]), float(t["student_acc"]), float(t["student_bias"])))
    out = {}
    for lead, rows in sorted(by.items()):
        r, a, b = zip(*rows)
        out[lead] = {"n": len(r), "rmse": st.mean(r), "acc": st.mean(a), "bias": st.mean(b)}
    return out
s, b = agg(smoke), agg(base)
print("=== ACCEPT GATE (v6) ===")
print("criteria: +6h RMSE<=1.28, +6h ACC>=0.96, +24h RMSE<=3.55")
r6 = s.get(6, {}).get("rmse")
a6 = s.get(6, {}).get("acc")
r24 = s.get(24, {}).get("rmse")
print(f"smoke +6h  RMSE={r6:.3f} ACC={a6:.3f}  (v5 full={b[6]['rmse']:.3f}/{b[6]['acc']:.3f})")
print(f"smoke +12h RMSE={s.get(12,{}).get('rmse', float('nan')):.3f}")
print(f"smoke +24h RMSE={r24:.3f}  (v5 full={b[24]['rmse']:.3f})")
ok = (r6 is not None and r6 <= 1.28 and a6 >= 0.96 and r24 <= 3.55)
print("VERDICT:", "ACCEPT -> promote to v6" if ok else "REJECT -> keep v5 freeze")
Path("reports/TRACKB_V6_T2M_GATE.json").write_text(json.dumps({
    "smoke_ckpt": "models/student_global_stable_v5_t2mRMSE.ckpt",
    "criteria": {"rmse6_max": 1.28, "acc6_min": 0.96, "rmse24_max": 3.55},
    "smoke": {str(k): v for k, v in s.items()},
    "v5_full_ref": {str(k): v for k, v in b.items() if k in (6, 12, 18, 24)},
    "accept": ok,
}, indent=2), encoding="utf-8")
print("wrote reports/TRACKB_V6_T2M_GATE.json")
PY

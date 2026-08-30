# Minimal experiment — reduce African t2m RMSE (from v5 freeze)

**Goal:** Lower analysis-forced **t2m RMSE**, especially at **+6 h** (and diagnose **+12 h**), without free-run, Cout redesign, or tp recovery.  
**Budget:** ~1–2 days on **Cassava GPU1**.  
**Baseline:** `models/student_global_stable_v5.ckpt` + `reports/TRACKB_T2M_EXPANDED.*`  
**Locked v6 accept:** +6 h RMSE **≤ 1.28**, +6 h ACC **≥ 0.96**, +24 h RMSE **≤ 3.55**.  
**Stretch:** +6 h RMSE ≤ 1.25.  
**Control doc:** `TRACKB_V6_T2M_PATH.md` · config `configs/student_global_v6_t2m.yaml` · `scripts/run_trackB_v6_t2m_rmse.sh`

---

## Hard rules (do not break)

- Keep **Cout=3**, analysis-forced only.  
- **Do not** chase tp (leave out-of-scope).  
- **Do not** open Cout≥65 / free-run.  
- **Do not** rebuild the full teacher cache unless diagnosis proves cache is wrong.  
- One continue-train branch only: `student_global_stable_v5_t2mRMSE.ckpt` (never overwrite `v5`).

---

## Day 0 — Diagnose before you train (2–4 h)

### 0.1 Look at the +12 h failure mode

From packaged maps / JSON:

| Question | Why it matters |
|----------|----------------|
| Is +12 h a **global cold bias** or regional? | Bias correction vs retrain |
| Does +12 h fail in **all seasons**? | Diurnal / IC timing vs season |
| Is IC time `init+(L−6)h` correct for L=12? | Protocol bug beats any retrain |

**Actions on Cassava:**

```bash
cd /local/Mthetho/lapai-forecast
# Inspect bias/rmse maps already packaged (or regenerate locally if needed)
ls -la reports/figures/trackb_t2m_v5_{bias,rmse}_L{006,024}h.png
python - <<'PY'
import json
d=json.load(open("reports/TRACKB_T2M_EXPANDED.json"))
# print lead aggregates if present under your schema
print(d.keys())
PY
```

**Decision gate:**

- If +12 h looks like a **systematic bias / timing issue** → fix IC/scoring first (or note as known bug); **do not** burn a long train hoping it disappears.  
- If +6 h is “good but not great” and spatially smooth → proceed to continue-train (§1).

### 0.2 Optional cheap win (no retrain): lead-wise bias correction

Fit mean bias on a **train-like** set of inits; apply to held-out inits; report corrected RMSE separately as `v5+biascorr` (not a new model).

- Useful for SAWS pilot storytelling.  
- Does **not** replace a model improvement claim.

---

## Day 1 — Continue-train for t2m (4–8 h wall)

### 1.1 Setup

```bash
ssh cassava-gpu
export CUDA_VISIBLE_DEVICES=1
export MKL_INTERFACE_LAYER=GNU
cd /local/Mthetho/lapai-forecast
# activate lapai-credit (or your Track B env)
```

Resume from frozen v5; reuse existing teacher cache (full 2020–2021 if present).

### 1.2 Config deltas (only these)

Relative to the v5 recipe (`β/γ` off, full cache, Africa mix):

| Knob | v5 (approx) | This experiment | Intent |
|------|-------------|-----------------|--------|
| `resume` | — | `student_global_stable_v5.ckpt` | Continue, don’t restart |
| `channel_weights` `(tp,msl,2t)` | e.g. high tp / mid 2t | **`[1.0, 0.25, 5.0]`** or **`[0.5, 0.25, 6.0]`** | Push **2t**; de-emphasise tp |
| `africa_mix` | ~0.5 | **0.6–0.7** | More Africa-weighted L_A |
| `beta` / `gamma` | 0 | **keep 0** | Don’t reopen feature/spectral chase |
| epochs / steps | short continue | **small**: e.g. 1–2 epochs or fixed step budget (~few thousand steps) | Avoid overnight overfitting |
| lr | as v5 | **≤ v5 lr** (often 0.5×) | Stable continue-train |
| output | — | `models/student_global_stable_v5_t2mRMSE.ckpt` | Preserve freeze |

**Do not** raise tp weight. Softplus/tp losses stay as-is or ignored in interpretation.

### 1.3 Launch pattern

Use your existing Track B train launcher (same as v5), only swapping resume + yaml overrides, e.g.:

```bash
# Pseudocode — match your run_trackB_stable_v5.sh flags
python -u training/train_student.py \
  --config configs/student_global_v5.yaml \   # or the yaml used for v5
  --resume models/student_global_stable_v5.ckpt \
  --out models/student_global_stable_v5_t2mRMSE.ckpt
# plus env/yaml edits for channel_weights + africa_mix + short schedule
```

Watch: finite loss, `L_A` trending down, no NaNs. Kill if loss explodes.

### 1.4 Smoke before full eval (30–60 min)

3–5 Jan-2023 inits, leads **6 and 12 only**, ARCO IC, Africa t2m RMSE vs v5:

```bash
# Reuse held-out / expanded entrypoint with --ckpt pointing at t2mRMSE
# Compare RMSE@6h and RMSE@12h to TRACKB_T2M_EXPANDED baseline
```

**Kill criteria after smoke:**

| Outcome | Action |
|---------|--------|
| +6 h RMSE **worse** than 1.38 | Stop; revert; try biascorr only |
| +6 h better, +12 h unchanged/worse | Keep branch; still run short expanded eval |
| Loss/NaN | Abort; restore v5 freeze story |

---

## Day 2 — Score and decide (3–6 h)

### 2.1 Minimal expanded re-score (not full 61 if time-tight)

Prefer one of:

**A (preferred if GPU free):** same protocol as expanded, but **subset**:  
- 16–20 inits (4–5 per season)  
- leads `{6,12,24}`  

**B (minimum):** 8–12 inits, leads `{6,24}` only  

Write:

- `reports/TRACKB_T2M_T2MRMSE_SMOKE.json` / `.md`  
- Do **not** overwrite `TRACKB_T2M_EXPANDED.*` until a clear win + full 61-init rerun.

### 2.2 Decision table

| Result | Verdict |
|--------|---------|
| +6 h RMSE ↓ ≥ **5%** relative (≲1.31) and ACC not worse | **Accept** as `v5_t2mRMSE`; optional full 61-init package later |
| +6 h ↓ &lt; 5% | **Weak**; keep v5 freeze; publish biascorr if it helps |
| +6 h ↑ or +12/+24 much worse | **Reject**; leave freeze unchanged |
| Only biascorr helps | Document as **post-process**, not new model |

### 2.3 What to tell SAWS / conference

- If accept: “continue-trained for t2m; +6 h RMSE improved from X → Y under same AF protocol.”  
- If reject: “v5 remains best packaged artefact; longer-lead RMSE needs architecture or IC work, not another night of train.”

---

## Out of scope for this 1–2 day box

- New cache build / CDS  
- tp POD recovery  
- LoRA / ONNX  
- Free-run / Cout=65  
- Full PLAN Week-9 multi-var  

---

## Checklist

- [x] Day 0: +12 h bias/IC diagnosis noted in 5–10 lines  
- [ ] Optional: `v5+biascorr` numbers (separate from model)  
- [x] Continue-train → `student_global_stable_v5_t2mRMSE.ckpt` (v5 untouched)  
- [x] Smoke +6/+12/+24 vs baseline  
- [x] Subset smoke JSON/MD (`TRACKB_T2M_T2MRMSE_SMOKE.*`)  
- [x] Accept / reject / freeze decision recorded (`TRACKB_V6_T2M_GATE.json` → **REJECT**; keep v5)

---

## Expected wall-clock (Cassava GPU1)

| Step | Time |
|------|------|
| Diagnose + maps | 2–4 h |
| Continue-train | 4–8 h |
| Smoke + subset eval | 3–6 h |
| **Total** | **~1–2 days** |

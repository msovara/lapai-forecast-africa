# Track B — Methodology, limitations, and reproduction (v5 freeze)

**Audience:** Code for Earth / NMHS handover.  
**Frozen checkpoint:** `models/student_global_stable_v5.ckpt`  
**Protocol lock:** analysis-forced t2m only — no free-run, no tp chase, no CDS, no architecture change.

---

## 1. What the student is

| Item | Value |
|------|-------|
| Input | 65-channel 1° ERA5 atmospheric state (`t/u/v/q/z` × 13 levels) |
| Output (Cout) | 3 channels: `tp`, `msl`, `2t` (scored as t2m) |
| Backbone | InceptionNeXt CNN student (MILES-CREDIT style) |
| Training | Distillation cache vs K1 teacher (`teacher_k1_cache_t256.zarr`) |
| Case | **A** — Cout=3 intentional; free-run impossible (see `TRACKB_STATE_CLOSURE.md`) |

---

## 2. Analysis-forced protocol (why, and how)

The student cannot roll out multi-step forecasts: the next input needs 65 channels, but the head only emits 3. **Honest multi-lead skill** therefore re-injects analysis ICs:

For lead \(L \in \{6,12,18,24\}\):

1. Build ERA5 IC at time `init + (L − 6) h` (ARCO public zarr; local npy cache under `data/cache/student_ic_arco/`).
2. Run **one** student forward (+6 h).
3. Score prediction at valid time `init + L` against ERA5.

This is **not** free-run. Do not claim multi-day autonomous rollout.

---

## 3. Scoring methodology (student and K1 identical)

On the Africa box (lat [−40, 40] × lon [−20, 55]):

- **RMSE** — cosine-latitude weighted
- **ACC** — anomaly correlation vs spatial climatology of the truth field
- **Bias** — cosine-latitude weighted mean error (student − ERA5)

Truth and ICs: public ARCO ERA5  
`gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3` (anonymous GCS).

K1 comparison uses the **same** RMSE/ACC/bias vs ERA5 wherever teacher forecast NetCDFs exist (`data/processed/trackA_prune_k1_a1bplus/forecasts/`). Inits without K1 still report student vs ERA5; coverage gaps are documented in the JSON.

---

## 4. Primary results location

| Artifact | Path |
|----------|------|
| Machine-readable | `reports/TRACKB_T2M_EXPANDED.json` |
| Human summary | `reports/TRACKB_T2M_EXPANDED.md` |
| Spatial maps | `reports/figures/trackb_t2m_v5_{bias,rmse}_L{006,024}h.png` |
| Narrative roll-up | `reports/TRACKB_REPORT.md` |
| State-closure | `reports/TRACKB_STATE_CLOSURE.md` |

---

## 5. Known failures / limitations

| Topic | Status |
|-------|--------|
| **tp** | Out of scope. Softplus / reweight did not escape all-dry collapse on held-out ICs. Do not gate on precip. |
| **msl** | On-cache gate only; no held-out GCS stem in this protocol. |
| **Free-run** | Impossible with Cout=3 (Case A). |
| **K1 seasonal coverage** | Existing teacher NCs are typically Jan-2023 weekly only unless regenerated. |
| **z500 / t850** | Not in student head — PLAN Week-9 multi-var table incomplete. |

---

## 6. Reproduce (Cassava)

```bash
ssh cassava-gpu
cd /local/Mthetho/lapai-forecast
bash scripts/run_trackB_t2m_expanded_cassava.sh
# or:
export CUDA_VISIBLE_DEVICES=1 MKL_INTERFACE_LAYER=GNU GOOGLE_CLOUD_PROJECT=dummy
source /local/Mthetho/miniforge3/etc/profile.d/conda.sh
conda activate /local/Mthetho/envs/lapai-anemoi   # needs gcsfs for ARCO
python -u evaluation/trackB_t2m_expanded.py \
  --ckpt models/student_global_stable_v5.ckpt --device cuda \
  --ic_source arco --step_days 5 --leads 6,12,18,24
```

**Requirements:** `models/student_global_stable_v5.ckpt`, network access to public ARCO ERA5 (or warm `data/cache/student_ic_arco/`), GPU1 preferred.

**Shorter pilot:** `--months 1,4,7,10 --step_days 5` (~24 inits).

---

## 7. What not to do before handover

- No v6 redesign / Cout expansion / free-run reconstruction
- No CDS API / `.cdsapirc` chase (ARCO replaces it)
- No further tp-only reweight training rounds
- Do not mix free-run claims with analysis-forced tables

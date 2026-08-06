# Email — reply to Chimwemwe GraphCast 1° benchmarking thread

**To:** Chimwemwe Chanda; Shruti Nath; (team)  
**Subject:** Re: GraphCast Africa (1°) benchmarking / evaluation thread

---

Hi Chimwemwe, Shruti, all,

Thanks Chimwemwe — this is a clear and useful baseline, and it matches where we want LapAI evaluation to go while AIFS teacher NaN / rollout trust is still open.

**Decision (per Shruti):** we will proceed **GraphCast-first** as the active benchmarking pipeline until the AIFS issue is resolved. AIFS Phase 0 / Track A work stays paused but retained (configs and training Zarr ready to resume). One project, two teachers, **one Africa scorecard**.

**What we accept from your setup**
- Domain **40°S–40°N, 20°W–70°E** (aligned with LapAI `config/domains.yaml`).
- **GraphCast Small (1°)** when 0.25° is memory-bound.
- ERA5 ICs + truth via CDS **already on the 1° grid** (no extra local regrid).
- Cosine-latitude-weighted RMSE vs ERA5, plus persistence skill  
  \(1 - (\mathrm{RMSE}_\mathrm{forecast}/\mathrm{RMSE}_\mathrm{persistence})^2\).
- Precip verification capped at **48 h** for now given the CDS gap; T2M/MSLP through **240 h**.

Your results support that call: T2M and MSLP stay clearly above persistence; precip is skillful but degrades faster, as expected.

**Small notes (not blockers)**
1. The **T2M skill dip at +30 h (~0.54)** looks like a **persistence diurnal artefact** (forecast RMSE is smooth; persistence RMSE collapses). Please keep publishing **RMSE and skill** side by side so that dip is not read as a model failure.
2. Please **state units** in tables/plots (T2M K; MSLP Pa or hPa; precip m or mm). The precip RMSEs (~1e−3) look like metres, which is fine if consistent.
3. **+6 h skill** blank / persistence RMSE = 0 is expected if persistence is the same analysis — omit skill at that lead or define persistence from t−6 h.
4. Next increment: **2–4 more inits** (not only 2023-01-01 12Z) so we do not overfit a single case.

On our side we have documented this interim policy and a scorecard alignment checklist in the LapAI repo (`reports/GRAPHCAST_FIRST.md`). Happy to ingest your CSV/figures into `reports/GRAPHCAST_SMALL_1DEG_SCORECARD.json` so GraphCast 1°, GraphCast 0.25° (GCS), and later AIFS share one evaluation contract.

If everyone is happy, let’s treat Chimwemwe’s method as the **v0 common GraphCast bench** and iterate from there.

Best regards,  
Mthetho

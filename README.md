# LapAI-Forecast / Mvula

Canonical Git repository: [github.com/msovara/lapai-forecast-africa](https://github.com/msovara/lapai-forecast-africa).

The same project often lives under `tiny-media-analysis/lapai-forecast/` on local machines; `origin` should point at the URL above.

**Mvula** (ECMWF Code for Earth 2026 — African Stream) shrinks advanced AI weather models toward laptop-scale use for African contexts.

> **Close-out (Aug 2026):** Frozen student **`student_global_stable_v5.ckpt`** delivers analysis-forced African **t2m** skill and ~**6×** compression vs the K1 teacher. It does **not** deliver a free-running 10-day forecast. Full claim boundary: [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md).

## Close-out quickstart

```text
Clone → install env → get v5 ckpt → view packaged AF t2m results → launch Streamlit
```

1. **Clone**
   ```bash
   git clone https://github.com/msovara/lapai-forecast-africa.git
   cd lapai-forecast-africa
   git checkout trackb-v5-c4e   # preferred freeze tag when published; else main
   ```

2. **Install** (Track B / student inference)
   ```bash
   # Option A — conda (Cassava / CHPC style)
   # conda env from environment-credit.yml or environment-credit-lengau.yml
   # Option B — editable pip
   pip install -e ".[dev]"
   pip install -r requirements_streamlit.txt   # for the dashboard
   ```

3. **Checkpoint** (not in git)
   - Cassava: `/local/Mthetho/lapai-forecast/models/student_global_stable_v5.ckpt` (~9 MiB)
   - Copy to `models/student_global_stable_v5.ckpt` locally if needed

4. **Results already packaged** (no re-run required to inspect skill)
   - [`reports/TRACKB_T2M_EXPANDED.md`](reports/TRACKB_T2M_EXPANDED.md) — 61 inits × leads 6/12/18/24
   - [`reports/MVULA_LAPTOP_BENCHMARK.md`](reports/MVULA_LAPTOP_BENCHMARK.md) — size / CPU proxy
   - [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md) — full close-out narrative

5. **Re-run AF t2m eval** (optional; needs ARCO/network + GPU recommended)
   ```bash
   # Cassava example
   bash scripts/run_trackB_t2m_expanded_cassava.sh
   ```

6. **Streamlit demo**
   ```bash
   streamlit run streamlit_status.py
   # Windows: run_status_dashboard.bat
   ```

### What Mvula v5 achieves vs limitations

| Achieves | Limitations |
|----------|-------------|
| ~6× smaller than K1; ~2 s/step CPU proxy | Not a measured i7/16 GB laptop yet |
| Strong-ish AF **+6 h t2m** on Africa (ACC≈0.97) | **+24 h** degrades sharply vs K1 |
| Open eval + dashboard | **No** free-run / 10-day student |
| Honest Case A docs | **tp** failed / out-of-scope |

---

## Original proposal targets (context)

The proposal aimed at a mid-range laptop (Intel i7, 16 GB RAM, no GPU), a **10-day** 1° global forecast, ≤15% RMSE vs AIFS globally, LoRA adaptation, and ONNX packaging. Those remain the **programme north star**; the **shipped freeze** is the scoped AF t2m + compression result above — see FINAL_REPORT §2.

A compressed, laptop-deployable AI Numerical Weather Prediction (NWP) model distilled from ECMWF AIFS, with regional adaptation for Africa.

## Targets (proposal)

- **Hardware:** Intel i7 CPU, 16 GB RAM, no discrete GPU.
- **Forecast spec:** 10-day global forecast at 1° resolution.
- **Skill envelope:** ≤ 15 % RMSE degradation vs. AIFS baseline globally; ≤ 20 % on African extreme events.
- **LoRA adaptation cost:** ≤ 6 h on a single consumer GPU.
- **Compute budget:** 1,650 GPU-hours total on CHPC.
- **Schedule:** 12 weeks.

## Pathways (one repo)

| Pathway | Location | Role |
| ------- | -------- | ---- |
| **AIFS teacher** | `teachers/aifs/` | Phase 0 Anemoi `n320_gt6` compression teacher; Track A A1 passed Aug 2026 |
| **GraphCast Africa teacher** | `teachers/graphcast/` | Parallel Africa baseline (GCS); see `reports/GRAPHCAST_FIRST.md` |
| **Student** | `students/` | Track A O96 / laptop target (A1 coarsened checkpoint on Lengau) |

Shared Africa eval box: **lat [−40, 40] × lon [−20, 70]** — see `config/domains.yaml` (aligned with graphcast-africa).

## Approach

Two complementary tracks:

| Framework        | Source     | Role                                                                                          |
| ---------------- | ---------- | --------------------------------------------------------------------------------------------- |
| **Anemoi**       | ECMWF      | Track A — structural compression of AIFS: grid coarsening (N320 → N96, processor O96 → O48) and CRPS-gated attention-head pruning. |
| **MILES-CREDIT** | NSF NCAR   | Track B — 10–15 M-parameter InceptionNeXt CNN student trained with area-weighted MAE + AIFS feature distillation (layers 10 & 14) + spectral distillation. |
| **PyTorch + LoRA** | —        | Phase 3 — rank-`r ∈ {4, 8, 16, 32}` adapters fine-tuned over Africa using ERA5 + ENACTS.       |
| **ONNX Runtime** | —          | Phase 4 — laptop-deployable inference package (Python + ONNX, with Docker / Singularity).     |

```
ECMWF AIFS  ──►  Track A (Anemoi: coarsen + prune)  ──►  pruned teacher
                                                              │
                                                              ▼
ERA5  ─────────────►  Track B (MILES-CREDIT student CNN, distillation losses)
                                                              │
                                                              ▼
                              Phase 3 — LoRA over Africa (ERA5 + ENACTS)
                                                              │
                                                              ▼
                              Phase 4 — ONNX export + `lapai_inference` package
```

## Teacher checkpoint (AIFS Single v1.0)

**Fast path (CHPC):** follow the *Minimal copy-paste path* in [`reports/RUNBOOK_BASELINE_LENGAU.md`](reports/RUNBOOK_BASELINE_LENGAU.md) after loading **`lapai-anemoi`**.

Pinned in [`configs/teacher_aifs.yaml`](configs/teacher_aifs.yaml). Download weights (**`huggingface_hub`** is listed in [`environment-anemoi.yml`](environment-anemoi.yml); purely local pip setups can use `pip install -e ".[hf]"`).

```bash
python scripts/download_teacher_ckpt.py --local-dir models/teacher
bash scripts/lapai_inference_gate.sh   # ckpt smoke + verify --strict + inference --dry-run
# optional: export LAPAI_INFER_TEMPLATE=configs/inference_aifs_netcdf_example.yaml
python scripts/run_aifs_inference.py   # anemoi-inference run (omit internal --dry-run; requires GPU/driver)
# or: qsub pbs/inference_aifs_teacher.pbs
#     qsub -v LAPAI_AIFS_INFER_DRY=1 pbs/inference_aifs_teacher.pbs
```

The Hugging Face file name is **`aifs-single-mse-1.0.ckpt`**; `revision` is pinned next to `checkpoint_filename` in [`configs/teacher_aifs.yaml`](configs/teacher_aifs.yaml). `scripts/download_teacher_ckpt.py` picks up that revision by default (`--revision ""` to float with `main`).

Template for **`anemoi-inference`**: [`configs/inference_aifs_minimal.yaml`](configs/inference_aifs_minimal.yaml). Week‑1 checklist: [`reports/RUNBOOK_BASELINE_LENGAU.md`](reports/RUNBOOK_BASELINE_LENGAU.md). **Troubleshooting (gate / PBS / YAML):** same file, **§7**.

After saving NetCDF forecasts, cosine‑latitude RMSE vs truth (JSON on stdout):

`python -m evaluation.eval_skill --pred-netcdf PATH --truth-netcdf PATH --var VAR [--isel time=0,step=…]`

Torch dict mode (keys `pred` / `era5`): `python -m evaluation.eval_skill --blob MODEL.pt`.

Install **`pip install -e ".[data]"`** when using NetCDF/Zarr.

## Repository status

The long-form technical plan (layouts, compute budget, risks, milestones) is in [`PLAN.md`](PLAN.md).

This tree includes a **working scaffold** aligned with that plan: student CNN (`lapai_inference`), distillation losses (`utils/losses_distillation.py`), cache schema (`lapai_inference/cache_schema.py`), Track A stub (`training/train_trackA.py`), training / LoRA / sensitivity drivers, evaluation hooks, ONNX export, `infer.py`, PBS templates, conda env YAMLs, and `dvc.yaml` stub. ECMWF Anemoi / challenge scorecards and real checkpoints still need to be wired per `PLAN.md` Week 1 gates.

Install locally:

```bash
cd lapai-forecast
pip install -e ".[dev,ort]"
# optional lightweight overlap: pip install -r requirements.txt  (UTF-8; CHPC stacks use conda YAMLs)
pytest
```

## Workflow (local vertical slice)

1. Demo teacher cache (random fields, schema-compliant shapes):

   `python training/build_demo_zarr_cache.py --out data/processed/demo_teacher_cache.zarr`

2. Train from that cache:

   `python training/train_student.py --cache data/processed/demo_teacher_cache.zarr --epochs 5 --batch_size 4`

3. Synthetic smoke (no Zarr):

   `python training/train_student.py --epochs 3 --steps_per_epoch 8`

Swap the demo store for real Anemoi/AIFS exports aligned with `lapai_inference/cache_schema.py` (`state_in`, `era5_target`, `teacher_pred`, `teacher_feat_L10`, `teacher_feat_L14`).

## Track B — reproduce analysis-forced t2m evaluation (v5 freeze)

**Close-out:** [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md) · methodology [`reports/TRACKB_METHODOLOGY_HANDOVER.md`](reports/TRACKB_METHODOLOGY_HANDOVER.md) · Case A [`reports/TRACKB_STATE_CLOSURE.md`](reports/TRACKB_STATE_CLOSURE.md).

Frozen student: `models/student_global_stable_v5.ckpt` (Cout=`tp/msl/2t`; **Case A** — no free-run). Primary gate variable: **t2m**. tp is out-of-scope.

Packaged results (preferred for reviewers): [`reports/TRACKB_T2M_EXPANDED.md`](reports/TRACKB_T2M_EXPANDED.md).

```bash
# Cassava (GPU1 + public ARCO ERA5 ICs; no CDS)
cd /local/Mthetho/lapai-forecast
bash scripts/run_trackB_t2m_expanded_cassava.sh
# → reports/TRACKB_T2M_EXPANDED.json / .md
# → reports/figures/trackb_t2m_v5_{bias,rmse}_L{006,024}h.png
```

Protocol: for each lead \(L\in\{6,12,18,24\}\), IC at `init+(L−6)h` → one +6h student step (analysis-forced). Production campaign: **61** inits across four seasons. Release tag: **`trackb-v5-c4e`**.

CPU size/speed proxy:

```bash
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_INTERFACE_LAYER=GNU
python -u scripts/bench_mvula_laptop_v5.py
```

## CHPC Lengau environments

**Operations guide:** see [`docs/LENGAU.md`](docs/LENGAU.md) — PBS **Track A** defaults to isolated **`lapai-anemoi`** ([`environment-anemoi.yml`](environment-anemoi.yml): PyTorch + CUDA + Anemoi packages). **Track B / LoRA** uses **`lapai-credit`** ([`environment-credit-lengau.yml`](environment-credit-lengau.yml)). Override with `LAPAI_CONDA_TRACK_A` / `B` for CHPC shared stacks.

On **Lengau**, clone this repo (or use `tiny-media-analysis/lapai-forecast`), then from the repo root:

```bash
bash scripts/setup_lengau_envs.sh
```

This loads **`module load chpc/python/anaconda/3-2024.10.1`** (adjust if `module avail` shows a newer Anaconda), creates **`lapai-anemoi`** from [`environment-anemoi.yml`](environment-anemoi.yml), and **`lapai-credit`** from [`environment-credit-lengau.yml`](environment-credit-lengau.yml) (PyTorch + **CUDA 12.1** via conda `nvidia`). Dependencies come from the YAML so PyTorch is not overwritten by pip; the script ends with **`python -m pip install -e . --no-deps`** to register the `lapai_inference` packages.

Optional: store envs on lustre if `$HOME` quota is tight:

```bash
export CONDA_ENVS_PATH=/mnt/lustre/users/$USER/conda-envs
mkdir -p "$CONDA_ENVS_PATH"
bash scripts/setup_lengau_envs.sh
```

GPU batch jobs (matches patterns used elsewhere on CHPC):

```bash
module load chpc/python/anaconda/3-2024.10.1
module load chpc/cuda/12.0/12.0
source /home/apps/chpc/bio/anaconda3-2024.10.1/etc/profile.d/conda.sh
conda activate lapai-credit
```

If conda solves are slow on the login node, submit [`pbs/setup_env_lengau.pbs`](pbs/setup_env_lengau.pbs) from the repo root (edit `cd` line if needed).

## Layout (implemented scaffold)

```
lapai-forecast/
├── PLAN.md                  Technical plan
├── README.md                This file
├── pyproject.toml           Package metadata (`lapai_inference`, utils, evaluation, training, inference)
├── requirements.txt
├── environment-anemoi.yml   Track A conda sketch
├── environment-credit.yml   Track B / ONNX conda sketch
├── infer.py                 Laptop inference entry (delegates to CLI)
├── dvc.yaml                 DVC stub stage (extend after `dvc init`)
├── configs/                 YAML knobs for student, LoRA, eval, Track A
├── recipes/                 Placeholders for Anemoi ERA5 recipes
├── lapai_inference/         Model, cache schema, dataset, preprocess/postprocess, CLI
├── utils/                   Losses, grid coarsen, LoRA, ONNX export
├── evaluation/              Baseline gate, skill + Africa extremes helpers
├── training/                train_student, build_demo_zarr_cache, train_trackA, train_lora, run_sensitivity
├── inference/               PyTorch rollout + ONNX Runtime benchmark
├── diagnostics/plot/        Callback config placeholder
├── containers/              Dockerfile + Singularity sketch
├── pbs/                     CHPC job scripts
├── tests/                   Unit smoke tests
├── data/, models/, logs/    Gitignored artefacts
└── reports/
```

## License

To be selected before the first code commit. Default plan: Apache-2.0 for code, CC-BY-4.0 for documentation, with model weights under a permissive open-weights licence compatible with NMHS redistribution.

## Acknowledgements

- ECMWF for the AIFS public checkpoints and the Anemoi framework.
- NSF NCAR MILES for the CREDIT framework.
- The Centre for High-Performance Computing (CHPC), South Africa, for compute resources.
- Code for Earth — African Stream.

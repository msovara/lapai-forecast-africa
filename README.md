# LapAI-Forecast / Mvula

<p align="center">
  <img src="docs/branding/mvula_banner.jpg" alt="Mvula — Compressing ECMWF’s AIFS for Edge Deployment" width="720">
</p>

<p align="center">
  <a href="https://share.streamlit.io/deploy?repository=msovara/lapai-forecast-africa&branch=main&mainModule=streamlit_status.py"><img src="https://img.shields.io/badge/Live_app-Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Live app Streamlit"></a>
  &nbsp;
  <a href="https://huggingface.co/spaces/C4E-Mvula/mvula-v5-demo"><img src="https://img.shields.io/badge/Demo-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black" alt="Hugging Face Space"></a>
</p>

**Mvula** (ECMWF Code for Earth 2026 — African Stream) compresses ECMWF AIFS toward laptop-scale African weather experimentation.

Canonical repo: [github.com/msovara/lapai-forecast-africa](https://github.com/msovara/lapai-forecast-africa) · freeze tag [`trackb-v5-c4e`](https://github.com/msovara/lapai-forecast-africa/releases/tag/trackb-v5-c4e)

**Start here:** [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md) · [`reports/MVULA_CODE4EARTH_STATUS_MATRIX.md`](reports/MVULA_CODE4EARTH_STATUS_MATRIX.md)

## What shipped (v5)

| | |
|--|--|
| **Model** | ~8.8 MiB student CNN (`models/student_global_stable_v5.ckpt`, tracked in git) |
| **Laptop** | ~2.5 s per +6 h step on Intel i7-11800H CPU (~1.2 GiB RSS; GPU not required) |
| **Skill** | Analysis-forced **one-step** African **t2m** (61 × 00Z inits, 2023). Headline = **00Z IC** → native +6 h step |
| **Not claimed** | Free-run / autoregressive / 10-day forecast; Cout=65 full-state student; **tp** skill; smartphone-ready |

| Achieves | Limitations |
|----------|-------------|
| ~6× smaller than K1; laptop CPU inference | IC fetch/build not in that timing |
| Strong AF **00Z IC** one-step t2m (RMSE≈1.38 °C; ACC≈0.97; ~+42% vs persistence) | **06Z IC** one-step cold-bias pathology (all 61 inits); 12Z/18Z partial recovery |
| Open eval package + Streamlit + Hugging Face model/demo | **No** autonomous multi-day / AR rollout (Case A Cout=3) |
| Honest Case A docs | **tp** failed / out of scope |

## Run locally (no HPC/GPU required)

| User | Path |
|------|------|
| Laptop / demo | **B** — Conda/Python |
| CHPC / paper reproducibility | **A** — Apptainer ([`containers/README.md`](containers/README.md)) |

### 1. Clone

```bash
git clone https://github.com/msovara/lapai-forecast-africa.git
cd lapai-forecast-africa
git checkout trackb-v5-c4e   # optional freeze tag
```

Checkpoint ships with the clone (`models/student_global_stable_v5.ckpt`). Cassava mirror if needed: `/local/Mthetho/lapai-forecast/models/…`.

### 2. Path B — conda (recommended on a laptop)

```bash
conda env create -f environment-mvula-enduser.yml
conda activate mvula-enduser
pip install -e ".[dev,data,ort]"
pip install -r requirements_streamlit.txt
python run_mvula.py info
python run_mvula.py bench
python run_mvula.py dashboard
```

### 3. Path A — Apptainer

```bash
apptainer build mvula-v5.sif containers/Apptainer.def
apptainer run -B "$PWD/models:/opt/lapai-forecast/models" mvula-v5.sif info
apptainer run -B "$PWD/models:/opt/lapai-forecast/models" \
  -B "$PWD/reports:/opt/lapai-forecast/reports" mvula-v5.sif bench
```

Weights are **bind-mounted**, not baked into the SIF. Docker: `containers/Dockerfile`.

### 4. Packaged results (no re-run needed)

- [`reports/DRAFT_PAPER_MVULA_V5.md`](reports/DRAFT_PAPER_MVULA_V5.md) / [`reports/DRAFT_PAPER_MVULA_V5.pdf`](reports/DRAFT_PAPER_MVULA_V5.pdf) — draft paper (narrative + figures)
- [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md) — close-out narrative
- [`reports/TRACKB_T2M_EXPANDED.md`](reports/TRACKB_T2M_EXPANDED.md) — AF t2m scorecard
- [`reports/MVULA_XAI_ATTRIBUTION.md`](reports/MVULA_XAI_ATTRIBUTION.md) — Fig.11 channel/spatial saliency
- [`reports/MVULA_LAPTOP_BENCHMARK.md`](reports/MVULA_LAPTOP_BENCHMARK.md) — size / CPU timing
- Headline figures: [Fig.2 primary 00Z](reports/figures/mvula_fig02_primary_00z_onestep.png) · [Fig.2 IC-hour backup](reports/figures/mvula_fig02_t2m_lead_curves.png) · [Fig.1 pipeline](reports/figures/mvula_fig01_pipeline.png) · [Fig.10 spatial](reports/figures/mvula_fig10_t2m_spatial_6h_24h_quad.png) · [Fig.11 XAI](reports/figures/mvula_fig11_t2m_xai_attribution.png)
- Spatial maps: [00Z RMSE](reports/figures/trackb_t2m_v5_rmse_L006h.png) · [00Z bias](reports/figures/trackb_t2m_v5_bias_L006h.png) · [18Z RMSE](reports/figures/trackb_t2m_v5_rmse_L024h.png) · [18Z bias](reports/figures/trackb_t2m_v5_bias_L024h.png)

Optional re-eval (ARCO + GPU recommended): `bash scripts/run_trackB_t2m_expanded_cassava.sh`

## Hugging Face

- Org: [C4E-Mvula](https://huggingface.co/C4E-Mvula)
- **Model:** [C4E-Mvula/mvula-v5-student](https://huggingface.co/C4E-Mvula/mvula-v5-student)
- **Demo Space:** [C4E-Mvula/mvula-v5-demo](https://huggingface.co/spaces/C4E-Mvula/mvula-v5-demo) · direct page: https://c4e-mvula-mvula-v5-demo.static.hf.space/
- Packaging sources: [`huggingface/`](huggingface/) · upload notes: [`huggingface/UPLOAD.md`](huggingface/UPLOAD.md)

## Approach (brief)

| Track | Role |
|-------|------|
| **A — Anemoi** | Structural compression of AIFS (grid / attention) |
| **B — CREDIT CNN** | Distilled student for African t2m (shipped freeze) |

Programme plan: [`PLAN.md`](PLAN.md). CHPC Lengau ops: [`docs/LENGAU.md`](docs/LENGAU.md). Teacher download / inference runbook: [`reports/RUNBOOK_BASELINE_LENGAU.md`](reports/RUNBOOK_BASELINE_LENGAU.md).

```
AIFS  →  Track A (coarsen / prune)  →  teacher
ERA5  →  Track B (CREDIT student + distillation)  →  Mvula v5
                                              →  Path A/B packaging + ONNX path
```

## Layout

```
configs/  containers/  evaluation/  lapai_inference/  models/  reports/
run_mvula.py  scripts/  training/  utils/  pbs/  tests/
```

GPU training envs: `environment-credit.yml` / `environment-credit-lengau.yml` (CHPC). End-user CPU env: `environment-mvula-enduser.yml`.

## Team & contact

| | |
|--|--|
| **Team (Mvula)** | Chimwemwe Chanda, Mthetho Sovara, Samuel Mathekga, Gabriel Elim, Fima Sichone |
| **Mentors** | Shruti Nath, Rendani Mbuvha, Mario Santa Cruz López |
| **Programme** | [ECMWF Code for Earth 2026 — African Stream](https://codeforearth.ecmwf.int/stream/africa-stream/) |
| **Project page** | [codeforearth.ecmwf.int/project/mvula](https://codeforearth.ecmwf.int/project/mvula/) |
| **Issues / questions** | [GitHub Issues](https://github.com/msovara/lapai-forecast-africa/issues) on this repository |
| **Contributing** | [`CONTRIBUTING.md`](CONTRIBUTING.md) — research demo norms and claim boundary |

## Citation

Cite the public snapshot with [`CITATION.cff`](CITATION.cff). The citable GitHub Release is [`mvula-v5.0.0`](https://github.com/msovara/lapai-forecast-africa/releases/tag/mvula-v5.0.0). The earlier tag [`trackb-v5-c4e`](https://github.com/msovara/lapai-forecast-africa/tree/trackb-v5-c4e) remains the pre-publication freeze; it is not the archive of record.

Chanda, C., Sovara, M., Mathekga, S., Elim, G., & Sichone, F. (2026). *Mvula v5.0.0: compressing ECMWF AIFS for laptop-scale African weather experimentation* [Software]. https://github.com/msovara/lapai-forecast-africa/releases/tag/mvula-v5.0.0

## License

Source code in this repository is licensed under the [Apache License 2.0](LICENSE).

- **Documentation** (reports, guides, figures under `reports/` and `docs/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Student checkpoint** (`models/student_global_stable_v5.ckpt`): provided for research and evaluation; redistribution for NMHS / research use is intended to be permissive. Third-party **teacher** weights (e.g. AIFS) remain under their original licences and terms.
- See also [`NOTICE`](NOTICE) for attribution.

## AI-assisted development

Portions of the software development and documentation process were assisted by AI coding tools (including [Cursor](https://cursor.com/)). All generated or modified code remains subject to human review, testing and validation by the project team. AI assistance does not constitute scientific validation. Reported results are based on the documented experimental protocol and evaluation.

## Acknowledgements

- **Cassava AI Factory** ([Cassava Technologies](https://www.cassava.ai/)) — GPU access for student training, AF evaluation, and related workloads.
- **Centre for High Performance Computing (CHPC)**, South Africa (Lengau) — HPC compute and environment support.
- **[AfriClimate AI](https://africlimate.ai/)** — community context for open, Africa-focused climate–AI collaboration.
- **ECMWF** — AIFS checkpoints and Anemoi.
- **NSF NCAR MILES** — CREDIT framework (Track B).
- **ECMWF Code for Earth** — African Stream mentoring and programme support.

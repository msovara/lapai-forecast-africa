# LapAI-Forecast

Canonical Git repository: [github.com/msovara/lapai-forecast-africa](https://github.com/msovara/lapai-forecast-africa).

The same project often lives under `tiny-media-analysis/lapai-forecast/` on local machines; `origin` should point at the URL above.

A compressed, laptop-deployable AI Numerical Weather Prediction (NWP) model distilled from ECMWF AIFS, with regional adaptation for Africa.

> **Code for Earth — African Stream proposal.** The goal is to bridge the gap in operational AI weather forecasting across Africa by producing a 10-day, 1° global forecast model that runs on a mid-range consumer laptop (Intel i7, 16 GB RAM, no GPU), and to enable low-cost regional fine-tuning by African National Meteorological and Hydrological Services (NMHS).

## Targets

- **Hardware:** Intel i7 CPU, 16 GB RAM, no discrete GPU.
- **Forecast spec:** 10-day global forecast at 1° resolution.
- **Skill envelope:** ≤ 15 % RMSE degradation vs. AIFS baseline globally; ≤ 20 % on African extreme events.
- **LoRA adaptation cost:** ≤ 6 h on a single consumer GPU.
- **Compute budget:** 1,650 GPU-hours total on CHPC.
- **Schedule:** 12 weeks.

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

## Repository status

The long-form technical plan (layouts, compute budget, risks, milestones) is in [`PLAN.md`](PLAN.md).

This tree includes a **working scaffold** aligned with that plan: student CNN (`lapai_inference`), distillation losses (`utils/losses_distillation.py`), cache schema (`lapai_inference/cache_schema.py`), Track A stub (`training/train_trackA.py`), training / LoRA / sensitivity drivers, evaluation hooks, ONNX export, `infer.py`, PBS templates, conda env YAMLs, and `dvc.yaml` stub. ECMWF Anemoi / challenge scorecards and real checkpoints still need to be wired per `PLAN.md` Week 1 gates.

Install locally:

```bash
cd lapai-forecast
pip install -e ".[dev,ort]"
pytest
```

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
├── lapai_inference/         Model, cache schema, preprocess/postprocess, CLI
├── utils/                   Losses, grid coarsen, LoRA, ONNX export
├── evaluation/              Baseline gate, skill + Africa extremes helpers
├── training/                train_student, train_trackA, train_lora, run_sensitivity
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

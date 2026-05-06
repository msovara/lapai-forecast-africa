# LapAI-Forecast

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

This repository is currently in the **planning phase**. The full technical implementation plan — directory layout, configs, hyperparameters, distillation losses, CHPC compute plan, risk register, week-by-week milestones, and acceptance criteria — lives in [`PLAN.md`](PLAN.md).

No training code has been written yet; that begins in Week 1 once the six open decisions in `PLAN.md §10` are closed.

## Planned layout

```
lapai-forecast/
├── PLAN.md                       Full 12-week technical plan
├── README.md                     this file
├── pyproject.toml                installable `lapai_inference` package
├── requirements.txt              pinned env (Anemoi + MILES-CREDIT + LoRA + ONNX)
├── environment-anemoi.yml        conda env for Track A
├── environment-credit.yml        conda env for Tracks B / 3 / 4
├── recipes/                      Anemoi data recipes (ERA5 N320 + N96)
├── configs/                      Hydra configs per phase
├── diagnostics/plot/             plotting callbacks
├── training/                     Track A / Track B / LoRA drivers
├── inference/                    global rollout + laptop ONNX driver
├── evaluation/                   global skill + Africa extremes
├── utils/                        grid coarsen, distillation losses, LoRA, ONNX export
├── lapai_inference/              installable Python package (Phase 4)
├── containers/                   Dockerfile + Singularity.def
├── pbs/                          CHPC PBS templates
├── data/, models/, logs/         (gitignored) artefacts
└── reports/                      per-phase reports
```

## License

To be selected before the first code commit. Default plan: Apache-2.0 for code, CC-BY-4.0 for documentation, with model weights under a permissive open-weights licence compatible with NMHS redistribution.

## Acknowledgements

- ECMWF for the AIFS public checkpoints and the Anemoi framework.
- NSF NCAR MILES for the CREDIT framework.
- The Centre for High-Performance Computing (CHPC), South Africa, for compute resources.
- Code for Earth — African Stream.

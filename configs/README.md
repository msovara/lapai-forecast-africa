# LapAI-Forecast config index

YAML files here drive training, pruning, evaluation, and the pinned teacher identity.

| File | Role |
|------|------|
| `teacher_aifs.yaml` | Hugging Face repo ID, checkpoint name, optional `revision`, local relative path (`models/teacher/…`), distillation hook placeholders (`layer_map`). |
| `trackA_coarsen.yaml` | Track A coarse-grid parameters (e.g. N320 → processor mesh hints for Anemoi / offline tools). |
| `trackA_prune.yaml` | CRPS gate and iterative head-prune knobs. |
| `student_global.yaml` | LapAIStudent CNN geometry and loss phase schedule pointers. |
| `student_distill.yaml` | Distillation loss λ weights etc. — align with PLAN.md. |
| `lora_africa.yaml` | LoRA ranks and Africa crop defaults. |
| `eval.yaml` | Lead hours and verification variable list for scorecards. |
| `inference_aifs_minimal.yaml` | Starter **`anemoi-inference run`** YAML (Quickstart-aligned). Use [`scripts/run_aifs_inference.py`](../scripts/run_aifs_inference.py) to inject absolute `checkpoint:` paths from **`teacher_aifs.yaml`**. |
| `inference_aifs_netcdf_example.yaml` | Example **`lead_time`** + commented **`netcdf`** output skeleton for baseline artefacts under `data/processed/lapai/` (adapt to your anemoi-inference version). |

**Anemoi training** expects Hydra YAML trees (often under `--config-path` bundles from ECMWF). This repo supplies **LapAI-specific** slabs only; glue them into `anemoi-training train` recipes on Lengau or consult the [AIFS Single HF model card](https://huggingface.co/ecmwf/aifs-single-1.0) for pinned package versions (`anemoi-training`, `anemoi-models`, `anemoi-graphs`).

**Teacher inference CLI** docs: [anemoi-inference quickstart](https://anemoi-inference.readthedocs.io/en/latest/usage/quickstart.html).

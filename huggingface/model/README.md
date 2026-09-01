---
license: apache-2.0
library_name: pytorch
tags:
  - weather
  - climate
  - africa
  - aifs
  - distillation
  - code-for-earth
  - cpu
pipeline_tag: other
---

# Mvula v5 — AIFS-distilled student (African short-range t2m)

**Programme:** ECMWF [Code for Earth 2026](https://codeforearth.ecmwf.int/) — African Stream (Challenge 40)  
**Code:** [github.com/msovara/lapai-forecast-africa](https://github.com/msovara/lapai-forecast-africa)  
**Freeze tag:** `trackb-v5-c4e`  
**Checkpoint file:** `student_global_stable_v5.ckpt` (~8.8 MiB, ~2.17 M parameters)

## What this model is

A small **CNN student** distilled from a pruned AIFS-family teacher (K1 / CREDIT-style Track B). It is intended for **laptop-scale** experimentation on **short-range African 2 m temperature** under an **analysis-forced** protocol.

It is **not** a free-running 10-day global NWP emulator and **not** an operational NMHS forecast system.

## Claim boundary

| Supported | Not supported / not claimed |
|-----------|-----------------------------|
| Cin=65 → Cout=3 (`tp`, `msl`, `2t`) | Autonomous free-run / 10-day rollout |
| Analysis-forced African **t2m** skill (+6…+24 h packaged) | Full multi-variable Week-9 free-run table |
| CPU inference (~2.5 s / +6 h step on i7-11800H) | Finished ONNX / INT8 product (optional later) |
| Open eval artefacts in the GitHub repo | LoRA / ENACTS country adapters |

**Precipitation (`tp`)** for this head failed (dry-collapse) and is **out of scope**.

## Packaged skill (Africa, 61 inits × 2023)

| Lead | RMSE (°C) | ACC | Bias (°C) |
|-----:|----------:|----:|----------:|
| +6 h | 1.38 | 0.97 | −0.12 |
| +12 h | 7.80 | 0.45 | −5.33 (systematic cold) |
| +18 h | 4.38 | 0.75 | −1.12 |
| +24 h | 3.23 | 0.84 | +1.30 |

Full tables, maps, and draft paper: see the GitHub `reports/` folder (including `DRAFT_PAPER_MVULA_V5.pdf`).

## How to load (PyTorch)

```python
import torch
from huggingface_hub import hf_hub_download

# After you upload this checkpoint to your model repo:
ckpt_path = hf_hub_download(
    repo_id="msovara/mvula-v5-student",
    filename="student_global_stable_v5.ckpt",
)
blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
# Prefer the GitHub package for the CNN class:
#   pip install "git+https://github.com/msovara/lapai-forecast-africa.git"
#   from lapai_inference.model import LapAIStudentCNN, LapAIStudentConfig
```

End-user paths on GitHub: `python run_mvula.py info|bench|dashboard` or Apptainer — see repo README.

## Intended users

NMHS / university experimenters, students, and AfriClimate AI community members who want a **small, inspectable** African temperature demo on a laptop or in a Hugging Face Space — not a replacement for official forecasts.

## Licence

- **Code / this card:** Apache-2.0  
- **Student weights:** research and evaluation use (see GitHub `NOTICE`)  
- **Teacher (AIFS / Anemoi) weights:** remain under their original terms — **not** redistributed here  

## Citation / credit

Team Mvula (Code for Earth 2026). Mentors: Shruti Nath, Rendani Mbuvha, Mario Santa Cruz López.  
GPU training/eval support: Cassava AI Factory. HPC: CHPC Lengau.

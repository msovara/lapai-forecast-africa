# Upload Mvula to Hugging Face

This folder is a **skeleton**. It does not create Hub repos by itself. Use the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) after `huggingface-cli login`.

Suggested names (replace `YOURUSER`):

| Hub repo | Source in this git tree |
|----------|-------------------------|
| `YOURUSER/mvula-v5-student` (model) | `huggingface/model/` + `models/student_global_stable_v5.ckpt` |
| `YOURUSER/mvula-v5-demo` (Space) | `huggingface/space/` |

## 1. Model repo

```bash
cd /path/to/lapai-forecast
huggingface-cli repo create mvula-v5-student --type model  # once

# README = model card
huggingface-cli upload YOURUSER/mvula-v5-student huggingface/model/README.md README.md

# Weights (~9 MiB)
huggingface-cli upload YOURUSER/mvula-v5-student \
  models/student_global_stable_v5.ckpt \
  student_global_stable_v5.ckpt
```

Edit `huggingface/model/README.md` and replace `YOUR_HF_USER/mvula-v5-student` with your real id before or after upload.

## 2. Space

```bash
# Optional gallery images
cp reports/figures/mvula_fig10_t2m_spatial_6h_24h_quad.png huggingface/space/assets/
cp reports/figures/mvula_fig11_t2m_xai_attribution.png huggingface/space/assets/

huggingface-cli repo create mvula-v5-demo --type space --space_sdk gradio  # once

huggingface-cli upload YOURUSER/mvula-v5-demo huggingface/space . --repo-type space
```

In the Space settings, add env:

- `HF_MODEL_ID=YOURUSER/mvula-v5-student`

For the live forward tab, the Space must be able to `import lapai_inference`. When GitHub is public, uncomment the git dependency in `space/requirements.txt`. While private, either:

- make a brief public mirror of `lapai_inference/`, or  
- vendor `lapai_inference/model.py` into the Space.

## 3. Link from GitHub README

After Hub URLs exist, add them under README “Try it” / packaged results (badges optional).

## 4. LinkedIn

Use the end-user post with:

- GitHub repo + PDF  
- `https://huggingface.co/YOURUSER/mvula-v5-student`  
- `https://huggingface.co/spaces/YOURUSER/mvula-v5-demo`

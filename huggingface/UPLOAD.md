# Upload Mvula to Hugging Face (`msovara`)

Profile: [huggingface.co/msovara](https://huggingface.co/msovara)

| Hub repo | Source in this git tree |
|----------|-------------------------|
| [`msovara/mvula-v5-student`](https://huggingface.co/msovara/mvula-v5-student) (model) | `huggingface/model/` + `models/student_global_stable_v5.ckpt` |
| [`msovara/mvula-v5-demo`](https://huggingface.co/spaces/msovara/mvula-v5-demo) (Space) | `huggingface/space/` |

## 0. Login (once)

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```

## 1. Model repo

```bash
cd /path/to/lapai-forecast

huggingface-cli repo create mvula-v5-student --type model  # once

huggingface-cli upload msovara/mvula-v5-student huggingface/model/README.md README.md

huggingface-cli upload msovara/mvula-v5-student \
  models/student_global_stable_v5.ckpt \
  student_global_stable_v5.ckpt
```

## 2. Space

```bash
# Gallery images (already in assets/ on git)
huggingface-cli repo create mvula-v5-demo --type space --space_sdk gradio  # once

huggingface-cli upload msovara/mvula-v5-demo huggingface/space . --repo-type space
```

In the Space **Settings → Variables**, set:

- `HF_MODEL_ID=msovara/mvula-v5-student`

For the live forward tab, `lapai_inference` must import. When GitHub is public, uncomment the git line in `space/requirements.txt`. While the GitHub repo is private, vendor `lapai_inference` or keep the gallery-only tabs.

## 3. LinkedIn URLs

- https://github.com/msovara/lapai-forecast-africa  
- https://huggingface.co/msovara/mvula-v5-student  
- https://huggingface.co/spaces/msovara/mvula-v5-demo  
- Profile: https://huggingface.co/msovara

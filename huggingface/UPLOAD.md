# Upload Mvula to Hugging Face

Profile: [huggingface.co/msovara](https://huggingface.co/msovara) · Org: [C4E-Mvula](https://huggingface.co/C4E-Mvula)

| Hub repo | Status |
|----------|--------|
| [`C4E-Mvula/mvula-v5-student`](https://huggingface.co/C4E-Mvula/mvula-v5-student) | Model card + `student_global_stable_v5.ckpt` |
| [`C4E-Mvula/mvula-v5-demo`](https://huggingface.co/spaces/C4E-Mvula/mvula-v5-demo) | **Static** end-user Space (free) |

Gradio Spaces need HF Pro (personal) or Team (org). Interactive app source remains in `huggingface/space/` for later.

## Refresh uploads

```powershell
$hf = "$env:USERPROFILE\anaconda3\envs\lapai-anemoi\Scripts\hf.exe"
& $hf upload C4E-Mvula/mvula-v5-student huggingface\model\README.md README.md
& $hf upload C4E-Mvula/mvula-v5-student models\student_global_stable_v5.ckpt student_global_stable_v5.ckpt
& $hf upload C4E-Mvula/mvula-v5-demo huggingface\space-static . --repo-type space
```

## LinkedIn URLs

- https://github.com/msovara/lapai-forecast-africa  
- https://huggingface.co/C4E-Mvula/mvula-v5-student  
- https://huggingface.co/spaces/C4E-Mvula/mvula-v5-demo  

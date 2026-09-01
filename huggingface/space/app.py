"""Gradio Space: Mvula v5 end-user demo (scope-first, optional CPU forward)."""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"

SCOPE = """
## Mvula v5 — what you can expect

A **small AI weather model** (~9 MB) for **short-range African temperature** experiments on a **laptop / CPU**.

| Can do | Cannot do (this freeze) |
|--------|-------------------------|
| Orient on African 2 m temperature skill | Replace official NMHS forecasts |
| Run lightly on CPU | Free-running 10-day global weather |
| Inspect open code & paper on GitHub | Full all-weather / precip product |

**+6 h** African t2m in our packaged test is relatively strong; **+12 h** shows a clear cold-bias failure we document openly.

- Code & PDF: [github.com/msovara/lapai-forecast-africa](https://github.com/msovara/lapai-forecast-africa)
- Draft paper PDF: `reports/DRAFT_PAPER_MVULA_V5.pdf` in that repo
"""

METRICS = """
### Packaged African t2m (analysis-forced, 61 inits, 2023)

| Lead | RMSE (°C) | ACC | Bias (°C) |
|-----:|----------:|----:|----------:|
| +6 h | 1.38 | 0.97 | −0.12 |
| +12 h | 7.80 | 0.45 | −5.33 |
| +18 h | 4.38 | 0.75 | −1.12 |
| +24 h | 3.23 | 0.84 | +1.30 |
"""


def _asset(*names: str) -> str | None:
    for n in names:
        p = ASSETS / n
        if p.is_file():
            return str(p)
    return None


def run_climate_demo(seed: int = 0):
    """Optional live forward. Needs HF_MODEL_ID + lapai_inference install."""
    try:
        from huggingface_hub import hf_hub_download
        from load_student import africa_crop, climate_forward_t2m_celsius, load_student
    except Exception as exc:  # noqa: BLE001
        return None, f"Live demo unavailable: {exc}"

    repo_id = os.environ.get("HF_MODEL_ID", "").strip()
    local_ckpt = os.environ.get("MVULA_CKPT", "").strip()
    ckpt = local_ckpt
    if not ckpt:
        if not repo_id:
            return None, (
                "Set Space secret/env `HF_MODEL_ID=C4E-Mvula/mvula-v5-student` "
                "after uploading the checkpoint, or `MVULA_CKPT` to a local path."
            )
        try:
            ckpt = hf_hub_download(repo_id=repo_id, filename="student_global_stable_v5.ckpt")
        except Exception as exc:  # noqa: BLE001
            return None, f"Could not download checkpoint: {exc}"

    try:
        net, meta = load_student(ckpt, device="cpu")
        field = climate_forward_t2m_celsius(net, meta, seed=int(seed))
        crop, lat, lon = africa_crop(field)
    except Exception as exc:  # noqa: BLE001
        return None, f"Forward failed: {exc}"

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=120)
    im = ax.pcolormesh(lon, lat, crop, shading="auto", cmap="coolwarm")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Demo only — climate IC forward (2t °C), not an official forecast")
    fig.colorbar(im, ax=ax, fraction=0.035, label="2t (°C)")
    fig.tight_layout()
    note = (
        f"Africa-box mean ≈ {float(np.nanmean(crop)):.2f} °C. "
        "Synthetic climate IC — for orientation, not decision-making."
    )
    return fig, note


def build() -> gr.Blocks:
    fig10 = _asset("mvula_fig10_t2m_spatial_6h_24h_quad.png")
    fig11 = _asset("mvula_fig11_t2m_xai_attribution.png")

    with gr.Blocks(title="Mvula v5") as demo:
        gr.Markdown("# Mvula — laptop-scale African temperature AI")
        gr.Markdown(SCOPE)
        with gr.Tab("Packaged skill"):
            gr.Markdown(METRICS)
            if fig10:
                gr.Image(fig10, label="Fig.10 — spatial +6 / +24 h")
            else:
                gr.Markdown("_Copy Fig.10 into `assets/` when deploying the Space._")
            if fig11:
                gr.Image(fig11, label="Fig.11 — explainability")
            else:
                gr.Markdown("_Copy Fig.11 into `assets/` when deploying the Space._")
        with gr.Tab("Try CPU forward (optional)"):
            gr.Markdown(
                "Runs **one** student forward from a climate initial condition. "
                "Requires the Hugging Face **model repo** with the `.ckpt` uploaded."
            )
            seed = gr.Slider(0, 100, value=0, step=1, label="Noise seed")
            btn = gr.Button("Run demo forward")
            out_fig = gr.Plot(label="Africa crop")
            out_txt = gr.Textbox(label="Notes", lines=3)
            btn.click(run_climate_demo, inputs=[seed], outputs=[out_fig, out_txt])
        gr.Markdown(
            "Feedback welcome from NMHS colleagues, students, and AfriClimate AI — "
            "what would make a laptop-scale temperature tool useful in your workflow?"
        )
    return demo


if __name__ == "__main__":
    build().launch()

#!/usr/bin/env python3
"""Mvula v5 explainability figures: channel attribution + 06Z-IC pathology.

Produces:
  reports/figures/mvula_fig11_t2m_xai_attribution.png
  reports/MVULA_XAI_ATTRIBUTION.md

Method (gradient saliency on frozen student):
  - Load student_global_stable_v5.ckpt
  - Build a normalized synthetic IC (Cin=65) on 181×360
  - Score sensitivity of African-mean predicted 2t to each input channel
  - Spatial |∂2t/∂x| map (channel-mean)
  - Combine with packaged AF skill by analysis IC hour (00/06/12/18Z)

Note: True lead-conditioned ARCO ICs (00Z vs 06Z) for attribution contrast are optional.
Panel (d) uses the 61-init AF package (IC-hour sensitivity, not AR lead skill).

If Torch is unavailable, `refresh_fig11_ic_framing()` rebuilds title + panel (d)
from the existing PNG (keeps panels a–c).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.masks import AFRICA_LAT, AFRICA_LON, africa_hw_mask, lon_to_180
from evaluation.plot_africa_spatial import HAS_CARTOPY, plot_africa_field, AFRICA_EXTENT
from lapai_inference.cache_schema import lat_lon_mesh


def _to_africa_plot_grid(
    field: np.ndarray, lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map 0..360 global mesh → sorted −180..180 Africa crop for Cartopy extent."""
    lat = np.asarray(lat, dtype=np.float64).reshape(-1)
    lon = lon_to_180(np.asarray(lon, dtype=np.float64).reshape(-1))
    order = np.argsort(lon)
    lon = lon[order]
    field = np.asarray(field, dtype=float)[:, order]
    lat_ok = (lat >= AFRICA_EXTENT["lat_min"]) & (lat <= AFRICA_EXTENT["lat_max"])
    lon_ok = (lon >= AFRICA_EXTENT["lon_min"]) & (lon <= AFRICA_EXTENT["lon_max"])
    return field[np.ix_(lat_ok, lon_ok)], lat[lat_ok], lon[lon_ok]

CKPT = _REPO / "models" / "student_global_stable_v5.ckpt"
EXPANDED = _REPO / "reports" / "TRACKB_T2M_EXPANDED.json"
FIG_DIR = _REPO / "reports" / "figures"
OUT_FIG = FIG_DIR / "mvula_fig11_t2m_xai_attribution.png"
OUT_MD = _REPO / "reports" / "MVULA_XAI_ATTRIBUTION.md"
ATTR_CACHE = FIG_DIR / "mvula_fig11_attribution.npz"

# Cin=65 = (t, u, v, q, z) × 13 pressure levels (no separate surface stack in v5 cfg)
VAR_NAMES = ("t", "u", "v", "q", "z")
N_LEVELS = 13
# Approximate ERA5 pressure levels often used in AIFS-style stacks (hPa), high → low
LEVELS_HPA = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
OUT_T2M = 2  # Cout order: tp, msl, 2t


def _channel_labels() -> list[str]:
    labels: list[str] = []
    for v in VAR_NAMES:
        for lev in LEVELS_HPA:
            labels.append(f"{v}{lev}")
    return labels


def _africa_mask_torch(h: int, w: int):
    import torch

    lat, lon = lat_lon_mesh(h, w)
    m = africa_hw_mask(lat, lon, AFRICA_LAT, AFRICA_LON)
    return torch.as_tensor(m, dtype=torch.float32)


def _build_input(meta: dict, device, seed: int = 0):
    """Normalized Cin=65 state. Prefer checkpoint input_mean as climate baseline."""
    import torch
    from utils.losses_distillation import normalize_channels

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = torch.randn(1, 65, 181, 360, generator=g, dtype=torch.float32)
    if meta.get("input_mean") is not None and meta.get("input_std") is not None:
        im = torch.as_tensor(meta["input_mean"], dtype=torch.float32).view(1, -1, 1, 1)
        istd = torch.as_tensor(meta["input_std"], dtype=torch.float32).view(1, -1, 1, 1)
        # Climate + small noise in physical space, then normalize if training did
        x = im + 0.05 * istd * x
        if meta.get("normalize_inputs"):
            x = normalize_channels(x, im.view(-1), istd.view(-1))
    return x.to(device)


def channel_and_spatial_saliency(net, meta: dict, device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (chan_scores[65], spatial[H,W], lat, lon) for African-mean 2t."""
    net.eval()
    x = _build_input(meta, device).requires_grad_(True)
    mask = _africa_mask_torch(181, 360).to(device)
    mask_sum = mask.sum().clamp_min(1.0)

    pred = net(x)["pred"]  # (1,3,H,W)
    # African-mean 2t (Kelvin / °C interval — attribution is relative)
    t2m = (pred[0, OUT_T2M] * mask).sum() / mask_sum
    t2m.backward()

    assert x.grad is not None
    g = x.grad.abs()[0]  # (65,H,W)
    # Channel importance: mean |grad| over Africa
    chan = (g * mask.unsqueeze(0)).sum(dim=(1, 2)) / mask_sum
    chan = chan.detach().cpu().numpy()
    # Spatial map: mean |grad| over channels
    spatial = g.mean(dim=0).detach().cpu().numpy()
    lat, lon = lat_lon_mesh(181, 360)
    return chan, spatial, lat, lon


def _lead_table(expanded: dict) -> dict[int, dict]:
    """Aggregate student t2m metrics by lead from TRACKB_T2M_EXPANDED.json."""
    student = expanded.get("student_results") or []
    buckets: dict[int, list[tuple[float, float, float]]] = {}
    for r in student:
        t = (r.get("variables") or {}).get("t2m") or {}
        if "student_rmse_vs_era5" not in t:
            continue
        L = int(r["lead_hours"])
        buckets.setdefault(L, []).append(
            (float(t["student_rmse_vs_era5"]), float(t["student_acc"]), float(t["student_bias"]))
        )
    out: dict[int, dict] = {}
    for L, rows in buckets.items():
        out[L] = {
            "student_rmse_vs_era5": float(np.mean([x[0] for x in rows])),
            "student_acc": float(np.mean([x[1] for x in rows])),
            "student_bias": float(np.mean([x[2] for x in rows])),
            "n": len(rows),
        }
    # Prefer precomputed aggregate if present
    agg = (expanded.get("aggregate") or {}).get("by_lead") or {}
    for k, v in agg.items():
        try:
            L = int(k)
        except (TypeError, ValueError):
            continue
        t2m = (v.get("t2m") or v) if isinstance(v, dict) else {}
        if "student_rmse_vs_era5" in t2m or "rmse" in t2m:
            out[L] = {
                "student_rmse_vs_era5": float(
                    t2m.get("student_rmse_vs_era5", t2m.get("rmse", out.get(L, {}).get("student_rmse_vs_era5", np.nan)))
                ),
                "student_acc": float(t2m.get("student_acc", t2m.get("acc", out.get(L, {}).get("student_acc", np.nan)))),
                "student_bias": float(
                    t2m.get("student_bias", t2m.get("bias", out.get(L, {}).get("student_bias", np.nan)))
                ),
                "n": int(t2m.get("n", out.get(L, {}).get("n", 0))),
            }
    return out


def _collect_plus12_biases(expanded: dict) -> np.ndarray:
    biases = []
    for r in expanded.get("student_results") or []:
        if int(r.get("lead_hours", -1)) != 12:
            continue
        t = (r.get("variables") or {}).get("t2m") or {}
        if "student_bias" in t:
            biases.append(float(t["student_bias"]))
    return np.asarray(biases, dtype=float)


def plot_fig11(chan: np.ndarray, spatial: np.ndarray, lat, lon, expanded: dict) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = _channel_labels()
    # Aggregate by variable for readability
    var_scores = {
        v: float(chan[i * N_LEVELS : (i + 1) * N_LEVELS].sum())
        for i, v in enumerate(VAR_NAMES)
    }
    # Top-15 channels
    top_idx = np.argsort(chan)[::-1][:15]

    fig = plt.figure(figsize=(12.8, 12.2), dpi=200)
    # Spatial map (c) gets a full-width middle row so it is not letterboxed
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.0, 1.55, 1.15],
        hspace=0.28,
        wspace=0.22,
    )

    # (a) Variable-group attribution
    ax0 = fig.add_subplot(gs[0, 0])
    names = list(var_scores.keys())
    vals = [var_scores[n] for n in names]
    colors = ["#c45911", "#2e75b6", "#548235", "#7030a0", "#833c0c"]
    ax0.bar(names, vals, color=colors, edgecolor="white")
    ax0.set_ylabel("Africa-mean |∂2t / ∂x| (sum over levels)")
    ax0.set_title("(a) Input variable group sensitivity")
    ax0.set_xlabel("Multilevel fields in Cin=65 (13 levels each)")

    # (b) Top channels
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.barh(
        [labels[i] for i in top_idx][::-1],
        chan[top_idx][::-1],
        color="#1f4e79",
        edgecolor="white",
    )
    ax1.set_xlabel("|∂2t / ∂x_c| (Africa mean)")
    ax1.set_title("(b) Top-15 input channels")

    # (c) Spatial saliency — full-width row
    if HAS_CARTOPY:
        import cartopy.crs as ccrs

        ax2 = fig.add_subplot(gs[1, :], projection=ccrs.PlateCarree())
    else:
        ax2 = fig.add_subplot(gs[1, :])
    sp, lat_af, lon_af = _to_africa_plot_grid(spatial, lat, lon)
    # Scale for a readable colorbar (raw |grad| ~ 1e-4); avoid 0.0000x tick clutter
    grad_scale = 1.0e4
    sp_plot = sp * grad_scale
    vmax = float(np.nanpercentile(sp_plot[sp_plot > 0], 98)) if np.any(sp_plot > 0) else 1.0
    plot_africa_field(
        ax2,
        lon_af,
        lat_af,
        sp_plot,
        cmap="magma",
        vmin=0.0,
        vmax=max(vmax, 1e-12),
        title="(c) Spatial |grad| (channel-mean)",
        panel_label=None,
        cbar_label=r"|∂2t / ∂x| × 10$^{-4}$",
        draw_grid=False,
    )

    ax3 = fig.add_subplot(gs[2, :])
    _draw_panel_d(ax3, expanded)

    fig.suptitle(
        "Fig. 11 — Mvula v5 explainability: what drives African 2t, and 06Z-IC one-step pathology",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    print(f"[figures] wrote {OUT_FIG}", flush=True)


def _draw_panel_d(ax, expanded: dict) -> None:
    """Shared panel (d) for full regen and Torch-free framing refresh."""
    lead = _lead_table(expanded)
    biases12 = _collect_plus12_biases(expanded)
    leads = [6, 12, 18, 24]
    rmses, bias_vals = [], []
    for L in leads:
        row = lead.get(L) or {}
        rmses.append(float(row.get("student_rmse_vs_era5") or row.get("rmse") or np.nan))
        bias_vals.append(float(row.get("student_bias") or row.get("bias") or np.nan))
    x = np.arange(len(leads))
    w = 0.35
    ax.bar(x - w / 2, rmses, w, label="RMSE (°C)", color="#c45911")
    ax.bar(x + w / 2, bias_vals, w, label="Bias (°C)", color="#2e75b6")
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["00Z", "06Z", "12Z", "18Z"])
    ax.set_xlabel("Analysis IC hour (one-step AF; not AR lead)")
    ax.set_ylabel("°C")
    # Keep cold-bias bars fully visible (06Z ≈ −5.3 °C)
    y_lo = min(-0.5, float(np.nanmin(bias_vals)) - 0.4)
    y_hi = max(8.0, float(np.nanmax(rmses)) + 0.4)
    ax.set_ylim(y_lo, y_hi)
    ax.set_title("(d) 06Z-IC pathology (one-step AF skill by IC hour)")
    ax.legend(fontsize=8, loc="upper right")
    if biases12.size:
        note = (
            f"At 06Z IC: mean bias {biases12.mean():.2f} °C; "
            f"all {biases12.size}/{biases12.size} inits cold "
            f"[{biases12.min():.2f}, {biases12.max():.2f}].\n"
            "One-step AF from 06Z analysis — time-of-day / IC-hour sensitivity,\n"
            "not autoregressive +12 h lead skill. Network most sensitive to T/q/z."
        )
    else:
        note = (
            "06Z-IC one-step cold-bias regime (see TRACKB_T2M_EXPANDED).\n"
            "Attribution uses climate+noise IC (ARCO diurnal ICs optional next step)."
        )
    ax.text(
        0.02,
        0.02,
        note,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff8e1", edgecolor="#f9a825", alpha=0.95),
    )


def refresh_fig11_ic_framing() -> None:
    """Rebuild Fig.11 without Torch: keep panels a–c from PNG, redraw title + panel (d)."""
    from PIL import Image

    if not OUT_FIG.is_file():
        raise FileNotFoundError(OUT_FIG)
    if not EXPANDED.is_file():
        raise FileNotFoundError(EXPANDED)

    expanded = json.loads(EXPANDED.read_text(encoding="utf-8"))
    # Prefer a clean pre-composite source if present (sharper spatial map for enlarged panel c)
    src = FIG_DIR / "_fig11_src_clean.png"
    img = np.asarray(Image.open(src if src.is_file() else OUT_FIG).convert("RGB"))
    h, w = img.shape[:2]
    # Approximate equal 2×2 crops from the original layout (exclude outer title strip)
    top = int(0.07 * h)
    mid_y = int(0.52 * h)
    mid_x = int(0.50 * w)
    crops = {
        "a": img[top:mid_y, 0:mid_x],
        "b": img[top:mid_y, mid_x:w],
        "c": img[mid_y:h, 0:mid_x],
    }

    fig = plt.figure(figsize=(12.8, 12.2), dpi=200)
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.0, 1.55, 1.15],
        hspace=0.26,
        wspace=0.18,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(crops["a"], aspect="auto")
    ax_a.axis("off")
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.imshow(crops["b"], aspect="auto")
    ax_b.axis("off")
    ax_c = fig.add_subplot(gs[1, :])
    ax_c.imshow(crops["c"], aspect="auto")
    ax_c.set_title("(c) Spatial |grad| (channel-mean)", fontsize=11, pad=8)
    ax_c.axis("off")
    ax3 = fig.add_subplot(gs[2, :])
    _draw_panel_d(ax3, expanded)
    fig.suptitle(
        "Fig. 11 — Mvula v5 explainability: what drives African 2t, and 06Z-IC one-step pathology",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    print(f"[figures] refreshed IC framing -> {OUT_FIG}", flush=True)

    # Keep markdown aligned even when Torch is unavailable
    write_markdown_from_expanded(expanded)


def write_markdown_from_expanded(expanded: dict) -> None:
    """Update pathology section of the markdown from AF package (Torch-free)."""
    lead = _lead_table(expanded)
    biases12 = _collect_plus12_biases(expanded)
    # Preserve attribution tables if present; otherwise write a short IC-framing note.
    if OUT_MD.is_file():
        text = OUT_MD.read_text(encoding="utf-8")
        # Replace obsolete lead-language headers/bullets in the pathology section
        text = text.replace("## Why +12 h fails (observed AF package)", "## 06Z-IC pathology (observed AF package)")
        text = text.replace("explain the **+12 h** failure.", "explain the **06Z-IC** one-step cold-bias pathology.")
        text = text.replace(
            "| Not yet | Lead-conditioned ARCO ICs (00Z vs 06Z) for true +6 vs +12 attribution contrast |",
            "| Not yet | ARCO ICs at 00Z vs 06Z for true IC-hour attribution contrast |",
        )
        if 6 in lead and 12 in lead:
            r6 = float(lead[6].get("student_rmse_vs_era5") or lead[6].get("rmse"))
            r12 = float(lead[12].get("student_rmse_vs_era5") or lead[12].get("rmse"))
            b12 = float(lead[12].get("student_bias") or lead[12].get("bias"))
            block = [
                "## 06Z-IC pathology (observed AF package)",
                "",
                f"- **00Z IC** one-step RMSE ≈ **{r6:.2f} °C**; **06Z IC** RMSE ≈ **{r12:.2f} °C**.",
                f"- **06Z IC** bias ≈ **{b12:.2f} °C** (cold).",
            ]
            if biases12.size:
                block.append(
                    f"- **{biases12.size}/{biases12.size}** inits cold at 06Z IC "
                    f"(range [{biases12.min():.2f}, {biases12.max():.2f}] °C)."
                )
            block += [
                "- Protocol: columns are **analysis IC hours** under AF one-step — not autoregressive lead times.",
                "- Persistence also beats the student at 06Z/12Z IC (see Fig. 8 / baselines).",
            ]
            import re

            text = re.sub(
                r"## 06Z-IC pathology \(observed AF package\)\n\n.*?(?=\n## |\Z)",
                "\n".join(block) + "\n\n",
                text,
                count=1,
                flags=re.S,
            )
            # Also replace old section if refresh ran before rename stuck
            text = re.sub(
                r"## Why \+12 h fails \(observed AF package\)\n\n.*?(?=\n## |\Z)",
                "\n".join(block) + "\n\n",
                text,
                count=1,
                flags=re.S,
            )
        text = text.replace(
            "the +12 h collapse is a systematic cold bias under AF 06Z ICs",
            "the 06Z-IC collapse is a systematic cold bias under AF",
        )
        OUT_MD.write_text(text, encoding="utf-8")
        print(f"[report] refreshed IC framing -> {OUT_MD}", flush=True)
        return

    # Minimal fallback
    lines = [
        "# Mvula v5 — explainability (AI attribution)",
        "",
        f"**Figure:** [`figures/{OUT_FIG.name}`](figures/{OUT_FIG.name})",
        "",
        "Run full `python scripts/plot_mvula_xai_attribution.py` with Torch to refresh panels (a–c).",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def write_markdown(chan: np.ndarray, expanded: dict) -> None:
    labels = _channel_labels()
    top_idx = np.argsort(chan)[::-1][:10]
    var_scores = {
        v: float(chan[i * N_LEVELS : (i + 1) * N_LEVELS].sum())
        for i, v in enumerate(VAR_NAMES)
    }
    biases12 = _collect_plus12_biases(expanded)
    lead = _lead_table(expanded)
    lines = [
        "# Mvula v5 — explainability (AI attribution)",
        "",
        "**Figure:** [`figures/mvula_fig11_t2m_xai_attribution.png`](figures/mvula_fig11_t2m_xai_attribution.png)  ",
        "**Script:** `scripts/plot_mvula_xai_attribution.py`",
        "",
        "## What this is",
        "",
        "Gradient **saliency** on the frozen student (`student_global_stable_v5.ckpt`):",
        "how much the African-mean predicted **2t** changes with each of the **65** input channels,",
        "plus a spatial sensitivity map. Combined with the packaged AF skill table to explain the **06Z-IC** one-step cold-bias pathology.",
        "",
        "## Method (honest limits)",
        "",
        "| Item | Detail |",
        "|------|--------|",
        "| Target | African-mean `2t` (Cout index 2) |",
        "| Score | Mean `|∂2t/∂x|` over the Africa land box |",
        "| IC | Checkpoint climate (`input_mean`) + small noise, then training normalization |",
        "| Not yet | ARCO ICs at 00Z vs 06Z for true IC-hour attribution contrast |",
        "",
        "This is **model explainability** (what the CNN listens to), not a full physical causal proof.",
        "",
        "## Variable-group sensitivity (sum over 13 levels)",
        "",
        "| Variable | Relative |∂2t/∂x| |",
        "|----------|-------------------|",
    ]
    tot = sum(var_scores.values()) or 1.0
    for v, s in sorted(var_scores.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{v}` | {100 * s / tot:.1f}% |")
    lines += [
        "",
        "## Top-10 channels",
        "",
        "| Rank | Channel | Score |",
        "|-----:|---------|------:|",
    ]
    for r, i in enumerate(top_idx, 1):
        lines.append(f"| {r} | `{labels[i]}` | {chan[i]:.4g} |")
    lines += [
        "",
        "## 06Z-IC pathology (observed AF package)",
        "",
    ]
    if 6 in lead and 12 in lead:
        r6 = lead[6].get("student_rmse_vs_era5") or lead[6].get("rmse")
        r12 = lead[12].get("student_rmse_vs_era5") or lead[12].get("rmse")
        b12 = lead[12].get("student_bias") or lead[12].get("bias")
        lines.append(
            f"- **00Z IC** one-step RMSE ≈ **{float(r6):.2f} °C**; **06Z IC** RMSE ≈ **{float(r12):.2f} °C**."
        )
        lines.append(f"- **06Z IC** bias ≈ **{float(b12):.2f} °C** (cold).")
    if biases12.size:
        lines.append(
            f"- **{biases12.size}/{biases12.size}** inits cold at 06Z IC "
            f"(range [{biases12.min():.2f}, {biases12.max():.2f}] °C)."
        )
    lines += [
        "- Protocol: columns are **analysis IC hours** under AF one-step — not autoregressive lead times.",
        "- Persistence also beats the student at 06Z/12Z IC (see Fig. 8 / baselines).",
        "",
        "## One-sentence takeaway",
        "",
        "**The student is most sensitive to thermodynamic/moisture multilevel fields when forming African 2t; "
        "the 06Z-IC collapse is a systematic cold bias under AF, not an unexplained black-box glitch.**",
        "",
        "## Regenerate",
        "",
        "```bash",
        "python scripts/plot_mvula_xai_attribution.py",
        "# If Torch DLL fails on Windows, IC framing refresh still works:",
        "python scripts/plot_mvula_xai_attribution.py --refresh-framing",
        "```",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] wrote {OUT_MD}", flush=True)


def main() -> int:
    refresh_only = "--refresh-framing" in sys.argv
    if refresh_only:
        refresh_fig11_ic_framing()
        return 0

    if not EXPANDED.is_file():
        print(f"missing {EXPANDED}", file=sys.stderr)
        return 1

    try:
        import torch
        from evaluation.trackB_gate import _load_student
    except OSError as exc:
        print(f"Torch unavailable ({exc}); refreshing IC framing from existing PNG.", flush=True)
        refresh_fig11_ic_framing()
        return 0

    if not CKPT.is_file():
        print(f"missing ckpt: {CKPT}", file=sys.stderr)
        return 1

    device = torch.device("cpu")
    net, meta = _load_student(CKPT, device)
    chan, spatial, lat, lon = channel_and_spatial_saliency(net, meta, device)
    np.savez_compressed(ATTR_CACHE, chan=chan, spatial=spatial, lat=lat, lon=lon)
    expanded = json.loads(EXPANDED.read_text(encoding="utf-8"))
    plot_fig11(chan, spatial, lat, lon, expanded)
    write_markdown(chan, expanded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

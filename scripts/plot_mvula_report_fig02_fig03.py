#!/usr/bin/env python3
"""Generate Mvula report figures from TRACKB_T2M_EXPANDED.json.

Mario Santa Cruz (ECMWF) framing (2026-09): scores are analysis-forced *one-step*
predictions. With a 00Z campaign, labelled horizons map to analysis IC hours
(00/06/12/18Z), not autoregressive lead times. Headline claim = 00Z IC one-step;
multi-hour panels are IC-hour sensitivity only.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
JSON_PATH = REPO / "reports" / "TRACKB_T2M_EXPANDED.json"
FIG_DIR = REPO / "reports" / "figures"
OUT1 = FIG_DIR / "mvula_fig01_pipeline.png"
OUT1_PUB = FIG_DIR / "mvula_fig01_pipeline_publication.png"
OUT2 = FIG_DIR / "mvula_fig02_t2m_lead_curves.png"  # IC-hour sensitivity (backup)
OUT2_PRIMARY = FIG_DIR / "mvula_fig02_primary_00z_onestep.png"  # presentation headline
OUT3 = FIG_DIR / "mvula_fig03_t2m_init_lead_heatmap.png"
OUT6 = FIG_DIR / "mvula_fig06_t2m_plus12h_pathology.png"
OUT7 = FIG_DIR / "mvula_fig07_t2m_seasonal.png"
OUT8 = FIG_DIR / "mvula_fig08_t2m_baselines.png"
OUT9 = FIG_DIR / "mvula_fig09_compute_panel.png"
# Keep legacy filename as a copy target for mentors already linking fig08 compute
OUT8_COMPUTE_LEGACY = FIG_DIR / "mvula_fig08_compute_panel.png"
BENCH_JSON = REPO / "reports" / "MVULA_LAPTOP_BENCHMARK.json"
BASELINES_JSON = REPO / "reports" / "TRACKB_T2M_BASELINES.json"

SEASON_ORDER = {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}


def lead_to_ic_hour(lead_h: int, init_hour: int = 0) -> int:
    """AF protocol: IC at init+(L−6)h for a campaign with fixed init_hour (00Z → 0)."""
    return (int(init_hour) + int(lead_h) - 6) % 24


def ic_tick(lead_h: int, init_hour: int = 0) -> str:
    return f"{lead_to_ic_hour(lead_h, init_hour):02d}Z"


def fig1_pipeline() -> None:
    """Scientific pathway schematic (not a CNN layer diagram)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11.2, 3.8))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 3.8)
    ax.axis("off")

    stages = [
        {
            "x": 0.35,
            "title": "Teacher",
            "body": "K1 GraphTransformer\n(pruned Anemoi GT)",
            "foot": "~52.8 MiB",
            "fc": "#fce4d6",
            "ec": "#c45911",
        },
        {
            "x": 2.55,
            "title": "Distillation",
            "body": "Knowledge distill\nteacher → student",
            "foot": "Track B train",
            "fc": "#fff2cc",
            "ec": "#bf8f00",
        },
        {
            "x": 4.75,
            "title": "Student v5",
            "body": "InceptionNeXt CNN\nCin=65 → Cout=3\n(tp, msl, 2t)",
            "foot": "8.8 MiB · Case A",
            "fc": "#ddebf7",
            "ec": "#1f4e79",
        },
        {
            "x": 6.95,
            "title": "Evaluation",
            "body": "AF one-step only\n(+6 h native step)\nnot autoregressive",
            "foot": "Case A / no free-run",
            "fc": "#e2efda",
            "ec": "#548235",
        },
        {
            "x": 9.15,
            "title": "Verify",
            "body": "Headline: 00Z IC\nAfrican t2m one-step\n(+ IC-hour checks)",
            "foot": "n=61 inits",
            "fc": "#f4cccc",
            "ec": "#990000",
        },
    ]

    box_w, box_h = 1.85, 2.15
    y0 = 0.95

    for i, st in enumerate(stages):
        box = FancyBboxPatch(
            (st["x"], y0),
            box_w,
            box_h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.6,
            edgecolor=st["ec"],
            facecolor=st["fc"],
            mutation_aspect=0.8,
        )
        ax.add_patch(box)
        ax.text(
            st["x"] + box_w / 2,
            y0 + box_h - 0.28,
            st["title"],
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color=st["ec"],
        )
        ax.text(
            st["x"] + box_w / 2,
            y0 + box_h / 2 - 0.05,
            st["body"],
            ha="center",
            va="center",
            fontsize=8,
            color="#333333",
            linespacing=1.25,
        )
        ax.text(
            st["x"] + box_w / 2,
            y0 + 0.18,
            st["foot"],
            ha="center",
            va="bottom",
            fontsize=7.5,
            style="italic",
            color="#555555",
        )
        if i < len(stages) - 1:
            x1 = st["x"] + box_w + 0.05
            x2 = stages[i + 1]["x"] - 0.05
            arr = FancyArrowPatch(
                (x1, y0 + box_h / 2),
                (x2, y0 + box_h / 2),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.4,
                color="#666666",
            )
            ax.add_patch(arr)

    ax.text(
        5.6,
        3.45,
        "Fig. 1 — Mvula scientific pipeline (learn → verify)",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#1f4e79",
    )
    ax.text(
        5.6,
        0.35,
        "Free-run / 10-day rollout is architecturally excluded (Cout=3, no 3→65 decoder). "
        "Primary metric: AF one-step African t2m from 00Z analysis IC (not AR lead skill).",
        ha="center",
        va="center",
        fontsize=8,
        color="#444444",
    )

    fig.savefig(OUT1, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT1}")


def fig1_pipeline_publication() -> None:
    """Journal-style vertical pathway (Learn → Compress → Verify → Access).

    Equal-width FancyBbox panels; text packed with OffsetBoxes and measured so
    top/bottom interior pad stays tight and uniform.
    """
    from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

    fig, ax = plt.subplots(figsize=(7.2, 8.6), dpi=300)
    ax.set_xlim(0, 7.2)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    ax.text(
        3.6,
        8.35,
        "Figure 1. Overview of the Mvula compressed student pathway",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#1a1a1a",
    )
    ax.text(
        3.6,
        7.88,
        "Distillation of an AIFS-derived GraphTransformer teacher into an\n"
        "analysis-forced African 2 m temperature student\n"
        "(Case A: Cout = 3; free-run not supported).",
        ha="center",
        va="center",
        fontsize=8,
        color="#444444",
        linespacing=1.35,
    )

    specs = [
        (
            "LEARN",
            "A. Teacher model",
            [
                "K1 pruned Anemoi GraphTransformer (accepted Track A teacher)",
                "Full atmospheric state capability; reference skill where available",
                "Disk footprint ≈ 52.8 MiB",
            ],
            "#8B4513",
            "#FBF3EB",
        ),
        (
            "COMPRESS",
            "B. Distilled student (Mvula v5)",
            [
                "InceptionNeXt-style CNN · Cin = 65 → Cout = 3  (tp, msl, 2t)",
                "Knowledge distillation from K1; Africa-weighted training mix",
                "Disk footprint 8.8 MiB (~6× compression) · 2.17 M parameters",
                "Architecture lock (Case A): no 3→65 decoder ⇒ no free-run / 10-day rollout",
            ],
            "#1f4e79",
            "#EEF3F8",
        ),
        (
            "VERIFY",
            "C. African t2m verification",
            [
                "Protocol: analysis-forced one-step (native +6 h); not autoregressive rollout",
                "Domain: Africa · ARCO ERA5 · n = 61 × 00Z inits (2023)",
                "Headline claim: one-step from 00Z analysis IC (RMSE≈1.38 °C; +41.8% vs persistence)",
                "Other labelled horizons = IC-hour sensitivity (06/12/18Z), not AR lead times",
            ],
            "#2E5A1C",
            "#F1F6ED",
        ),
        (
            "ACCESS",
            "D. Accessibility contribution",
            [
                "CPU inference on a consumer laptop (measured): ~2.53 s / +6 h step",
                "Peak RSS ≈ 1.2 GiB · GPU not required for the student head",
                "Intended use: experimental AF +6 h t2m research tool — not ops NWP",
            ],
            "#6B2D5C",
            "#F7EFF4",
        ),
    ]

    # Fixed equal panel width (data coords), left-aligned text inside
    x0, box_w = 0.72, 5.85
    text_x = x0 + 0.18
    pad_y = 0.07
    arrow_gap = 0.22

    y_cursor = 7.15
    prev_box_bot = None
    inv = ax.transData.inverted()

    for phase, title, lines, edge, face in specs:
        title_ta = TextArea(
            title,
            textprops={
                "color": edge,
                "fontsize": 9,
                "fontweight": "bold",
                "fontfamily": "sans-serif",
            },
        )
        body_ta = TextArea(
            "\n".join(lines),
            textprops={
                "color": "#222222",
                "fontsize": 8,
                "fontfamily": "sans-serif",
                "linespacing": 1.25,
            },
        )
        pack = VPacker(children=[title_ta, body_ta], align="left", pad=0, sep=5)

        # Text top just below pad; measure pack, then draw equal-width frame around it
        ab = AnnotationBbox(
            pack,
            (text_x, y_cursor - pad_y),
            xycoords="data",
            box_alignment=(0.0, 1.0),
            frameon=False,
            pad=0.0,
            zorder=3,
        )
        ax.add_artist(ab)
        fig.canvas.draw()
        tb = ab.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(inv)

        box_top = y_cursor
        box_bot = tb.y0 - pad_y
        ax.add_patch(
            FancyBboxPatch(
                (x0, box_bot),
                box_w,
                box_top - box_bot,
                boxstyle="round,pad=0.0,rounding_size=0.05",
                linewidth=1.2,
                edgecolor=edge,
                facecolor=face,
                zorder=1,
            )
        )

        ax.text(
            0.28,
            (box_top + box_bot) / 2,
            phase,
            ha="left",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color="#888888",
            rotation=90,
            zorder=3,
        )

        if prev_box_bot is not None:
            ax.add_patch(
                FancyArrowPatch(
                    (3.6, prev_box_bot - 0.02),
                    (3.6, box_top + 0.02),
                    arrowstyle="-|>",
                    mutation_scale=18,
                    linewidth=1.5,
                    color="#555555",
                    zorder=2,
                )
            )

        prev_box_bot = box_bot
        y_cursor = box_bot - arrow_gap

    # Footer sits just under panel D (not pinned to the axes origin)
    footer_gap = 0.20
    footer_h = 0.38
    footer_top = prev_box_bot - footer_gap
    footer_bot = footer_top - footer_h
    ax.add_patch(
        Rectangle((0.35, footer_bot), 6.5, footer_h, facecolor="#f0f0f0", edgecolor="none", zorder=0)
    )
    ax.text(
        3.6,
        footer_bot + footer_h / 2,
        "Primary scientific question: can an AIFS-derived model be compressed into something\n"
        "accessible while retaining useful African short-range t2m skill?",
        ha="center",
        va="center",
        fontsize=7,
        color="#333333",
        style="italic",
        linespacing=1.25,
        zorder=1,
    )
    ax.set_ylim(footer_bot - 0.12, 8.6)

    fig.savefig(OUT1_PUB, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT1_PUB}")

def load_rows():
    blob = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    student = blob["student_results"]
    k1 = blob.get("k1_results") or []
    return blob, student, k1


def fig2_primary_00z(student, k1) -> None:
    """Presentation headline: one-step AF skill from 00Z analysis IC only."""
    rows = [r for r in student if int(r["lead_hours"]) == 6]
    rmse = np.array([float(r["variables"]["t2m"]["student_rmse_vs_era5"]) for r in rows])
    acc = np.array([float(r["variables"]["t2m"]["student_acc"]) for r in rows])
    bias = np.array([float(r["variables"]["t2m"]["student_bias"]) for r in rows])
    k1_rows = [r for r in k1 if int(r["lead_hours"]) == 6]
    k1_rmse = (
        float(np.mean([float(r["variables"]["t2m"]["rmse"]) for r in k1_rows])) if k1_rows else float("nan")
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.0), constrained_layout=True)
    panels = [
        (axes[0], "RMSE (°C)", float(rmse.mean()), float(rmse.std(ddof=1)), k1_rmse, "#1f4e79"),
        (axes[1], "ACC", float(acc.mean()), float(acc.std(ddof=1)), None, "#2e5a1c"),
        (axes[2], "Bias (°C)", float(bias.mean()), float(bias.std(ddof=1)), None, "#6b2d5c"),
    ]
    k1_handle = None
    for ax, ylabel, mean, std, k1v, color in panels:
        ax.bar([0], [mean], color=color, width=0.45, zorder=2)
        ax.errorbar([0], [mean], yerr=[std], fmt="none", ecolor="#333333", capsize=4, zorder=3)
        ax.set_xticks([0])
        ax.set_xticklabels(["00Z IC\n→ one +6 h step"])
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3, zorder=0)
        ax.set_xlim(-0.75, 0.75)
        # Clear room above the error bar for the numeric label
        top = mean + abs(std)
        bottom = mean - abs(std)
        if ylabel.startswith("Bias"):
            pad = 0.35 * max(abs(top), abs(bottom), 0.2)
            ax.set_ylim(min(0.0, bottom) - pad, max(0.0, top) + pad)
            ax.axhline(0.0, color="#666666", lw=0.8, zorder=1)
        elif ylabel == "ACC":
            ax.set_ylim(0.0, 1.08)
        else:
            ax.set_ylim(0.0, top + 0.28)
        label = f"{mean:.3g}" if ylabel == "ACC" else f"{mean:.2f}"
        # Numeric value above/right of the bar, clear of the error bar
        if ylabel.startswith("Bias"):
            y_text = max(0.0, top) + 0.04
        elif ylabel == "ACC":
            y_text = mean + abs(std) + 0.025
        else:
            y_text = top + 0.06
        ax.annotate(
            label,
            xy=(0, mean),
            xytext=(0.32, y_text),
            fontsize=10,
            ha="left",
            va="bottom",
            annotation_clip=False,
        )
        if k1v is not None and isinstance(k1v, (int, float)) and k1v == k1v:
            k1_handle = ax.axhline(
                k1v,
                color="#c45911",
                ls="--",
                lw=1.4,
                zorder=1,
                label=f"K1 teacher (RMSE={k1v:.2f}, n={len(k1_rows)})",
            )

    legend_handles = [
        Patch(facecolor="#1f4e79", edgecolor="none", label="Student v5 (bars)"),
    ]
    if k1_handle is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#c45911",
                ls="--",
                lw=1.4,
                label=k1_handle.get_label(),
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "Fig. 2 (primary) — Headline result: analysis-forced one-step African t2m from 00Z IC\n"
        f"Student v5 · n={len(rows)} inits (2023) · not an autoregressive lead-time score. "
        "Error bars: ±1 std across inits.",
        fontsize=9,
    )
    fig.savefig(OUT2_PRIMARY, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT2_PRIMARY}")


def fig2_lead_curves(student, k1) -> None:
    """Backup: one-step AF skill by analysis IC hour (00Z campaign)."""
    by_lead = defaultdict(list)
    for r in student:
        t = r["variables"]["t2m"]
        by_lead[r["lead_hours"]].append(
            (float(t["student_rmse_vs_era5"]), float(t["student_acc"]), float(t["student_bias"]))
        )
    leads = sorted(by_lead)
    ticks = [ic_tick(L) for L in leads]
    rmse = np.array([np.mean([x[0] for x in by_lead[L]]) for L in leads])
    rmse_std = np.array([np.std([x[0] for x in by_lead[L]], ddof=1) for L in leads])
    acc = np.array([np.mean([x[1] for x in by_lead[L]]) for L in leads])
    acc_std = np.array([np.std([x[1] for x in by_lead[L]], ddof=1) for L in leads])

    k1_by = defaultdict(list)
    for r in k1:
        t = r["variables"]["t2m"]
        k1_by[r["lead_hours"]].append((float(t["rmse"]), float(t["acc"])))
    k1_leads = sorted(k1_by)
    k1_rmse = [np.mean([x[0] for x in k1_by[L]]) for L in k1_leads]
    k1_acc = [np.mean([x[1] for x in k1_by[L]]) for L in k1_leads]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

    ax = axes[0]
    ax.errorbar(
        leads,
        rmse,
        yerr=rmse_std,
        fmt="-o",
        color="#1f4e79",
        ecolor="#8faadc",
        capsize=3,
        label=f"Student v5 (n={len(by_lead[leads[0]])} inits)",
    )
    if k1_leads:
        ax.plot(k1_leads, k1_rmse, "s--", color="#c45911", label=f"K1 teacher (n={len(k1_by[k1_leads[0]])})")
    ax.set_xlabel("Analysis IC hour (00Z campaign; one-step AF)")
    ax.set_ylabel("RMSE (°C)")
    ax.set_title("African t2m RMSE by IC hour")
    ax.set_xticks(leads)
    ax.set_xticklabels(ticks)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.errorbar(
        leads,
        acc,
        yerr=acc_std,
        fmt="-o",
        color="#1f4e79",
        ecolor="#8faadc",
        capsize=3,
        label="Student v5",
    )
    if k1_leads:
        ax.plot(k1_leads, k1_acc, "s--", color="#c45911", label="K1 teacher")
    ax.set_xlabel("Analysis IC hour (00Z campaign; one-step AF)")
    ax.set_ylabel("ACC")
    ax.set_title("African t2m ACC by IC hour")
    ax.set_xticks(leads)
    ax.set_xticklabels(ticks)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Fig. 2 (backup) — One-step AF skill by analysis IC hour (not autoregressive lead time)\n"
        "Each point: ERA5 IC at that hour → one +6 h step. 00Z = headline; 06Z column = cold-bias sensitivity. "
        "Error bars: ±1 std across inits.",
        fontsize=9,
        y=1.08,
    )
    fig.savefig(OUT2, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT2}")


def fig3_heatmap(student) -> None:
    # One row per init, columns = analysis IC hour
    inits = sorted({r["init_date"] for r in student})
    leads = sorted({r["lead_hours"] for r in student})
    season_of = {}
    for r in student:
        season_of[r["init_date"]] = r["season"]
    inits = sorted(inits, key=lambda d: (SEASON_ORDER.get(season_of[d], 9), d))

    mat = np.full((len(inits), len(leads)), np.nan)
    lookup = {(r["init_date"], r["lead_hours"]): float(r["variables"]["t2m"]["student_rmse_vs_era5"]) for r in student}
    for i, init in enumerate(inits):
        for j, lead in enumerate(leads):
            mat[i, j] = lookup.get((init, lead), np.nan)

    fig_h = max(6.5, 0.14 * len(inits))
    fig, ax = plt.subplots(figsize=(7.2, fig_h), constrained_layout=True)
    vmax = float(np.nanpercentile(mat, 95))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=0.0, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(len(leads)))
    ax.set_xticklabels([ic_tick(L) for L in leads])
    ylabels = []
    for init in inits:
        ylabels.append(f"{init[4:6]}-{init[6:]} {season_of[init]}")
    step = max(1, len(inits) // 20)
    ax.set_yticks(range(0, len(inits), step))
    ax.set_yticklabels([ylabels[i] for i in range(0, len(inits), step)], fontsize=7)
    ax.set_xlabel("Analysis IC hour (one-step AF)")
    ax.set_ylabel("Initialisation date (MM-DD, season)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("RMSE (°C)")

    seasons = [season_of[i] for i in inits]
    for i in range(1, len(seasons)):
        if seasons[i] != seasons[i - 1]:
            ax.axhline(i - 0.5, color="black", lw=0.8, alpha=0.7)

    ax.set_title(
        "Fig. 3 — Init × analysis-IC-hour African t2m RMSE (student v5, n=61; one-step AF)\n"
        "Bright 06Z column = systematic cold-bias sensitivity (not AR +12 h lead skill).",
        fontsize=9,
    )
    fig.savefig(OUT3, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT3}")


def fig6_plus12_pathology(student) -> None:
    biases = []
    rmses = []
    seasons = []
    for r in student:
        if r["lead_hours"] != 12:
            continue
        t = r["variables"]["t2m"]
        biases.append(float(t["student_bias"]))
        rmses.append(float(t["student_rmse_vs_era5"]))
        seasons.append(r["season"])
    biases = np.asarray(biases)
    rmses = np.asarray(rmses)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)

    ax = axes[0]
    ax.hist(biases, bins=12, color="#1f4e79", edgecolor="white", alpha=0.9)
    ax.axvline(biases.mean(), color="#c45911", ls="--", lw=1.5, label=f"mean={biases.mean():.2f} °C")
    ax.set_xlabel("Bias (°C)  (student − ERA5)")
    ax.set_ylabel("Count of inits")
    ax.set_title("06Z-IC one-step bias (n=61)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    order = ["DJF", "MAM", "JJA", "SON"]
    data = [[biases[i] for i, s in enumerate(seasons) if s == sea] for sea in order]
    bp = ax.boxplot(data, labels=order, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#8faadc")
        patch.set_alpha(0.85)
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.set_ylabel("Bias (°C)")
    ax.set_title("06Z-IC bias by season")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Fig. 6 — 06Z analysis-IC one-step cold-bias pathology (student v5)\n"
        f"All {len(biases)} inits are cold (bias ∈ [{biases.min():.2f}, {biases.max():.2f}] K); "
        f"mean RMSE={rmses.mean():.2f} °C. Time-of-day / IC-hour sensitivity under AF — not AR lead error.",
        fontsize=9,
        y=1.06,
    )
    fig.savefig(OUT6, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT6}")


def fig7_seasonal(student) -> None:
    """+6 h / +24 h RMSE and ACC by season (means ±1 std across inits)."""
    seasons = ["DJF", "MAM", "JJA", "SON"]
    leads = [6, 24]
    metrics = {
        (sea, L): {"rmse": [], "acc": [], "bias": []}
        for sea in seasons
        for L in leads
    }
    for r in student:
        L = r["lead_hours"]
        if L not in leads:
            continue
        sea = r["season"]
        if (sea, L) not in metrics:
            continue
        t = r["variables"]["t2m"]
        metrics[(sea, L)]["rmse"].append(float(t["student_rmse_vs_era5"]))
        metrics[(sea, L)]["acc"].append(float(t["student_acc"]))
        metrics[(sea, L)]["bias"].append(float(t["student_bias"]))

    def _mean_std(vals: list[float]) -> tuple[float, float]:
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0:
            return float("nan"), float("nan")
        if arr.size == 1:
            return float(arr[0]), 0.0
        return float(arr.mean()), float(arr.std(ddof=1))

    x = np.arange(len(seasons))
    width = 0.35
    c6, c24 = "#1f4e79", "#c45911"

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

    # RMSE panel
    ax = axes[0]
    m6, s6 = zip(*[_mean_std(metrics[(s, 6)]["rmse"]) for s in seasons])
    m24, s24 = zip(*[_mean_std(metrics[(s, 24)]["rmse"]) for s in seasons])
    n6 = [len(metrics[(s, 6)]["rmse"]) for s in seasons]
    bars1 = ax.bar(x - width / 2, m6, width, yerr=s6, capsize=3, color=c6, label="00Z IC", ecolor="#8faadc")
    bars2 = ax.bar(x + width / 2, m24, width, yerr=s24, capsize=3, color=c24, label="18Z IC", ecolor="#f4b183")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(seasons, n6)])
    ax.set_ylabel("RMSE (°C)")
    ax.set_title("African t2m RMSE by season (one-step AF)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.bar_label(bars1, fmt="%.2f", padding=2, fontsize=7)
    ax.bar_label(bars2, fmt="%.2f", padding=2, fontsize=7)

    # ACC panel
    ax = axes[1]
    a6, as6 = zip(*[_mean_std(metrics[(s, 6)]["acc"]) for s in seasons])
    a24, as24 = zip(*[_mean_std(metrics[(s, 24)]["acc"]) for s in seasons])
    ax.bar(x - width / 2, a6, width, yerr=as6, capsize=3, color=c6, label="00Z IC", ecolor="#8faadc")
    ax.bar(x + width / 2, a24, width, yerr=as24, capsize=3, color=c24, label="18Z IC", ecolor="#f4b183")
    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.set_ylabel("ACC")
    ax.set_ylim(0.6, 1.02)
    ax.set_title("African t2m ACC by season (one-step AF)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    for i, (v6, v24) in enumerate(zip(a6, a24)):
        ax.text(i - width / 2, v6 + 0.01, f"{v6:.3f}", ha="center", fontsize=7)
        ax.text(i + width / 2, v24 + 0.01, f"{v24:.3f}", ha="center", fontsize=7)

    growth = [100.0 * (m24[i] / m6[i] - 1.0) for i in range(len(seasons))]
    growth_txt = ", ".join(f"{s} +{g:.0f}%" for s, g in zip(seasons, growth))
    fig.suptitle(
        "Fig. 7 — Seasonal one-step AF skill: 00Z IC vs 18Z IC (student v5)\n"
        f"00Z IC stays strong in all seasons; 18Z IC RMSE growth vs 00Z: {growth_txt}. "
        "Not AR +6/+24 h lead skill. Error bars: ±1 std across inits.",
        fontsize=9,
        y=1.08,
    )
    fig.savefig(OUT7, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT7}")


def fig8_baselines() -> None:
    """Fig. 8 — student skill relative to AF persistence (RMSE + skill %).

    Main panel: AF persistence, Mvula v5, K1 (and climatology if present).
    Side panel: relative skill vs AF persistence (%).
    """
    if not BASELINES_JSON.is_file():
        print(f"skip Fig.8 baselines — missing {BASELINES_JSON}")
        return
    blob = json.loads(BASELINES_JSON.read_text(encoding="utf-8"))
    agg = blob["aggregate"]
    skill = blob["skill_vs_persistence_af"]
    leads = sorted(int(L) for L in agg["student"])

    def series(model: str, key: str = "rmse_mean") -> list[float]:
        return [float(agg[model][str(L)][key]) for L in leads]

    def series_std(model: str) -> list[float]:
        return [float(agg[model][str(L)].get("rmse_std", 0.0)) for L in leads]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)

    ax = axes[0]
    # Core three-line story; init-persistence kept in JSON but not plotted (dilutes message).
    ax.errorbar(
        leads,
        series("persistence_af"),
        yerr=series_std("persistence_af"),
        fmt=":^",
        color="#7f7f7f",
        capsize=3,
        lw=1.8,
        label="AF persistence",
    )
    ax.errorbar(
        leads,
        series("student"),
        yerr=series_std("student"),
        fmt="-o",
        color="#1f4e79",
        capsize=3,
        lw=2.0,
        label="Mvula v5",
    )
    if "k1" in agg and agg["k1"]:
        k1_leads = sorted(int(L) for L in agg["k1"])
        ax.plot(
            k1_leads,
            [float(agg["k1"][str(L)]["rmse_mean"]) for L in k1_leads],
            "s--",
            color="#c45911",
            lw=1.6,
            label=f"K1 (n={agg['k1'][str(k1_leads[0])]['n']})",
        )
    if "climatology" in agg and agg["climatology"]:
        ax.plot(
            leads,
            series("climatology"),
            "D-.",
            color="#548235",
            lw=1.4,
            label="Climatology (MM–DD HH)",
        )
    ax.set_xlabel("Analysis IC hour (one-step AF)")
    ax.set_ylabel("RMSE (°C)")
    ax.set_xticks(leads)
    ax.set_xticklabels([ic_tick(L) for L in leads])
    ax.set_title("RMSE by IC hour: persistence → Mvula → K1")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    skills = [100.0 * float(skill[str(L)]["student_skill_vs_persistence_af"]) for L in leads]
    colors = ["#1f4e79" if s > 0 else "#c00000" for s in skills]
    bars = ax.bar([ic_tick(L) for L in leads], skills, color=colors, width=0.55, zorder=2)
    ax.axhline(0.0, color="black", lw=0.8, zorder=1)
    ax.set_xlabel("Analysis IC hour (one-step AF)")
    ax.set_ylabel("Skill vs AF persistence (%)")
    ax.set_title("Relative skill  1 − RMSE$_\\mathrm{stu}$ / RMSE$_\\mathrm{pers}$")
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.bar_label(bars, fmt="%+.1f%%", padding=3, fontsize=8, zorder=3)

    # Regime meaning in legend (avoids labels colliding with bars)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="#1f4e79", edgecolor="none", label="Beats persistence (useful)"),
            Patch(facecolor="#c00000", edgecolor="none", label="Worse than persistence (IC-hour failure)"),
        ],
        frameon=True,
        fancybox=False,
        fontsize=7.5,
        loc="upper right",
        framealpha=0.95,
    )
    y_pad = max(8.0, 0.12 * (max(skills) - min(skills)))
    ax.set_ylim(min(skills) - y_pad, max(skills) + y_pad)

    fig.suptitle(
        "Fig. 8 — One-step AF skill vs persistence by analysis IC hour (not AR lead time)\n"
        "AF persistence: T̂(valid)=T(IC). Positive % ⇒ useful state information beyond copying the analysis. "
        "Headline claim uses 00Z only.",
        fontsize=9,
        y=1.08,
    )
    fig.savefig(OUT8, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT8}")


def fig9_compute_panel() -> None:
    bench = json.loads(BENCH_JSON.read_text(encoding="utf-8"))
    student_mib = float(bench["student_ckpt_mib"])
    teacher_mib = float(bench["teacher_ckpt_mib"])
    step_s = float(bench["mean_step_seconds"])
    rss = float(bench["rss_peak_mib"])
    params_m = float(bench["student_params_m"])
    cpu = (bench.get("hardware") or {}).get("cpu_model") or "CPU"

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), constrained_layout=True)

    ax = axes[0]
    bars = ax.bar(["Student v5", "K1 teacher"], [student_mib, teacher_mib], color=["#1f4e79", "#c45911"])
    ax.set_ylabel("Disk size (MiB)")
    ax.set_title(f"Checkpoint size (~{teacher_mib/student_mib:.0f}× shrink)")
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(["+6 h step"], [step_s], color="#1f4e79", width=0.45)
    ax.set_ylabel("Seconds")
    ax.set_title("CPU inference (4 threads)")
    ax.set_ylim(0, max(3.5, step_s * 1.4))
    ax.text(0, step_s + 0.08, f"{step_s:.2f} s", ha="center", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[2]
    ax.bar(["Peak RSS"], [rss / 1024.0], color="#548235", width=0.45)
    ax.set_ylabel("GiB")
    ax.set_title("Process memory")
    ax.set_ylim(0, max(2.0, rss / 1024.0 * 1.4))
    ax.text(0, rss / 1024.0 + 0.05, f"{rss/1024:.2f} GiB", ha="center", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Fig. 9 — Laptop compute / accessibility panel (Mvula student v5)\n"
        f"{cpu}; {params_m:.2f} M params; GPU not required; IC fetch/build excluded from timing.",
        fontsize=9,
        y=1.08,
    )
    fig.savefig(OUT9, dpi=160, bbox_inches="tight")
    plt.close(fig)
    try:
        import shutil

        shutil.copyfile(OUT9, OUT8_COMPUTE_LEGACY)
    except OSError:
        pass
    print(f"wrote {OUT9}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig1_pipeline()
    fig1_pipeline_publication()
    _blob, student, k1 = load_rows()
    fig2_primary_00z(student, k1)
    fig2_lead_curves(student, k1)
    fig3_heatmap(student)
    fig6_plus12_pathology(student)
    fig7_seasonal(student)
    fig8_baselines()
    fig9_compute_panel()


if __name__ == "__main__":
    main()

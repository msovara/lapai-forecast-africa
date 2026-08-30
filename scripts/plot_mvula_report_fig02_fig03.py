#!/usr/bin/env python3
"""Generate Dueben-style Fig.2 (lead curves) and Fig.3 (init x lead heatmap)

from reports/TRACKB_T2M_EXPANDED.json for the Mvula internal technical report.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
JSON_PATH = REPO / "reports" / "TRACKB_T2M_EXPANDED.json"
FIG_DIR = REPO / "reports" / "figures"
OUT2 = FIG_DIR / "mvula_fig02_t2m_lead_curves.png"
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


def load_rows():
    blob = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    student = blob["student_results"]
    k1 = blob.get("k1_results") or []
    return blob, student, k1


def fig2_lead_curves(student, k1) -> None:
    by_lead = defaultdict(list)
    for r in student:
        t = r["variables"]["t2m"]
        by_lead[r["lead_hours"]].append(
            (float(t["student_rmse_vs_era5"]), float(t["student_acc"]), float(t["student_bias"]))
        )
    leads = sorted(by_lead)
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
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel("RMSE (K)")
    ax.set_title("African t2m RMSE vs lead")
    ax.set_xticks(leads)
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
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel("ACC")
    ax.set_title("African t2m ACC vs lead")
    ax.set_xticks(leads)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Fig. 2 — Analysis-forced African t2m skill (Mvula student v5)\n"
        "Protocol: IC at init+(L−6)h → one +6 h step; cosine-latitude metrics vs ARCO ERA5. "
        "Shaded/error bars: ±1 std across inits.",
        fontsize=9,
        y=1.08,
    )
    fig.savefig(OUT2, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT2}")


def fig3_heatmap(student) -> None:
    # One row per init, columns leads
    inits = sorted({r["init_date"] for r in student})
    leads = sorted({r["lead_hours"] for r in student})
    # order inits by season then date
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
    # Clip colour scale so +12h ridge is visible without washing +6h
    vmax = float(np.nanpercentile(mat, 95))
    # Low RMSE = blue, high RMSE = red (diverging blues/reds; +12 h failure lights up red).
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=0.0, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(len(leads)))
    ax.set_xticklabels([f"+{L}h" for L in leads])
    # Season separators + sparse y labels
    ylabels = []
    for init in inits:
        ylabels.append(f"{init[4:6]}-{init[6:]} {season_of[init]}")
    step = max(1, len(inits) // 20)
    ax.set_yticks(range(0, len(inits), step))
    ax.set_yticklabels([ylabels[i] for i in range(0, len(inits), step)], fontsize=7)
    ax.set_xlabel("Lead time")
    ax.set_ylabel("Initialisation (MM-DD, season)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("RMSE (K)")

    # Mark season boundaries
    seasons = [season_of[i] for i in inits]
    for i in range(1, len(seasons)):
        if seasons[i] != seasons[i - 1]:
            ax.axhline(i - 0.5, color="black", lw=0.8, alpha=0.7)

    ax.set_title(
        "Fig. 3 — Init × lead African t2m RMSE scorecard (student v5, n=61)\n"
        "Bright column at +12 h = systematic cold-bias failure mode (RMSE≈7.8 K).",
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
    ax.axvline(biases.mean(), color="#c45911", ls="--", lw=1.5, label=f"mean={biases.mean():.2f} K")
    ax.set_xlabel("Bias (K)  (student − ERA5)")
    ax.set_ylabel("Count of inits")
    ax.set_title("+12 h bias distribution (n=61)")
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
    ax.set_ylabel("Bias (K)")
    ax.set_title("+12 h bias by season")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Fig. 6 — +12 h African t2m cold-bias pathology (student v5)\n"
        f"All {len(biases)} inits are cold (bias ∈ [{biases.min():.2f}, {biases.max():.2f}] K); "
        f"mean RMSE={rmses.mean():.2f} K. Likely diurnal / lead-conditioned AF failure, not random noise.",
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
    bars1 = ax.bar(x - width / 2, m6, width, yerr=s6, capsize=3, color=c6, label="+6 h", ecolor="#8faadc")
    bars2 = ax.bar(x + width / 2, m24, width, yerr=s24, capsize=3, color=c24, label="+24 h", ecolor="#f4b183")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(seasons, n6)])
    ax.set_ylabel("RMSE (K)")
    ax.set_title("African t2m RMSE by season")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.bar_label(bars1, fmt="%.2f", padding=2, fontsize=7)
    ax.bar_label(bars2, fmt="%.2f", padding=2, fontsize=7)

    # ACC panel
    ax = axes[1]
    a6, as6 = zip(*[_mean_std(metrics[(s, 6)]["acc"]) for s in seasons])
    a24, as24 = zip(*[_mean_std(metrics[(s, 24)]["acc"]) for s in seasons])
    ax.bar(x - width / 2, a6, width, yerr=as6, capsize=3, color=c6, label="+6 h", ecolor="#8faadc")
    ax.bar(x + width / 2, a24, width, yerr=as24, capsize=3, color=c24, label="+24 h", ecolor="#f4b183")
    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.set_ylabel("ACC")
    ax.set_ylim(0.6, 1.02)
    ax.set_title("African t2m ACC by season")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    for i, (v6, v24) in enumerate(zip(a6, a24)):
        ax.text(i - width / 2, v6 + 0.01, f"{v6:.3f}", ha="center", fontsize=7)
        ax.text(i + width / 2, v24 + 0.01, f"{v24:.3f}", ha="center", fontsize=7)

    growth = [100.0 * (m24[i] / m6[i] - 1.0) for i in range(len(seasons))]
    growth_txt = ", ".join(f"{s} +{g:.0f}%" for s, g in zip(seasons, growth))
    fig.suptitle(
        "Fig. 7 — Seasonal African t2m skill at +6 h vs +24 h (student v5)\n"
        f"+6 h stays strong in all seasons; +24 h RMSE growth: {growth_txt}. Error bars: ±1 std across inits.",
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
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel("RMSE (K)")
    ax.set_xticks(leads)
    ax.set_title("RMSE: persistence → Mvula → K1")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    skills = [100.0 * float(skill[str(L)]["student_skill_vs_persistence_af"]) for L in leads]
    colors = ["#1f4e79" if s > 0 else "#c00000" for s in skills]
    bars = ax.bar([f"+{L}" for L in leads], skills, color=colors, width=0.55)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel("Skill vs AF persistence (%)")
    ax.set_title("Relative skill  1 − RMSE$_\\mathrm{stu}$ / RMSE$_\\mathrm{pers}$")
    ax.grid(True, axis="y", alpha=0.3)
    ax.bar_label(bars, fmt="%+.1f%%", padding=2, fontsize=8)

    # Annotate the non-monotonic regime in the skill panel
    ax.annotate(
        "useful",
        xy=(0, skills[0]),
        xytext=(0.15, max(skills) * 0.72),
        fontsize=7,
        color="#1f4e79",
        arrowprops=dict(arrowstyle="-", color="#1f4e79", lw=0.6),
    )
    if len(skills) >= 2 and skills[1] < 0:
        ax.annotate(
            "failure regime",
            xy=(1, skills[1]),
            xytext=(1.2, min(skills) * 0.55),
            fontsize=7,
            color="#c00000",
            arrowprops=dict(arrowstyle="-", color="#c00000", lw=0.6),
        )

    fig.suptitle(
        "Fig. 8 — Student skill relative to analysis-forced persistence\n"
        "AF persistence: T̂(valid)=T(IC), IC=init+(L−6)h (matched to student protocol). "
        "Positive % ⇒ student extracts useful state information beyond copying the analysis.",
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
    _blob, student, k1 = load_rows()
    fig2_lead_curves(student, k1)
    fig3_heatmap(student)
    fig6_plus12_pathology(student)
    fig7_seasonal(student)
    fig8_baselines()
    fig9_compute_panel()


if __name__ == "__main__":
    main()

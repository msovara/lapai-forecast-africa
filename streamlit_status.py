#!/usr/bin/env python3
"""LapAI-Forecast — interactive project status dashboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO = Path(__file__).resolve().parent
REPORTS = REPO / "reports"
LOGO = REPO / "docs" / "branding" / "mvula_banner.jpg"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

st.set_page_config(
    page_title="Mvula / LapAI Status",
    page_icon=str(LOGO) if LOGO.is_file() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block { padding: 0.75rem 1rem; border-radius: 0.4rem; margin-bottom: 0.5rem; }
.done { background: #e8f5e9; border-left: 4px solid #2e7d32; }
.warn { background: #fff8e1; border-left: 4px solid #f9a825; }
.info { background: #e3f2fd; border-left: 4px solid #1565c0; }
.muted { color: #666; font-size: 0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)


def _load_json(name: str) -> dict | None:
    path = REPORTS / name
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _gate_summary(gate: dict | None) -> tuple[int, int, bool | None]:
    if not gate:
        return 0, 0, None
    checks = gate.get("checks") or []
    passed = sum(1 for c in checks if c.get("passed"))
    return passed, len(checks), bool(gate.get("passed"))


def _t2m_24h_table(track_a: dict | None, phase0: dict | None) -> pd.DataFrame | None:
    if not track_a or not phase0:
        return None

    def rows(sc: dict, label: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for r in sc.get("results") or []:
            if r.get("lead_hours") != 24:
                continue
            init = r.get("init_date")
            t2m = (r.get("variables") or {}).get("t2m") or {}
            rmse = t2m.get("rmse")
            if init and rmse is not None:
                out[str(init)] = float(rmse)
        return out

    cand = rows(track_a, "coarsened")
    base = rows(phase0, "teacher")
    if not cand or not base:
        return None
    inits = sorted(set(cand) & set(base))
    data = []
    for init in inits:
        b, c = base[init], cand[init]
        data.append(
            {
                "Init": init,
                "Teacher RMSE (K)": round(b, 2),
                "Coarsened RMSE (K)": round(c, 2),
                "Degradation %": round(100 * (c - b) / b, 1),
            }
        )
    return pd.DataFrame(data)


def _scorecard_long(sc: dict | None, model: str) -> pd.DataFrame:
    if not sc:
        return pd.DataFrame()
    rows: list[dict] = []
    for r in sc.get("results") or []:
        init = r.get("init_date")
        lead = r.get("lead_hours")
        for var, metrics in (r.get("variables") or {}).items():
            rmse = (metrics or {}).get("rmse")
            acc = (metrics or {}).get("acc")
            if rmse is None and acc is None:
                continue
            rows.append(
                {
                    "model": model,
                    "init": str(init),
                    "lead_h": int(lead),
                    "variable": str(var),
                    "rmse": float(rmse) if rmse is not None else None,
                    "acc": float(acc) if acc is not None else None,
                }
            )
    return pd.DataFrame(rows)


def _plotly_template() -> str:
    return "plotly_dark" if _is_dark_theme() else "plotly_white"


def _layout(fig: go.Figure, *, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        template=_plotly_template(),
        title=title,
        height=height,
        margin=dict(t=50, b=40, l=50, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def _lead_h_to_ic_label(lead_h) -> str:
    """AF 00Z-campaign packaging: lead_hours columns map to analysis IC hours."""
    try:
        L = int(lead_h)
    except (TypeError, ValueError):
        return str(lead_h)
    return {6: "00Z", 12: "06Z", 18: "12Z", 24: "18Z"}.get(L, f"+{L}h")


def _rmse_vs_lead_figure(base: pd.DataFrame, cand: pd.DataFrame, variable: str) -> go.Figure:
    fig = go.Figure()
    colors = {"Teacher (N320)": "#64b5f6", "Coarsened O96": "#ffb74d"}
    for model, df, label in (
        ("teacher", base, "Teacher (N320)"),
        ("coarsened", cand, "Coarsened O96"),
    ):
        sub = df[(df["variable"] == variable) & df["rmse"].notna()]
        if sub.empty:
            continue
        agg = sub.groupby("lead_h", as_index=False)["rmse"].mean()
        fig.add_trace(
            go.Scatter(
                x=agg["lead_h"],
                y=agg["rmse"],
                mode="lines+markers",
                name=label,
                line=dict(color=colors[label], width=2.5),
                marker=dict(size=8),
            )
        )
    fig.update_xaxes(title="Lead time (h)", dtick=24)
    fig.update_yaxes(title="RMSE (model units)")
    return _layout(fig, title=f"{variable} — RMSE vs lead (mean over 5 inits · Africa)")


def _acc_vs_lead_figure(base: pd.DataFrame, cand: pd.DataFrame, variable: str) -> go.Figure:
    fig = go.Figure()
    colors = {"Teacher (N320)": "#64b5f6", "Coarsened O96": "#ffb74d"}
    for df, label in ((base, "Teacher (N320)"), (cand, "Coarsened O96")):
        sub = df[(df["variable"] == variable) & df["acc"].notna()]
        if sub.empty:
            continue
        agg = sub.groupby("lead_h", as_index=False)["acc"].mean()
        fig.add_trace(
            go.Scatter(
                x=agg["lead_h"],
                y=agg["acc"],
                mode="lines+markers",
                name=label,
                line=dict(color=colors[label], width=2.5),
                marker=dict(size=8),
            )
        )
    fig.update_xaxes(title="Lead time (h)", dtick=24)
    fig.update_yaxes(title="ACC")
    return _layout(fig, title=f"{variable} — ACC vs lead (mean over 5 inits · Africa)")


def _rmse_by_init_figure(base: pd.DataFrame, cand: pd.DataFrame, *, lead_h: int, variable: str) -> go.Figure:
    fig = go.Figure()
    inits = sorted(set(base["init"].unique()) & set(cand["init"].unique()))
    b = base[(base["variable"] == variable) & (base["lead_h"] == lead_h)].set_index("init")["rmse"]
    c = cand[(cand["variable"] == variable) & (cand["lead_h"] == lead_h)].set_index("init")["rmse"]
    fig.add_trace(go.Bar(name="Teacher (N320)", x=inits, y=[b.get(i) for i in inits], marker_color="#64b5f6"))
    fig.add_trace(go.Bar(name="Coarsened O96", x=inits, y=[c.get(i) for i in inits], marker_color="#ffb74d"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Init date")
    fig.update_yaxes(title="RMSE")
    return _layout(fig, title=f"{variable} RMSE by init at +{lead_h} h", height=400)


def _gate_degradation_heatmap(gdf: pd.DataFrame, lead_h: int) -> go.Figure:
    sub = gdf[gdf["Lead (h)"] == lead_h].copy()
    if sub.empty:
        return go.Figure()
    pivot = sub.pivot(index="Init", columns="Var", values="Degradation %")
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="RdYlGn_r",
            zmid=5.0,
            text=[[f"{v:.0f}%" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            colorbar=dict(title="Degradation %"),
        )
    )
    fig.update_xaxes(title="Variable")
    fig.update_yaxes(title="Init date")
    return _layout(fig, title=f"A1 gate RMSE degradation at +{lead_h} h (limit 5%)", height=360)


def _gate_pass_summary(gdf: pd.DataFrame) -> go.Figure:
    summary = gdf.groupby("Var")["Passed"].agg(["sum", "count"]).reset_index()
    summary["failed"] = summary["count"] - summary["sum"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Passed", x=summary["Var"], y=summary["sum"], marker_color="#66bb6a"))
    fig.add_trace(go.Bar(name="Failed", x=summary["Var"], y=summary["failed"], marker_color="#ef5350"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Variable")
    fig.update_yaxes(title="Gate checks (inits × leads)")
    return _layout(fig, title="A1 gate pass/fail by variable (+24 h & +48 h)", height=360)


def _contour_figure(
    z: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    title: str,
    colorscale: str = "Viridis",
    zmin: float | None = None,
    zmax: float | None = None,
    zmid: float | None = None,
    height: int = 320,
) -> go.Figure:
    fig = go.Figure(
        go.Contour(
            x=lon,
            y=lat,
            z=z,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            zmid=zmid,
            contours=dict(coloring="heatmap"),
            colorbar=dict(len=0.75),
        )
    )
    fig.update_xaxes(title="Longitude (°E)")
    fig.update_yaxes(title="Latitude (°N)", scaleanchor="x", scaleratio=1)
    return _layout(fig, title=title, height=height)


def _field_colorscale(variable: str) -> str:
    return "Plasma" if variable == "t2m" else "RdBu_r"


@st.cache_data(show_spinner="Loading forecast fields…")
def _fetch_spatial_fields(
    init: str,
    lead_h: int,
    variable: str,
    gcs_bucket: str,
    include_era5: bool,
) -> dict:
    from evaluation.spatial_maps import (
        PHASE0_FORECAST_DIR,
        TRACKA_FORECAST_DIR,
        load_era5_slice,
        load_forecast_slice,
    )

    teacher_path = REPO / PHASE0_FORECAST_DIR / f"{init}_00Z.nc"
    coarsened_path = REPO / TRACKA_FORECAST_DIR / f"{init}_00Z.nc"
    teacher, valid_time = load_forecast_slice(teacher_path, variable, lead_h)
    coarsened, _ = load_forecast_slice(coarsened_path, variable, lead_h)
    lat = np.asarray(teacher.latitude.values, dtype=np.float64)
    lon = np.asarray(teacher.longitude.values, dtype=np.float64)
    out: dict = {
        "valid_time": str(valid_time),
        "lat": lat,
        "lon": lon,
        "teacher": np.asarray(teacher.values, dtype=np.float64),
        "coarsened": np.asarray(coarsened.values, dtype=np.float64),
        "era5": None,
    }
    if include_era5:
        era5 = load_era5_slice(variable, valid_time, pred_template=teacher, gcs_bucket=gcs_bucket)
        out["era5"] = np.asarray(era5.values, dtype=np.float64)
    return out


def _render_spatial_maps_tab() -> None:
    import importlib

    import evaluation.spatial_maps as spatial_maps

    # Streamlit often keeps imported helpers cached; force-reload after code fixes.
    spatial_maps = importlib.reload(spatial_maps)
    from evaluation.spatial_maps import DEFAULT_INITS, DEFAULT_LEADS, VAR_LABELS, field_stats

    st.subheader("Spatial patterns — ERA5 vs AIFS teacher vs coarsened output")
    st.caption(
        "Africa eval grid (0.25°) · Phase 0 NetCDF = N320 AIFS teacher · "
        "Track A NetCDF = coarsened O96 smoke run · ERA5 from GCS (needs ADC login)"
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        init = st.selectbox("Init date", DEFAULT_INITS, index=0, key="spatial_init")
    with c2:
        lead_h = st.selectbox("Lead (h)", DEFAULT_LEADS, index=0, key="spatial_lead")
    with c3:
        variable = st.selectbox("Variable", ["t2m", "u10", "v10", "tp"], index=0, key="spatial_var")
    with c4:
        include_era5 = st.checkbox("Include ERA5 (GCS)", value=True, help="First GCS fetch can take 1–2 min")

    gcs_bucket = "gs://code4earth/era5"
    teacher_nc = REPO / "data/processed/phase0/forecasts" / f"{init}_00Z.nc"
    coarsened_nc = REPO / "data/processed/trackA/forecasts" / f"{init}_00Z.nc"
    if not teacher_nc.is_file() or not coarsened_nc.is_file():
        st.warning(
            f"Forecast NetCDF missing. Expected:\n- `{teacher_nc}`\n- `{coarsened_nc}`\n\n"
            "Copy from Lengau or re-run `run_phase0_forecast.py` / Track A forecast PBS."
        )
        return

    try:
        with st.spinner("Loading NetCDF forecasts" + (" and ERA5 from GCS…" if include_era5 else "…")):
            bundle = _fetch_spatial_fields(init, lead_h, variable, gcs_bucket, include_era5)
    except Exception as exc:
        st.error(f"Could not load fields: {exc}")
        if include_era5:
            st.info(
                "ERA5 maps need Google Cloud access (`gcloud auth application-default login`) "
                "or uncheck **Include ERA5** to compare teacher vs coarsened only."
            )
        return

    label, units = VAR_LABELS.get(variable, (variable, ""))
    lat, lon = bundle["lat"], bundle["lon"]
    era5 = bundle["era5"]
    teacher = bundle["teacher"]
    coarsened = bundle["coarsened"]
    valid_time = bundle["valid_time"]

    if not np.isfinite(coarsened).any() and variable == "tp":
        st.warning("`tp` is all-NaN in the coarsened forecast NetCDF — maps show teacher only where noted.")

    st.markdown(f"**{label}** · valid time `{valid_time}` · init `{init}` · +{lead_h} h")

    cs = _field_colorscale(variable)

    if era5 is not None:
        finite = [a for a in (era5, teacher, coarsened) if np.isfinite(a).any()]
        if not finite:
            st.error("No finite data to plot.")
            return
        zmin = float(min(np.nanmin(a) for a in finite))
        zmax = float(max(np.nanmax(a) for a in finite))

        st.markdown("**Fields** (shared colour scale)")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.plotly_chart(
                _contour_figure(era5, lat, lon, title=f"ERA5 truth ({units})", colorscale=cs, zmin=zmin, zmax=zmax),
                use_container_width=True,
            )
        with m2:
            st.plotly_chart(
                _contour_figure(
                    teacher, lat, lon, title=f"AIFS teacher N320 ({units})", colorscale=cs, zmin=zmin, zmax=zmax
                ),
                use_container_width=True,
            )
        with m3:
            st.plotly_chart(
                _contour_figure(
                    coarsened, lat, lon, title=f"Coarsened O96 ({units})", colorscale=cs, zmin=zmin, zmax=zmax
                ),
                use_container_width=True,
            )

        st.divider()
        st.markdown("**Errors / compression loss** (diverging scale)")
        d_te = teacher - era5
        d_co = coarsened - era5
        d_tc = coarsened - teacher
        diffs = [d for d in (d_te, d_co, d_tc) if np.isfinite(d).any()]
        absmax = float(max(np.nanmax(np.abs(d)) for d in diffs)) if diffs else 1.0
        absmax = max(absmax, 1e-6)

        stats_te = field_stats(teacher, era5)
        stats_co = field_stats(coarsened, era5)
        stats_tc = field_stats(coarsened, teacher)

        e1, e2, e3 = st.columns(3)
        with e1:
            st.plotly_chart(
                _contour_figure(
                    d_te,
                    lat,
                    lon,
                    title=f"Teacher − ERA5 (RMSE {stats_te['rmse']:.2f})",
                    colorscale="RdBu_r",
                    zmin=-absmax,
                    zmax=absmax,
                    zmid=0,
                ),
                use_container_width=True,
            )
        with e2:
            st.plotly_chart(
                _contour_figure(
                    d_co,
                    lat,
                    lon,
                    title=f"Coarsened − ERA5 (RMSE {stats_co['rmse']:.2f})",
                    colorscale="RdBu_r",
                    zmin=-absmax,
                    zmax=absmax,
                    zmid=0,
                ),
                use_container_width=True,
            )
        with e3:
            st.plotly_chart(
                _contour_figure(
                    d_tc,
                    lat,
                    lon,
                    title=f"Coarsened − Teacher (RMSE {stats_tc['rmse']:.2f})",
                    colorscale="RdBu_r",
                    zmin=-absmax,
                    zmax=absmax,
                    zmid=0,
                ),
                use_container_width=True,
            )
    else:
        finite = [a for a in (teacher, coarsened) if np.isfinite(a).any()]
        if not finite:
            st.error("No finite data to plot.")
            return
        zmin = float(min(np.nanmin(a) for a in finite))
        zmax = float(max(np.nanmax(a) for a in finite))

        st.markdown("**Fields** (teacher vs coarsened — no ERA5)")
        m1, m2 = st.columns(2)
        with m1:
            st.plotly_chart(
                _contour_figure(
                    teacher, lat, lon, title=f"AIFS teacher N320 ({units})", colorscale=cs, zmin=zmin, zmax=zmax
                ),
                use_container_width=True,
            )
        with m2:
            st.plotly_chart(
                _contour_figure(
                    coarsened, lat, lon, title=f"Coarsened O96 ({units})", colorscale=cs, zmin=zmin, zmax=zmax
                ),
                use_container_width=True,
            )
        d_tc = coarsened - teacher
        if np.isfinite(d_tc).any():
            absmax = max(float(np.nanmax(np.abs(d_tc))), 1e-6)
            stats_tc = field_stats(coarsened, teacher)
            st.plotly_chart(
                _contour_figure(
                    d_tc,
                    lat,
                    lon,
                    title=f"Coarsened − Teacher (RMSE {stats_tc['rmse']:.2f})",
                    colorscale="RdBu_r",
                    zmin=-absmax,
                    zmax=absmax,
                    zmid=0,
                ),
                use_container_width=True,
            )

    st.caption(
        "Africa 0.25° eval grid · RMSE on finite points (unweighted). "
        "Enable ERA5 for full three-way comparison."
    )


def _is_dark_theme() -> bool:
    try:
        return st.get_option("theme.base") == "dark"
    except Exception:
        return False


def _pipeline_graph(
    nodes: dict[str, tuple[str, str | None]],
    edges: list[tuple[str, str]],
) -> str:
    """Build a Graphviz digraph with theme-aware edge and label colours."""
    dark = _is_dark_theme()
    edge_color = "#7ec8ff" if dark else "#1565c0"
    font_color = "#f5f5f5" if dark else "#1a1a1a"
    default_fill = "#1a4480" if dark else "#e3f2fd"
    warn_fill = "#6b3030" if dark else "#ffebee"
    accent_fill = "#5a4a1a" if dark else "#fff8e1"

    lines = [
        "digraph {",
        "  rankdir=LR;",
        '  graph [bgcolor="transparent"];',
        f'  node [shape=box style="filled,rounded" fontname="Arial" fontsize=11 '
        f'fontcolor="{font_color}" color="{edge_color}"];',
        f'  edge [color="{edge_color}" penwidth=2.0 arrowsize=0.9 fontcolor="{font_color}"];',
    ]
    for nid, (label, kind) in nodes.items():
        if kind == "warn":
            fill = warn_fill
        elif kind == "accent":
            fill = accent_fill
        else:
            fill = default_fill
        safe = label.replace('"', '\\"')
        lines.append(f'  {nid} [label="{safe}" fillcolor="{fill}"];')
    for src, dst in edges:
        lines.append(f"  {src} -> {dst};")
    lines.append("}")
    return "\n".join(lines)


def _phase0_graph() -> str:
    return _pipeline_graph(
        nodes={
            "A": ("CDS ERA5 cache\\n0.25° GRIB", None),
            "B": ("IC regrid\\nearthkit → N320", None),
            "C": ("N320 teacher\\nC4E n320_gt6", None),
            "D": ("Eval NetCDF\\nAfrica 0.25°", None),
            "E": ("Scorecard\\nGCS · laptop", None),
        },
        edges=[("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")],
    )


def _tracka_graph() -> str:
    return _pipeline_graph(
        nodes={
            "Z": ("O96 ERA5 Zarr", "accent"),
            "T": ("Coarsen train\\n50-step smoke", "accent"),
            "K": ("O96 checkpoint", "accent"),
            "C": ("CDS ICs\\nN320 path", "accent"),
            "B": ("IC bridge\\nN320→O96", "accent"),
            "I": ("O96 rollout\\n5 inits", "accent"),
            "EV": ("Eval NetCDF", "accent"),
            "S": ("Scorecard", "accent"),
            "G": ("A1 gate", "warn"),
        },
        edges=[
            ("Z", "T"),
            ("T", "K"),
            ("K", "I"),
            ("C", "B"),
            ("B", "I"),
            ("I", "EV"),
            ("EV", "S"),
            ("S", "G"),
        ],
    )


TODOS = [
    ("Phase 0 closed — N320 teacher baseline + scorecard", True),
    ("Track A A1 coarsening gate passed; K1 prune accepted as teacher", True),
    ("Track B student v5 frozen (Cout=3: tp/msl/2t; Case A — no free-run)", True),
    ("Laptop CPU bench on consumer i7 (measured; ~9 MiB, ~2.5 s/step)", True),
    ("Mvula status matrix + state closure docs on GitHub", True),
    ("AF t2m production campaign (multi-season × IC hours 00/06/12/18Z)", True),
    ("Package TRACKB_T2M_EXPANDED + Dueben-style skill scorecard", True),
    ("FINAL_REPORT + README freeze + tag trackb-v5-c4e", True),
    ("Optional ONNX smoke (not required for v5 close-out)", False),
    ("Do not reopen free-run / Cout=65 / tp recovery before close-out", True),
]

# Hard-coded matrix fallback if markdown parse fails (matches MVULA_CODE4EARTH_STATUS_MATRIX.md).
MVULA_MATRIX_ROWS = [
    ("AIFS compression", "PARTIAL"),
    ("Grid coarsening", "DONE"),
    ("Attention-head pruning", "PARTIAL"),
    ("LoRA", "NOT SHOWN"),
    ("Quantization", "NOT SHOWN"),
    ("t2m retention", "DONE (expanded AF)"),
    ("Multi-lead evaluation", "DONE (AF IC-hour)"),
    ("African evaluation", "PARTIAL"),
    ("Laptop inference", "DONE"),
    ("Model size reduction", "DONE"),
    ("Inference speed-up", "PARTIAL"),
    ("10-day forecast", "NOT POSSIBLE"),
    ("Free-running v5", "NOT POSSIBLE"),
    ("tp prediction", "FAILED"),
    ("Open-source repository", "DONE"),
    ("Reproducibility", "PARTIAL"),
    ("Documentation", "DONE"),
    ("Community / local relevance", "PARTIAL"),
]


def _load_text(name: str) -> str | None:
    path = REPORTS / name
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _held_out_student_long(blob: dict | None) -> pd.DataFrame:
    """Flatten student_results from TRACKB_HELD_OUT_* JSON."""
    if not blob:
        return pd.DataFrame()
    rows: list[dict] = []
    for r in blob.get("student_results") or []:
        init = r.get("init_date")
        lead = r.get("lead_hours")
        for var, metrics in (r.get("variables") or {}).items():
            m = metrics or {}
            rows.append(
                {
                    "init": str(init),
                    "lead_h": int(lead) if lead is not None else None,
                    "variable": str(var),
                    "student_rmse": m.get("student_rmse_vs_era5"),
                    "teacher_rmse": m.get("teacher_rmse_vs_era5"),
                    "student_acc": m.get("student_acc"),
                    "teacher_acc": m.get("teacher_acc"),
                    "deg_pct": m.get("degradation_pct_vs_teacher"),
                    "pod_1mm": m.get("student_pod_1mm"),
                }
            )
    return pd.DataFrame(rows)


def _expanded_t2m_long(blob: dict | None) -> pd.DataFrame:
    """Flatten TRACKB_T2M_EXPANDED.json (aggregate + optional per-init rows)."""
    if not blob:
        return pd.DataFrame()
    rows: list[dict] = []

    by_lead = ((blob.get("aggregate") or {}).get("by_lead")) or {}
    for lead_s, m in by_lead.items():
        m = m or {}
        rows.append(
            {
                "init": "ALL",
                "season": "ALL",
                "lead_h": int(lead_s),
                "variable": "t2m",
                "student_rmse": m.get("student_rmse_mean"),
                "teacher_rmse": m.get("teacher_rmse_mean"),
                "student_acc": m.get("student_acc_mean"),
                "teacher_acc": m.get("teacher_acc_mean"),
                "deg_pct": m.get("degradation_pct_mean"),
                "bias": m.get("student_bias_mean"),
                "n": m.get("n_student"),
            }
        )

    by_season = ((blob.get("aggregate") or {}).get("by_season_lead")) or {}
    for season, leads in by_season.items():
        for lead_s, m in (leads or {}).items():
            m = m or {}
            rows.append(
                {
                    "init": "ALL",
                    "season": str(season),
                    "lead_h": int(lead_s),
                    "variable": "t2m",
                    "student_rmse": m.get("student_rmse_mean"),
                    "teacher_rmse": m.get("teacher_rmse_mean"),
                    "student_acc": m.get("student_acc_mean"),
                    "teacher_acc": m.get("teacher_acc_mean"),
                    "deg_pct": m.get("degradation_pct_mean"),
                    "bias": m.get("student_bias_mean"),
                    "n": m.get("n_student"),
                }
            )

    # Fallback: raw student_results list if aggregate missing
    if not rows:
        records = blob.get("results") or blob.get("student_results") or []
        for r in records:
            variables = r.get("variables") or {}
            m = variables.get("t2m") or variables.get("2t") or {}
            if not m and (r.get("variable") in (None, "t2m", "2t")):
                m = r
            if not m:
                continue
            rows.append(
                {
                    "init": str(r.get("init_date") or r.get("init")),
                    "season": r.get("season") or r.get("month"),
                    "lead_h": int(r.get("lead_hours") or r.get("lead_h") or 0),
                    "variable": "t2m",
                    "student_rmse": m.get("student_rmse_vs_era5") or m.get("rmse"),
                    "teacher_rmse": m.get("teacher_rmse_vs_era5") or m.get("teacher_rmse"),
                    "student_acc": m.get("student_acc") or m.get("acc"),
                    "teacher_acc": m.get("teacher_acc"),
                    "deg_pct": m.get("degradation_pct_vs_teacher") or m.get("deg_pct"),
                    "bias": m.get("bias") or m.get("student_bias"),
                }
            )
    return pd.DataFrame(rows)


def _skill_matrix_heatmap(df: pd.DataFrame, *, value_col: str, title: str) -> go.Figure:
    """Dueben-style skill matrix: lead × season (or init) heatmap."""
    if df.empty or value_col not in df.columns:
        return go.Figure()
    work = df.dropna(subset=[value_col, "lead_h"]).copy()
    if work.empty:
        return go.Figure()
    if "season" in work.columns and work["season"].notna().any():
        idx_col = "season"
    else:
        idx_col = "init"
    pivot = work.pivot_table(index=idx_col, columns="lead_h", values=value_col, aggfunc="mean")
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[_lead_h_to_ic_label(c) for c in pivot.columns],
            y=[str(i) for i in pivot.index],
            colorscale="RdYlGn_r" if "deg" in value_col or "rmse" in value_col.lower() else "RdYlGn",
            text=[[f"{v:.2f}" if pd.notna(v) else "" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            colorbar=dict(title=value_col),
        )
    )
    fig.update_xaxes(title="Analysis IC hour (one-step AF)")
    fig.update_yaxes(title=idx_col)
    return _layout(fig, title=title, height=380)


def _lead_skill_lines(df: pd.DataFrame, *, metric: str) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        return fig
    s_col = "student_rmse" if metric == "rmse" else "student_acc"
    t_col = "teacher_rmse" if metric == "rmse" else "teacher_acc"
    for col, label, color in (
        (s_col, "Student v5", "#ffb74d"),
        (t_col, "K1 teacher", "#64b5f6"),
    ):
        if col not in df.columns:
            continue
        sub = df.dropna(subset=[col, "lead_h"])
        if sub.empty:
            continue
        agg = sub.groupby("lead_h", as_index=False)[col].mean()
        fig.add_trace(
            go.Scatter(
                x=[_lead_h_to_ic_label(v) for v in agg["lead_h"]],
                y=agg[col],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2.5),
                marker=dict(size=8),
            )
        )
    fig.update_xaxes(title="Analysis IC hour (one-step AF; not AR lead)")
    fig.update_yaxes(title="RMSE (K)" if metric == "rmse" else "ACC")
    return _layout(fig, title=f"t2m {metric.upper()} by analysis IC hour (Africa · one-step AF)")


def _render_mvula_tab() -> None:
    st.subheader("Mvula Code for Earth — dual deliverables")
    st.markdown(
        """
**Close-out control doc:** `reports/FINAL_REPORT.md`

**Headline close-out (→ 23 Sep 2026):**
1. **Forecast skill** — analysis-forced **t2m** (frozen `student_global_stable_v5`)
2. **Laptop / deployment** — size + CPU inference demo

Do **not** claim 10-day free-run from v5 (Case A: Cin=65 → Cout=3, no decoder).
"""
    )
    matrix_md = _load_text("MVULA_CODE4EARTH_STATUS_MATRIX.md")
    st.markdown("#### Objective status matrix")
    st.dataframe(
        pd.DataFrame(MVULA_MATRIX_ROWS, columns=["Objective", "Status"]),
        use_container_width=True,
        hide_index=True,
    )
    if matrix_md:
        with st.expander("Full matrix markdown"):
            st.markdown(matrix_md)
    else:
        st.caption("reports/MVULA_CODE4EARTH_STATUS_MATRIX.md not found — showing built-in summary.")

    closure = _load_text("TRACKB_STATE_CLOSURE.md")
    st.markdown("#### Case A — state closure")
    if closure:
        st.success("Free-run **NOT POSSIBLE** for v5 · outputs `tp`, `msl`, `2t` only · AF / re-IC multi-lead OK")
        with st.expander("TRACKB_STATE_CLOSURE.md"):
            st.markdown(closure)
    else:
        st.warning("reports/TRACKB_STATE_CLOSURE.md missing")

    st.markdown("#### Original PLAN vs freeze")
    st.dataframe(
        pd.DataFrame(
            [
                ("10-day free-run laptop AIFS", "AF t2m skill + CPU size/speed demo"),
                ("Full-state student", "Cout=3 partial-state MVP"),
                ("Week-9 multi-var 24–240h ≤15%", "t2m @ 6–24h AF vs K1"),
                ("LoRA + ONNX product", "Docs + measured laptop CPU; ONNX optional"),
                ("tp in skill budget", "tp FAILED / out-of-scope"),
            ],
            columns=["Original PLAN", "New close-out"],
        ),
        use_container_width=True,
        hide_index=True,
    )


def _render_trackb_tab() -> None:
    st.subheader("Track B — frozen student v5")
    gate_b = _load_json("TRACKB_GATE_V5.json") or _load_json("TRACKB_GATE.json")
    held = (
        _load_json("TRACKB_T2M_EXPANDED.json")
        or _load_json("TRACKB_HELD_OUT_JAN2023_V5.json")
        or _load_json("TRACKB_HELD_OUT_JAN2023.json")
    )
    expanded = _load_json("TRACKB_T2M_EXPANDED.json")

    if gate_b:
        checks = gate_b.get("checks") or []
        pc = sum(1 for c in checks if c.get("passed"))
        tc = len(checks)
        ok = bool(gate_b.get("passed"))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cache MVP gate", "PASS" if ok else "FAIL (soft)", f"{pc}/{tc}")
        m2.metric("Student", "v5", "Cout=3 · AF only")
        m3.metric("Teacher", "K1", "pruned GT")
        m4.metric("tp", "Out of scope", "dry collapse")
        gdf = pd.DataFrame(
            [
                {
                    "Var": c.get("variable"),
                    "Lead (h)": c.get("lead_hours"),
                    "Passed": c.get("passed"),
                    "Deg % vs K1": round(float(c.get("degradation_pct", 0)), 1),
                    "Student RMSE": c.get("student_rmse_vs_era5"),
                    "Teacher RMSE": c.get("teacher_rmse_vs_era5"),
                }
                for c in checks
            ]
        )
        st.markdown("**On-cache MVP gate** (not held-out)")
        st.dataframe(gdf, use_container_width=True, hide_index=True)
    else:
        st.warning("TRACKB_GATE_V5.json not found")

    st.divider()
    st.markdown("#### Held-out / production AF t2m skill")
    st.caption(
        "Analysis-forced **one-step** only. Packaged columns are **analysis IC hours** "
        "(00Z / 06Z / 12Z / 18Z for a 00Z campaign) — not autoregressive lead times. "
        "Headline = **00Z**; pathology = **06Z** cold bias."
    )
    if expanded:
        st.success(
            f"Production expanded results loaded — "
            f"n_inits={expanded.get('n_inits', '?')}, "
            f"leads={expanded.get('student_leads_hours', [])}"
        )
        verdict = (expanded.get("verdict") or {}).get("summary") or ""
        if verdict:
            st.info(verdict)
        df = _expanded_t2m_long(expanded)
    elif held:
        st.info(
            "Showing Jan-2023 held-out v5 (AF 00Z / 18Z IC one-step). "
            "Multi-season `TRACKB_T2M_EXPANDED.json` not written yet."
        )
        df = _held_out_student_long(held)
    else:
        st.warning("No held-out / expanded Track B JSON under reports/")
        df = pd.DataFrame()

    if not df.empty:
        t2m = df[df["variable"].isin(["t2m", "2t"])].copy() if "variable" in df.columns else df.copy()
        if t2m.empty and "student_rmse" in df.columns:
            t2m = df.copy()
        # Overall lead curves (exclude per-season duplicates when present)
        overall = t2m[t2m["season"].isin(["ALL", None]) | t2m["season"].isna()].copy() if "season" in t2m.columns else t2m
        if overall.empty:
            overall = t2m
        seasonal = (
            t2m[t2m["season"].notna() & ~t2m["season"].isin(["ALL"])].copy()
            if "season" in t2m.columns
            else pd.DataFrame()
        )

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(_lead_skill_lines(overall, metric="rmse"), use_container_width=True)
        with c2:
            st.plotly_chart(_lead_skill_lines(overall, metric="acc"), use_container_width=True)

        heat_src = seasonal if not seasonal.empty else overall
        if "student_rmse" in heat_src.columns and heat_src["student_rmse"].notna().any():
            st.plotly_chart(
                _skill_matrix_heatmap(
                    heat_src,
                    value_col="student_rmse",
                    title="t2m RMSE skill matrix (season × analysis IC hour)",
                ),
                use_container_width=True,
            )
        if "deg_pct" in heat_src.columns and heat_src["deg_pct"].notna().any():
            st.plotly_chart(
                _skill_matrix_heatmap(
                    heat_src,
                    value_col="deg_pct",
                    title="t2m degradation % vs K1 (Dueben-style)",
                ),
                use_container_width=True,
            )

        metric_cols = [
            c
            for c in ("student_rmse", "teacher_rmse", "student_acc", "teacher_acc", "deg_pct", "bias")
            if c in overall.columns
        ]
        if "lead_h" in overall.columns and metric_cols:
            summary = overall.groupby("lead_h", as_index=False)[metric_cols].mean(numeric_only=True)
            summary.insert(0, "Analysis IC", summary["lead_h"].map(_lead_h_to_ic_label))
            summary = summary.drop(columns=["lead_h"])
            st.markdown("**IC-hour skill table (mean)**")
            st.caption("Analysis IC hour (one-step AF) — not autoregressive lead time")
            st.dataframe(summary.round(3), use_container_width=True, hide_index=True)

        if "variable" in df.columns:
            tp = df[df["variable"] == "tp"]
            if not tp.empty and "student_acc" in tp.columns:
                pod = float(tp["pod_1mm"].mean()) if "pod_1mm" in tp.columns else float("nan")
                st.caption(
                    f"tp diagnostic: mean ACC≈{tp['student_acc'].mean():.3f}, "
                    f"POD₁ₘₘ={pod:.3f} (out-of-scope)"
                )

    # Spatial error maps from campaign figures
    fig_dir = REPORTS / "figures"
    map_files = [
        ("00Z RMSE", fig_dir / "trackb_t2m_v5_rmse_L006h.png"),
        ("00Z bias", fig_dir / "trackb_t2m_v5_bias_L006h.png"),
        ("18Z RMSE", fig_dir / "trackb_t2m_v5_rmse_L024h.png"),
        ("18Z bias", fig_dir / "trackb_t2m_v5_bias_L024h.png"),
    ]
    present = [(title, p) for title, p in map_files if p.is_file()]
    if present:
        st.divider()
        st.markdown("#### Spatial error maps (Africa)")
        cols = st.columns(2)
        for i, (title, path) in enumerate(present):
            with cols[i % 2]:
                st.image(str(path), caption=title, use_container_width=True)

    md = _load_text("TRACKB_T2M_EXPANDED.md")
    if md:
        with st.expander("TRACKB_T2M_EXPANDED.md"):
            st.markdown(md)


def _render_laptop_tab() -> None:
    st.subheader("Laptop / deployment demonstration")
    bench = _load_json("MVULA_LAPTOP_BENCHMARK.json")
    md = _load_text("MVULA_LAPTOP_BENCHMARK.md")
    if not bench:
        st.warning("reports/MVULA_LAPTOP_BENCHMARK.json not found")
        if md:
            st.markdown(md)
        return
    host_class = bench.get("host_class") or ""
    if host_class == "consumer_laptop":
        st.success(
            bench.get("caveat")
            or "Measured on a consumer laptop (CPU-only)."
        )
    else:
        st.warning(
            bench.get("caveat")
            or "CPU proxy bench — re-run on a consumer laptop for close-out claims."
        )
    a, b, c, d = st.columns(4)
    a.metric("Student ckpt", f"{bench.get('student_ckpt_mib', 0):.1f} MiB", f"{bench.get('student_params_m', 0):.2f} M params")
    b.metric("vs K1 teacher", f"{bench.get('size_reduction_factor_disk', 0):.1f}× smaller", f"teacher {bench.get('teacher_ckpt_mib', 0):.1f} MiB")
    c.metric("+6h step (CPU)", f"{bench.get('mean_step_seconds', 0):.2f} s", f"{bench.get('omp_num_threads', '?')} threads")
    d.metric("GPU required", "No" if not bench.get("gpu_required") else "Yes", "free-run: No")
    hw = bench.get("hardware") or {}
    cpu_label = hw.get("cpu_model") or hw.get("processor") or "CPU"
    st.markdown(
        f"- Host class: **{host_class or 'unknown'}** · {cpu_label}\n"
        f"- AF 4-lead inference-only package ≈ **{bench.get('af_package_4leads_infer_only_seconds', 0):.1f} s** "
        "(IC build not included)\n"
        f"- Peak RSS ≈ **{bench.get('rss_peak_mib', 0):.0f} MiB**\n"
        f"- Free-run supported: **{bench.get('free_run_supported')}**"
    )
    fig = go.Figure(
        data=[
            go.Bar(
                name="Disk size (MiB)",
                x=["Student v5", "K1 teacher"],
                y=[bench.get("student_ckpt_mib"), bench.get("teacher_ckpt_mib")],
                marker_color=["#ffb74d", "#64b5f6"],
            )
        ]
    )
    st.plotly_chart(_layout(fig, title="Checkpoint disk size", height=320), use_container_width=True)
    if md:
        with st.expander("Full laptop benchmark notes"):
            st.markdown(md)

    st.divider()
    st.subheader("Explainability (Fig. 11)")
    st.caption(
        "Gradient saliency: what input channels drive African 2t, plus 06Z-IC one-step pathology. "
        "See reports/MVULA_XAI_ATTRIBUTION.md"
    )
    xai = REPORTS / "figures" / "mvula_fig11_t2m_xai_attribution.png"
    if xai.is_file():
        st.image(str(xai), caption="Fig. 11 — channel attribution + 06Z-IC pathology", use_container_width=True)
    else:
        st.info("Generate with: `python scripts/plot_mvula_xai_attribution.py`")
    xai_md = _load_text("MVULA_XAI_ATTRIBUTION.md")
    if xai_md:
        with st.expander("MVULA_XAI_ATTRIBUTION.md"):
            st.markdown(xai_md)


def main() -> None:
    gate = _load_json("TRACKA_A1_GATE.json")
    track_a_sc = _load_json("TRACKA_COARSEN_SCORECARD.json")
    phase0_sc = _load_json("PHASE0_BASELINE_SCORECARD.json")
    passed, total, gate_ok = _gate_summary(gate)
    gate_b = _load_json("TRACKB_GATE_V5.json") or _load_json("TRACKB_GATE.json")
    bench = _load_json("MVULA_LAPTOP_BENCHMARK.json")
    expanded_ready = (REPORTS / "TRACKB_T2M_EXPANDED.json").is_file()

    st.title("Mvula / LapAI-Forecast — project status")
    st.caption("Code for Earth · African Stream · dual close-out: AF one-step t2m (00Z headline) + laptop demo")

    if LOGO.is_file():
        h1, h2 = st.columns([2, 3])
        with h1:
            st.image(str(LOGO), use_container_width=True)
        with h2:
            st.markdown(
                "**Mvula** — Compressing ECMWF’s AIFS for Edge Deployment.\n\n"
                "Code for Earth 2026 · African Stream · freeze **v5**: "
                "analysis-forced **one-step** African **t2m** (headline **00Z** IC) · "
                "~8.8 MiB student · CPU inference · no free-run / AR."
            )
    else:
        st.caption("Banner missing — expected at `docs/branding/mvula_banner.jpg`.")

    st.info(
        "Freeze **v5** as the Code for Earth student artefact: analysis-forced **one-step** "
        "**t2m** (headline **00Z** IC; **06Z** cold-bias pathology documented) vs K1, "
        "plus CPU size/speed evidence. **No** free-running / AR 10-day claim from Cout=3."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Phase 0 / Track A", "Done", "K1 teacher accepted")
    c2.metric("Student", "v5 freeze", "65→3 · Case A")
    if gate_b:
        checks = gate_b.get("checks") or []
        pc = sum(1 for c in checks if c.get("passed"))
        c3.metric("Cache gate", f"{pc}/{len(checks)}", "tp soft-fail")
    else:
        c3.metric("Cache gate", "—", "no JSON")
    c4.metric(
        "AF t2m campaign",
        "Ready" if expanded_ready else "Running / pending",
        "expanded JSON" if expanded_ready else "see Track B tab",
    )
    if bench:
        c5.metric(
            "Laptop CPU",
            f"{bench.get('student_ckpt_mib', 0):.1f} MiB",
            f"~{bench.get('mean_step_seconds', 0):.1f}s/step",
        )
    else:
        c5.metric("Laptop CPU", "—", "bench JSON missing")

    (
        tab_mvula,
        tab_trackb,
        tab_laptop,
        tab_overview,
        tab_figures,
        tab_spatial,
        tab_pipe,
        tab_ic,
        tab_team,
        tab_gate,
        tab_todo,
    ) = st.tabs(
        [
            "Mvula matrix",
            "Track B skill",
            "Laptop demo",
            "Track A overview",
            "Track A figures",
            "Spatial maps",
            "Pipelines",
            "IC regridding",
            "2024–2025 split",
            "A1 gate",
            "Checklist",
        ]
    )

    with tab_mvula:
        _render_mvula_tab()
    with tab_trackb:
        _render_trackb_tab()
    with tab_laptop:
        _render_laptop_tab()

    base_long = _scorecard_long(phase0_sc, "teacher")
    cand_long = _scorecard_long(track_a_sc, "coarsened")

    with tab_overview:
        st.subheader("t2m RMSE at +24 h — coarsened vs teacher")
        df = _t2m_24h_table(track_a_sc, phase0_sc)
        if df is not None and not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
            fig = go.Figure()
            fig.add_trace(
                go.Bar(name="Teacher (N320)", x=df["Init"], y=df["Teacher RMSE (K)"], marker_color="#1565c0")
            )
            fig.add_trace(
                go.Bar(name="Coarsened O96", x=df["Init"], y=df["Coarsened RMSE (K)"], marker_color="#f9a825")
            )
            fig.update_layout(barmode="group", xaxis_title="Init date", yaxis_title="RMSE (K)")
            st.plotly_chart(_layout(fig, title="t2m RMSE at +24 h by init date", height=360), use_container_width=True)
            st.caption("Source: reports/TRACKA_COARSEN_SCORECARD.json vs PHASE0_BASELINE_SCORECARD.json · smoke run")
        else:
            st.warning("Scorecard JSON not found under reports/ — run scoring locally to populate charts.")

    with tab_figures:
        st.subheader("Forecast skill — teacher vs coarsened smoke run")
        st.caption(
            "Jan 2023 weekly inits · Africa eval domain · GCS ERA5 truth · "
            "coarsened model = 50-step smoke fine-tune (not production skill)"
        )
        if base_long.empty or cand_long.empty:
            st.warning("Need both PHASE0_BASELINE_SCORECARD.json and TRACKA_COARSEN_SCORECARD.json in reports/.")
        else:
            gate_vars = ["t2m", "u10", "v10"]
            c1, c2, c3 = st.columns(3)
            with c1:
                st.plotly_chart(_rmse_vs_lead_figure(base_long, cand_long, "t2m"), use_container_width=True)
            with c2:
                st.plotly_chart(_rmse_vs_lead_figure(base_long, cand_long, "u10"), use_container_width=True)
            with c3:
                st.plotly_chart(_rmse_vs_lead_figure(base_long, cand_long, "v10"), use_container_width=True)

            st.divider()
            st.markdown("**Anomaly correlation (ACC)** — same inits, mean over 5 starts")
            a1, a2, a3 = st.columns(3)
            with a1:
                st.plotly_chart(_acc_vs_lead_figure(base_long, cand_long, "t2m"), use_container_width=True)
            with a2:
                st.plotly_chart(_acc_vs_lead_figure(base_long, cand_long, "u10"), use_container_width=True)
            with a3:
                st.plotly_chart(_acc_vs_lead_figure(base_long, cand_long, "v10"), use_container_width=True)

            st.divider()
            bc1, bc2 = st.columns(2)
            with bc1:
                lead_pick = st.selectbox("Lead time (h)", [24, 48, 72, 120], index=0)
            with bc2:
                var_pick = st.selectbox("Variable", gate_vars, index=0)
            st.plotly_chart(
                _rmse_by_init_figure(base_long, cand_long, lead_h=lead_pick, variable=var_pick),
                use_container_width=True,
            )

            if gate:
                gdf = pd.DataFrame(
                    [
                        {
                            "Init": c.get("init_date"),
                            "Lead (h)": c.get("lead_hours"),
                            "Var": c.get("variable"),
                            "Passed": c.get("passed"),
                            "Degradation %": c.get("degradation_pct", 0),
                        }
                        for c in gate.get("checks") or []
                    ]
                )
                st.divider()
                st.markdown("**A1 gate figures** — criterion: ≤5% RMSE degradation vs Phase 0 at +24 h / +48 h")
                g1, g2 = st.columns(2)
                with g1:
                    st.plotly_chart(_gate_degradation_heatmap(gdf, 24), use_container_width=True)
                with g2:
                    st.plotly_chart(_gate_degradation_heatmap(gdf, 48), use_container_width=True)
                st.plotly_chart(_gate_pass_summary(gdf), use_container_width=True)

            st.divider()
            st.markdown("**Key takeaway (smoke run)**")
            st.markdown(
                "- **t2m** degrades ~38–58% vs teacher at **+24 h** (main gate failure)\n"
                "- **Winds (u10/v10)** mixed — sometimes better at +24 h, worse at +48 h\n"
                "- Long leads (+120 h+): both models lose ACC on Africa box (teacher too)\n"
                "- **tp** not scored for coarsened run (NaN in forecast NetCDF)"
            )

    with tab_spatial:
        _render_spatial_maps_tab()

    with tab_pipe:
        st.subheader("Phase 0 — N320 teacher baseline (done)")
        st.graphviz_chart(_phase0_graph(), use_container_width=True)
        st.markdown(
            '<p class="muted">Offline Lengau · CDS cache + earthkit N320 matrices · baseline at '
            "reports/PHASE0_BASELINE_SCORECARD.json</p>",
            unsafe_allow_html=True,
        )
        st.subheader("Track A — coarsened O96 student (smoke)")
        st.graphviz_chart(_tracka_graph(), use_container_width=True)
        st.markdown(
            '<p class="muted">Lengau jobs 7318569 (train), 7318585 (forecast) · gate failed on 50-step smoke (expected)</p>',
            unsafe_allow_html=True,
        )

    with tab_ic:
        st.subheader("Mario vs current approach")
        left, right = st.columns(2)
        with left:
            st.markdown("**Today (Lengau offline)**")
            st.markdown(
                "- CDS → **N320** → post-fetch **scipy bridge** → O96 model\n"
                "- Works without O96 earthkit matrices pre-staged\n"
                "- Implemented in `utils/grid_bridge.py` + `run_phase0_forecast.py`"
            )
        with right:
            st.markdown("**Target (2024–2025 benchmark)**")
            st.markdown(
                "- CDS → **earthkit O96** in IC builder (`utils/cds_ic.py`)\n"
                "- Same offline pattern as N320 if matrices are pre-cached on laptop → rsync\n"
                "- Bridge remains **fallback only**"
            )
        st.markdown(
            "**Training** uses native **O96 Zarr** already — the workaround is **forecast ICs only**. "
            "See `PLAN.md` §4.1 and `reports/PHASE0_CLOSURE.md`."
        )

    with tab_team:
        st.subheader("Proposed verification split (Sh)")
        a, b = st.columns(2)
        with a:
            st.markdown("#### We · Lengau (CHPC)")
            st.markdown(
                "- Coarsened **O96 student** train + forecast\n"
                "- CDS + IC path (→ Mario O96-at-fetch)\n"
                "- Score vs Phase 0 + Oxford outputs\n"
                "- GPU nodes: **no outbound internet** (cache prep on laptop)"
            )
        with b:
            st.markdown("#### Sh · Oxford")
            st.markdown(
                "- **N320 AIFS** production (~5–9 days for 2 years)\n"
                "- Native **O96 AIFS** (optional if not parallel)\n"
                "- CDS pipeline (not open-data)\n"
                "- We run **O96 coarsened in parallel** while she runs N320"
            )

    with tab_gate:
        if not gate:
            st.warning("reports/TRACKA_A1_GATE.json not found.")
        else:
            st.subheader(f"A1 gate — {'PASSED' if gate_ok else 'FAILED'}")
            st.markdown(
                f"Variables: {', '.join(gate.get('variables') or [])} · "
                f"Leads: {gate.get('leads_hours')} h · "
                f"Max degradation: {gate.get('max_rmse_degradation_pct')}%"
            )
            rows = []
            for c in gate.get("checks") or []:
                rows.append(
                    {
                        "Init": c.get("init_date"),
                        "Lead (h)": c.get("lead_hours"),
                        "Var": c.get("variable"),
                        "Passed": c.get("passed"),
                        "Baseline RMSE": round(c.get("baseline_rmse", 0), 3),
                        "Candidate RMSE": round(c.get("candidate_rmse", 0), 3),
                        "Degradation %": round(c.get("degradation_pct", 0), 1),
                    }
                )
            gdf = pd.DataFrame(rows)
            st.dataframe(gdf, use_container_width=True, hide_index=True)
            fail = gdf[~gdf["Passed"]]
            if not fail.empty:
                st.markdown("**Failed checks (pattern):** mostly **t2m +24 h** (~38–58% worse than teacher).")

    with tab_todo:
        st.subheader("Done vs next")
        for text, done in TODOS:
            css = "done" if done else "warn"
            mark = "Done" if done else "Next"
            st.markdown(f'<div class="block {css}"><strong>{mark}</strong> — {text}</div>', unsafe_allow_html=True)

    with st.sidebar:
        if LOGO.is_file():
            st.image(str(LOGO), use_container_width=True)
        st.header("Reports")
        for name in (
            "FINAL_REPORT.md",
            "MVULA_CODE4EARTH_STATUS_MATRIX.md",
            "MVULA_LAPTOP_BENCHMARK.json",
            "TRACKB_STATE_CLOSURE.md",
            "TRACKB_GATE_V5.json",
            "TRACKB_HELD_OUT_JAN2023_V5.json",
            "TRACKB_T2M_EXPANDED.json",
            "TRACKA_A1_GATE.json",
            "TRACKA_COARSEN_SCORECARD.json",
            "PHASE0_BASELINE_SCORECARD.json",
        ):
            p = REPORTS / name
            st.write(f"{'OK' if p.is_file() else '—'} `{name}`")
        st.divider()
        st.markdown("**Run locally**")
        st.code("streamlit run streamlit_status.py", language="bash")
        st.caption(f"Repo: `{REPO}`")
        st.caption("Close-out: freeze v5 · AF one-step t2m (00Z headline) · laptop demo · no free-run / AR claim")


if __name__ == "__main__":
    main()

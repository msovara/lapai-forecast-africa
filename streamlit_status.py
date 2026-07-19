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
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

st.set_page_config(
    page_title="LapAI-Forecast Status",
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
    ("Lengau lustre env (anemoi 0.14, sklearn, trimesh fixes)", True),
    ("Track A smoke coarsen training (finite loss, O96 ckpt)", True),
    ("Track A offline forecasts — 5 Jan 2023 inits", True),
    ("GCS scorecard + A1 gate (failed on smoke — expected)", True),
    ("IC regridding design doc in PLAN.md", True),
    ("Mario-style O96 IC builder + earthkit O96 cache prep", False),
    ("Rebuild full training Zarr (fix missing timesteps)", False),
    ("Longer coarsen fine-tune + real A1 gate attempt", False),
    ("2024–2025 paired verification with Oxford", False),
]


def main() -> None:
    gate = _load_json("TRACKA_A1_GATE.json")
    track_a_sc = _load_json("TRACKA_COARSEN_SCORECARD.json")
    phase0_sc = _load_json("PHASE0_BASELINE_SCORECARD.json")
    passed, total, gate_ok = _gate_summary(gate)

    st.title("LapAI-Forecast — project status")
    st.caption("Code for Earth · African Stream · Track A coarsening on Lengau")

    st.info(
        "We got the coarsened weather model working end-to-end on Lengau — train, "
        "forecast offline, score — and we are aligning with Sh on GPU split and IC handling for 2024–2025."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Phase 0", "Closed", "N320 teacher · Jan 2023")
    c2.metric("Track A pipeline", "Green", "train → forecast → score")
    c3.metric("A1 gate (smoke)", "Failed" if gate_ok is False else ("Passed" if gate_ok else "N/A"),
              f"{passed}/{total} checks" if total else "no gate JSON")
    c4.metric("Forecasts scored", "5 inits", "20230101–20230129")

    tab_overview, tab_figures, tab_spatial, tab_pipe, tab_ic, tab_team, tab_gate, tab_todo = st.tabs(
        [
            "Overview",
            "Analysis figures",
            "Spatial maps",
            "Pipelines",
            "IC regridding",
            "2024–2025 split",
            "Gate details",
            "Checklist",
        ]
    )

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
        st.header("Reports")
        for name in (
            "PHASE0_BASELINE_SCORECARD.json",
            "TRACKA_COARSEN_SCORECARD.json",
            "TRACKA_A1_GATE.json",
            "PHASE0_CLOSURE.md",
            "LAPAI_STATUS_DECK.pdf",
        ):
            p = REPORTS / name
            st.write(f"{'OK' if p.is_file() else '—'} `{name}`")
        st.divider()
        st.markdown("**Run locally**")
        st.code("streamlit run streamlit_status.py", language="bash")
        st.caption(f"Repo: `{REPO}`")


if __name__ == "__main__":
    main()

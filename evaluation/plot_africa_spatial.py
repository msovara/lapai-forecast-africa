"""IOD-style Africa spatial maps (Cartopy coastlines / grid / panel labels).

Formatting follows ``comprehensive_iod_drought_statistics_optimized._plot_composite_map``
from the IOD–drought paper workflow: PlateCarree, coastlines+borders, dashed lat/lon
grid, contourf levels, optional panel labels, horizontal colorbar.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from evaluation.masks import lon_to_180

# Africa analysis box used in Track B / Mvula packaged campaign
AFRICA_EXTENT = dict(lon_min=-20.0, lon_max=55.0, lat_min=-40.0, lat_max=40.0)


def _lon180_sorted(lon: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert 0..360 lon → [-180, 180) and reorder columns so lon is ascending.

    Without this, West Africa (340–359°E ≡ −20–−1°) never appears in the
    Africa extent (−20…55) under PlateCarree pcolormesh.
    """
    lon = lon_to_180(np.asarray(lon, dtype=np.float64).reshape(-1))
    order = np.argsort(lon)
    return lon[order], np.asarray(field)[:, order]

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

    HAS_CARTOPY = True
except Exception:  # noqa: BLE001
    ccrs = None  # type: ignore[assignment]
    cfeature = None  # type: ignore[assignment]
    LONGITUDE_FORMATTER = LATITUDE_FORMATTER = None  # type: ignore[assignment]
    HAS_CARTOPY = False


_LAND_MASK_CACHE: dict[tuple[int, int, float, float, float, float], np.ndarray] = {}


def _land_mask(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Boolean land mask on the lon/lat grid (True = land), via Natural Earth."""
    if not HAS_CARTOPY:
        return np.ones((lat.size, lon.size), dtype=bool)

    key = (lat.size, lon.size, float(lon[0]), float(lon[-1]), float(lat[0]), float(lat[-1]))
    if key in _LAND_MASK_CACHE:
        return _LAND_MASK_CACHE[key]

    import cartopy.io.shapereader as shpreader
    from matplotlib.path import Path as MplPath
    from shapely.geometry import box
    from shapely.ops import unary_union

    bbox = box(
        AFRICA_EXTENT["lon_min"] - 1,
        AFRICA_EXTENT["lat_min"] - 1,
        AFRICA_EXTENT["lon_max"] + 1,
        AFRICA_EXTENT["lat_max"] + 1,
    )
    land_geom = unary_union(
        [
            geom
            for geom in shpreader.Reader(
                shpreader.natural_earth(resolution="50m", category="physical", name="land")
            ).geometries()
            if geom.intersects(bbox)
        ]
    ).intersection(bbox)

    if land_geom.is_empty:
        mask2d = np.ones((lat.size, lon.size), dtype=bool)
        _LAND_MASK_CACHE[key] = mask2d
        return mask2d

    if land_geom.geom_type == "Polygon":
        polys = [land_geom]
    else:
        polys = [g for g in land_geom.geoms if g.geom_type == "Polygon"]

    lon2d, lat2d = np.meshgrid(lon, lat)
    pts = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    mask = np.zeros(pts.shape[0], dtype=bool)
    for poly in polys:
        exterior = MplPath(np.asarray(poly.exterior.coords))
        inside = exterior.contains_points(pts)
        for interior in poly.interiors:
            hole = MplPath(np.asarray(interior.coords))
            inside &= ~hole.contains_points(pts)
        mask |= inside
    mask2d = mask.reshape(lon2d.shape)
    # Shrink land by one grid cell so contour-adjacent ocean cells stay masked
    from numpy import roll

    eroded = (
        mask2d
        & roll(mask2d, 1, 0)
        & roll(mask2d, -1, 0)
        & roll(mask2d, 1, 1)
        & roll(mask2d, -1, 1)
    )
    _LAND_MASK_CACHE[key] = eroded
    return eroded


def plot_africa_field(
    ax,
    lon: np.ndarray,
    lat: np.ndarray,
    field: np.ndarray,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    title: str,
    cbar_label: str | None = None,
    panel_label: str | None = None,
    add_colorbar: bool = True,
    n_levels: int = 21,  # kept for API compat; pcolormesh uses continuous norm
    mask_ocean: bool = True,
    left_labels: bool = True,
    bottom_labels: bool = True,
    draw_grid: bool = True,
):
    """Draw one Africa map in IOD publication style. Returns the pcolormesh mappable."""
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    field = np.asarray(field, dtype=float).copy()
    lon, field = _lon180_sorted(lon, field)
    if mask_ocean:
        land = _land_mask(lon, lat)
        field = np.where(land, field, np.nan)
    field_ma = np.ma.masked_invalid(field)

    if HAS_CARTOPY:
        import matplotlib.ticker as mticker
        from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

        ax.set_extent(
            [
                AFRICA_EXTENT["lon_min"],
                AFRICA_EXTENT["lon_max"],
                AFRICA_EXTENT["lat_min"],
                AFRICA_EXTENT["lat_max"],
            ],
            crs=ccrs.PlateCarree(),
        )
        ax.set_facecolor("white")
        ax.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="#f7f7f7", edgecolor="none", zorder=1)
        # pcolormesh (not contourf): NaNs stay empty — no ocean bleed
        im = ax.pcolormesh(
            lon,
            lat,
            field_ma,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            shading="auto",
            zorder=2,
        )
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, zorder=3)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="black", zorder=4)

        # Optional dashed lat/lon grid (off for compact publication quads)
        if draw_grid:
            ax.gridlines(
                crs=ccrs.PlateCarree(),
                draw_labels=False,
                linewidth=0.5,
                color="gray",
                alpha=0.5,
                linestyle="--",
                xlocs=[-20, -10, 0, 10, 20, 30, 40, 50],
                ylocs=[-40, -30, -20, -10, 0, 10, 20, 30, 40],
            )
        xticks = [-20, -10, 0, 10, 20, 30, 40, 50]
        yticks = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
        ax.set_xticks(xticks, crs=ccrs.PlateCarree())
        ax.set_yticks(yticks, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol="°"))
        ax.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol="°"))
        ax.tick_params(
            axis="both",
            labelsize=8,
            length=3,
            pad=2,
            labelbottom=bottom_labels,
            labelleft=left_labels,
            labeltop=False,
            labelright=False,
            bottom=True,
            left=left_labels,
            top=False,
            right=False,
        )
        if bottom_labels:
            ax.set_xlabel("Longitude", fontsize=9, fontweight="bold", labelpad=2)
        else:
            ax.set_xlabel("")
        if left_labels:
            ax.set_ylabel("Latitude", fontsize=9, fontweight="bold", labelpad=2)
        else:
            ax.set_ylabel("")
    else:
        im = ax.pcolormesh(lon, lat, field_ma, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        ax.set_xlim(AFRICA_EXTENT["lon_min"], AFRICA_EXTENT["lon_max"])
        ax.set_ylim(AFRICA_EXTENT["lat_min"], AFRICA_EXTENT["lat_max"])
        if draw_grid:
            ax.grid(True, alpha=0.3, linestyle="--")
        # Titles applied once at end (after optional colorbar)

    if add_colorbar:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, ScalarFormatter

        cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.12, shrink=0.8, aspect=30)
        cbar.ax.tick_params(labelsize=8)
        # Avoid overlapping 0.0000x tick strings on horizontal bars
        cbar.locator = MaxNLocator(nbins=5)
        span = abs(float(vmax) - float(vmin)) if vmax is not None and vmin is not None else 0.0
        if span > 0 and span < 1e-2:
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((-1, 1))
            cbar.formatter = fmt
        cbar.update_ticks()
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=10, fontweight="bold")

    display_title = f"({panel_label}) {title}" if panel_label else title
    ax.set_title(display_title, fontsize=11, fontweight="bold", pad=6)

    # Axis titles after colorbar so they are not swallowed by layout changes
    if bottom_labels:
        ax.set_xlabel("Longitude", fontsize=9, fontweight="bold", labelpad=4)
    if left_labels:
        ax.set_ylabel("Latitude", fontsize=9, fontweight="bold", labelpad=4)
    return im


def write_lead_spatial_figures(
    *,
    lead: int,
    lat: np.ndarray,
    lon: np.ndarray,
    mean_bias: np.ndarray,
    mean_rmse: np.ndarray,
    n_inits: int,
    figures_dir: Path,
    also_combined: bool = True,
) -> dict[str, str]:
    """Write single-panel bias/RMSE maps and optional combined Fig.4/5-style panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    # 00Z campaign AF map: lead L ↔ analysis IC hour (L−6) % 24
    ic_hour = (int(lead) - 6) % 24
    ic_lab = f"{ic_hour:02d}Z"

    bias_vmax = float(np.nanpercentile(np.abs(mean_bias), 98))
    bias_vmax = max(0.5, float(np.ceil(bias_vmax * 2) / 2))  # neat 0.5 °C steps
    rmse_vmax = float(np.nanpercentile(mean_rmse, 98))
    rmse_vmax = max(0.5, float(np.ceil(rmse_vmax * 2) / 2))

    specs = (
        ("bias", mean_bias, "RdBu_r", -bias_vmax, bias_vmax, "Mean bias (°C)"),
        ("rmse", mean_rmse, "YlOrRd", 0.0, rmse_vmax, "RMSE (°C)"),
    )

    for kind, field, cmap, vmin, vmax, label in specs:
        if HAS_CARTOPY:
            fig, ax = plt.subplots(
                figsize=(8.5, 7.0),
                dpi=200,
                subplot_kw={"projection": ccrs.PlateCarree()},
            )
        else:
            fig, ax = plt.subplots(figsize=(8.5, 7.0), dpi=200)
        plot_africa_field(
            ax,
            lon,
            lat,
            field,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            title=f"Student v5 t2m {kind} · Africa · {ic_lab} IC one-step (n={n_inits})",
            cbar_label=label,
            add_colorbar=True,
        )
        out = figures_dir / f"trackb_t2m_v5_{kind}_L{lead:03d}h.png"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths[f"{kind}_L{lead}h"] = str(out)
        print(f"[figures] wrote {out}", flush=True)

    if also_combined:
        # Publication 1×2: aspect-matched panels; colorbars attached under each map.
        lon_span = AFRICA_EXTENT["lon_max"] - AFRICA_EXTENT["lon_min"]
        lat_span = AFRICA_EXTENT["lat_max"] - AFRICA_EXTENT["lat_min"]
        data_aspect = lon_span / lat_span
        fig_w, fig_h = 10.0, 5.2
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)

        panel_h = 0.72
        panel_w = data_aspect * panel_h * (fig_h / fig_w)
        gap = 0.015
        left0 = 0.10
        left1 = left0 + panel_w + gap
        map_bottom = 0.20

        if HAS_CARTOPY:
            ax0 = fig.add_axes([left0, map_bottom, panel_w, panel_h], projection=ccrs.PlateCarree())
            ax1 = fig.add_axes([left1, map_bottom, panel_w, panel_h], projection=ccrs.PlateCarree())
        else:
            ax0 = fig.add_axes([left0, map_bottom, panel_w, panel_h])
            ax1 = fig.add_axes([left1, map_bottom, panel_w, panel_h])

        im0 = plot_africa_field(
            ax0,
            lon,
            lat,
            mean_bias,
            cmap="RdBu_r",
            vmin=-bias_vmax,
            vmax=bias_vmax,
            title=f"Mean bias · {ic_lab} IC (one-step)",
            panel_label="a",
            add_colorbar=False,
            left_labels=True,
            bottom_labels=True,
        )
        im1 = plot_africa_field(
            ax1,
            lon,
            lat,
            mean_rmse,
            cmap="YlOrRd",
            vmin=0.0,
            vmax=rmse_vmax,
            title=f"RMSE · {ic_lab} IC (one-step)",
            panel_label="b",
            add_colorbar=False,
            left_labels=False,
            bottom_labels=True,
        )

        # Colorbar under each panel; Longitude title placed in axes coords so it
        # cannot be dropped by GeoAxes/colorbar layout.
        cbar0 = fig.colorbar(
            im0, ax=ax0, orientation="horizontal", fraction=0.048, pad=0.16, aspect=30
        )
        cbar0.set_label("Mean bias (°C)", fontsize=9, fontweight="bold")
        cbar0.ax.tick_params(labelsize=8)

        cbar1 = fig.colorbar(
            im1, ax=ax1, orientation="horizontal", fraction=0.048, pad=0.16, aspect=30
        )
        cbar1.set_label("RMSE (°C)", fontsize=9, fontweight="bold")
        cbar1.ax.tick_params(labelsize=8)

        ax0.set_ylabel("Latitude", fontsize=9, fontweight="bold", labelpad=3)
        ax1.set_ylabel("")
        for ax in (ax0, ax1):
            ax.set_xlabel("")  # avoid double labels
            ax.text(
                0.5,
                -0.10,
                "Longitude",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
                fontweight="bold",
                clip_on=False,
                zorder=10,
            )

        fig.suptitle(
            f"African t2m spatial verification · student v5 · {ic_lab} analysis IC (one-step AF) "
            f"(n={n_inits} inits, 2023)",
            fontsize=11,
            fontweight="bold",
            y=0.98,
        )

        fig_id = 4 if lead == 6 else 5 if lead == 24 else None
        if fig_id is not None:
            out_c = figures_dir / f"mvula_fig{fig_id:02d}_t2m_spatial_plus{lead}h.png"
        else:
            out_c = figures_dir / f"mvula_t2m_spatial_plus{lead}h.png"
        fig.savefig(out_c, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.04)
        plt.close(fig)
        paths[f"combined_L{lead}h"] = str(out_c)
        print(f"[figures] wrote {out_c}", flush=True)

    return paths


def write_quad_spatial_figure(
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    bias_6: np.ndarray,
    rmse_6: np.ndarray,
    bias_24: np.ndarray,
    rmse_24: np.ndarray,
    n_inits: int,
    figures_dir: Path,
    out_name: str = "mvula_fig10_t2m_spatial_6h_24h_quad.png",
) -> Path:
    """Publication 2×2: bias/RMSE at +6 h and +24 h with shared color scales."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    bias_vmax = float(
        np.nanpercentile(np.abs(np.concatenate([bias_6.ravel(), bias_24.ravel()])), 98)
    )
    bias_vmax = max(0.5, float(np.ceil(bias_vmax * 2) / 2))
    rmse_vmax = float(np.nanpercentile(np.concatenate([rmse_6.ravel(), rmse_24.ravel()]), 98))
    rmse_vmax = max(0.5, float(np.ceil(rmse_vmax * 2) / 2))

    # Rows = metric (bias, RMSE); columns = analysis IC hour (00Z / 18Z for 00Z campaign)
    panels = (
        (bias_6, "RdBu_r", -bias_vmax, bias_vmax, "a", "Mean bias · 00Z IC (one-step)"),
        (bias_24, "RdBu_r", -bias_vmax, bias_vmax, "b", "Mean bias · 18Z IC (one-step)"),
        (rmse_6, "YlOrRd", 0.0, rmse_vmax, "c", "RMSE · 00Z IC (one-step)"),
        (rmse_24, "YlOrRd", 0.0, rmse_vmax, "d", "RMSE · 18Z IC (one-step)"),
    )

    fig_w, fig_h = 10.4, 8.6
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)

    # Compact 2×2: modest gaps so panels read as one figure without crowding
    left, right = 0.08, 0.82
    bottom, top = 0.055, 0.935
    wspace, hspace = 0.032, 0.055
    cell_w = (right - left - wspace) / 2
    cell_h = (top - bottom - hspace) / 2

    positions = (
        (left, bottom + cell_h + hspace, cell_w, cell_h),  # a
        (left + cell_w + wspace, bottom + cell_h + hspace, cell_w, cell_h),  # b
        (left, bottom, cell_w, cell_h),  # c
        (left + cell_w + wspace, bottom, cell_w, cell_h),  # d
    )

    ims: list[Any] = []
    axes = []
    for (field, cmap, vmin, vmax, label, title), pos in zip(panels, positions):
        if HAS_CARTOPY:
            ax = fig.add_axes(pos, projection=ccrs.PlateCarree())
        else:
            ax = fig.add_axes(pos)
        axes.append(ax)
        col = 0 if label in ("a", "c") else 1
        row = 0 if label in ("a", "b") else 1
        im = plot_africa_field(
            ax,
            lon,
            lat,
            field,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            title=title,
            panel_label=label,
            add_colorbar=False,
            left_labels=(col == 0),
            bottom_labels=(row == 1),
            draw_grid=False,
        )
        ims.append(im)
        if row == 0:
            ax.set_xlabel("")
        if col == 1:
            ax.set_ylabel("")

    # Shared vertical colorbars (bias top row, RMSE bottom row)
    cax_bias = fig.add_axes([0.855, bottom + cell_h + hspace + 0.02, 0.022, cell_h - 0.04])
    cbar_b = fig.colorbar(ims[0], cax=cax_bias, orientation="vertical")
    cbar_b.set_label("Mean bias (°C)", fontsize=10, fontweight="bold")
    cbar_b.ax.tick_params(labelsize=8)

    cax_rmse = fig.add_axes([0.855, bottom + 0.02, 0.022, cell_h - 0.04])
    cbar_r = fig.colorbar(ims[2], cax=cax_rmse, orientation="vertical")
    cbar_r.set_label("RMSE (°C)", fontsize=10, fontweight="bold")
    cbar_r.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"African t2m spatial verification · student v5 · 00Z vs 18Z analysis IC (one-step AF) "
        f"(n={n_inits} inits, 2023)",
        fontsize=12,
        fontweight="bold",
        y=0.975,
    )

    out = figures_dir / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.05)
    plt.close(fig)
    print(f"[figures] wrote {out}", flush=True)
    return out


def replot_from_npz(figures_dir: Path | None = None) -> dict[str, Any]:
    """Regenerate PNGs from saved ``trackb_t2m_v5_spatial_L*.npz`` grids."""
    _LAND_MASK_CACHE.clear()
    root = Path(__file__).resolve().parents[1]
    figures_dir = Path(figures_dir) if figures_dir else root / "reports" / "figures"
    written: dict[str, Any] = {}
    by_lead: dict[int, dict[str, np.ndarray]] = {}
    for npz_path in sorted(figures_dir.glob("trackb_t2m_v5_spatial_L*h.npz")):
        lead = int(npz_path.stem.split("_L")[-1].replace("h", ""))
        blob = np.load(npz_path)
        paths = write_lead_spatial_figures(
            lead=lead,
            lat=blob["lat"],
            lon=blob["lon"],
            mean_bias=blob["mean_bias"],
            mean_rmse=blob["mean_rmse"],
            n_inits=int(blob["n"][0]),
            figures_dir=figures_dir,
            also_combined=True,
        )
        written[str(lead)] = paths
        by_lead[lead] = {
            "lat": blob["lat"],
            "lon": blob["lon"],
            "mean_bias": blob["mean_bias"],
            "mean_rmse": blob["mean_rmse"],
            "n": int(blob["n"][0]),
        }

    if 6 in by_lead and 24 in by_lead:
        n_inits = min(by_lead[6]["n"], by_lead[24]["n"])
        quad = write_quad_spatial_figure(
            lat=by_lead[6]["lat"],
            lon=by_lead[6]["lon"],
            bias_6=by_lead[6]["mean_bias"],
            rmse_6=by_lead[6]["mean_rmse"],
            bias_24=by_lead[24]["mean_bias"],
            rmse_24=by_lead[24]["mean_rmse"],
            n_inits=n_inits,
            figures_dir=figures_dir,
        )
        written["quad_6_24"] = str(quad)
    return written


if __name__ == "__main__":
    out = replot_from_npz()
    print(f"regenerated leads: {sorted(out)}")

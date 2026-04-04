from __future__ import annotations

"""Global map visualizations using cartopy.

Styling guide:
- Shared: ocean=#f0f2f5, land-no-data=#e8e4df, coastlines=#888/0.3,
  borders=#ccc/0.15, DejaVu Sans, no pure black, dpi=300
- Type 1 (diverging): cmcrameri 'vik', symmetric around 0
- Type 2 (sequential error): cmcrameri 'lajolla', 0→high
- Type 3 (multi-panel): PlateCarree, shared colorbar, panel labels
"""

import logging
import string
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# ── Shared constants ──────────────────────────────────────────────

_OCEAN_COLOR = "#f0f2f5"
_LAND_COLOR = "#e8e4df"
_COAST_KW = {"linewidth": 0.3, "color": "#888888"}
_BORDER_KW = {"linewidth": 0.15, "edgecolor": "#cccccc", "facecolor": "none"}
_GRID_KW = {"linestyle": "--", "color": "#dddddd", "linewidth": 0.2, "alpha": 0.3}
_SPINE_KW = {"linewidth": 0.5, "edgecolor": "#aaaaaa"}
_TITLE_KW = {"fontsize": 11, "fontweight": "regular", "color": "#333333", "pad": 12, "fontfamily": "DejaVu Sans"}
_LABEL_COLOR = "#333333"
_TICK_COLOR = "#555555"

DISPLAY_NAMES = {
    "era5land": "ERA5-Land",
    "gleam": "GLEAM",
    "pmlv2": "PML v2",
    "sap_velocity": "Sap Velocity",
    "gldas": "GLDAS",
}


def _fmt_name(raw: str) -> str:
    """Format a product key into a display-ready name."""
    return DISPLAY_NAMES.get(raw, raw.replace("_", " ").title())


def _add_features(ax, *, gridlines: bool = True) -> None:
    """Add ocean, land, coastlines, borders, gridlines to an axis."""
    import cartopy.feature as cfeature

    ax.add_feature(cfeature.OCEAN, facecolor=_OCEAN_COLOR, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor=_LAND_COLOR, zorder=0)
    ax.coastlines(**_COAST_KW)
    ax.add_feature(cfeature.BORDERS, **_BORDER_KW)
    if gridlines:
        gl = ax.gridlines(draw_labels=False, **_GRID_KW)
        gl.top_labels = gl.right_labels = False
    ax.spines["geo"].set(**_SPINE_KW)


def _horizontal_cbar(fig, im, ax, *, label: str = "", ticks=None) -> None:
    """Add a thin horizontal colorbar centered below one axis."""

    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        fraction=0.025,
        pad=0.06,
        shrink=0.6,
        aspect=35,
    )
    cbar.set_label(label, fontsize=9, color=_LABEL_COLOR, fontfamily="DejaVu Sans")
    cbar.ax.tick_params(labelsize=8, labelcolor=_TICK_COLOR)
    cbar.outline.set_edgecolor("#aaaaaa")
    cbar.outline.set_linewidth(0.5)
    if ticks is not None:
        cbar.set_ticks(ticks)


def _get_diverging_cmap():
    """Return cmcrameri 'vik' if available, else RdBu_r."""
    try:
        from cmcrameri import cm

        return cm.vik
    except ImportError:
        return plt.get_cmap("RdBu_r")


def _get_sequential_error_cmap():
    """Return cmcrameri 'lajolla' if available, else YlOrRd."""
    try:
        from cmcrameri import cm

        return cm.lajolla
    except ImportError:
        return plt.get_cmap("YlOrRd")


def _save(fig, output_path: Path) -> None:
    """Save figure as PNG (always) and PDF (always), at 300 dpi."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


# ── Type 1: Single global map (general / diverging) ──────────────


def plot_global_map(
    data: xr.DataArray,
    title: str,
    output_path: Path,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    units: str = "",
    figsize: tuple[int, int] = (14, 7),
) -> None:
    """Plot a single global map with Robinson projection."""
    import cartopy.crs as ccrs

    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={"projection": ccrs.Robinson()})
    _add_features(ax)

    resolved_cmap = cmap if cmap is not None else "RdYlBu_r"
    im = data.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=resolved_cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
    )
    ax.set_global()
    ax.set_title(title, **_TITLE_KW)
    _horizontal_cbar(fig, im, ax, label=units)

    _save(fig, output_path)


# ── Type 1a: Diverging correlation map ────────────────────────────


def plot_correlation_map(
    corr: xr.DataArray,
    title: str,
    output_path: Path,
    figsize: tuple[int, int] = (14, 7),
) -> None:
    """Diverging map for correlation values (-1 to 1)."""
    import cartopy.crs as ccrs

    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={"projection": ccrs.Robinson()})
    _add_features(ax)

    im = corr.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=_get_diverging_cmap(),
        vmin=-1,
        vmax=1,
        add_colorbar=False,
    )
    ax.set_global()
    ax.set_title(title, **_TITLE_KW)
    _horizontal_cbar(fig, im, ax, label="Pearson r", ticks=[-1, -0.5, 0, 0.5, 1])

    _save(fig, output_path)


# ── Type 2: Sequential error map (RMSE, MAE) ─────────────────────


def plot_error_map(
    data: xr.DataArray,
    title: str,
    output_path: Path,
    units: str = "RMSE",
    vmin: float = 0,
    vmax: float | None = None,
    figsize: tuple[int, int] = (14, 7),
) -> None:
    """Sequential error map (0 → high). Low = good, high = bad."""
    import cartopy.crs as ccrs

    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={"projection": ccrs.Robinson()})
    _add_features(ax)

    cmap = _get_sequential_error_cmap()
    # Truncate bottom 5% so low values are visible against land fill
    from matplotlib.colors import LinearSegmentedColormap

    colors = cmap(np.linspace(0.05, 1.0, 256))
    cmap_trunc = LinearSegmentedColormap.from_list("lajolla_trunc", colors)

    im = data.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap_trunc,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
    )
    ax.set_global()
    ax.set_title(title, **_TITLE_KW)
    _horizontal_cbar(fig, im, ax, label=units)

    _save(fig, output_path)


# ── Type 3: Multi-panel comparison ────────────────────────────────


def plot_multi_product_maps(
    datasets: dict[str, xr.DataArray],
    title: str,
    output_path: Path,
    cmap: str = "RdYlBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    units: str = "",
) -> None:
    """Multi-panel map with Plate Carree, shared colorbar, panel labels."""
    import cartopy.crs as ccrs

    n = len(datasets)
    if n <= 0:
        return

    # Grid layout: avoid orphan panels
    layout = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2), 5: (2, 3), 6: (2, 3)}
    nrows, ncols = layout.get(n, (2, (n + 1) // 2))
    figw = {1: 8, 2: 12, 3: 16}.get(ncols, 14)
    figh = {1: 5, 2: 8}.get(nrows, 4 * nrows)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figw, figh),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes).flatten()

    im = None
    panel_labels = list(string.ascii_lowercase)
    for idx, (name, data) in enumerate(datasets.items()):
        ax = axes[idx]
        _add_features(ax, gridlines=False)
        im = data.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False,
        )
        ax.set_global()
        ax.set_title(_fmt_name(name), fontsize=9, fontweight="medium", color=_LABEL_COLOR, fontfamily="DejaVu Sans")
        # Panel label (a), (b), ...
        ax.text(
            0.02,
            0.95,
            f"({panel_labels[idx]})",
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            color=_LABEL_COLOR,
            va="top",
            fontfamily="DejaVu Sans",
        )

    # Hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    # Shared colorbar below all panels
    if im is not None:
        cbar = fig.colorbar(
            im,
            ax=axes[:n].tolist(),
            orientation="horizontal",
            fraction=0.03,
            pad=0.08,
            shrink=0.7,
            aspect=40,
        )
        cbar.set_label(units, fontsize=9, color=_LABEL_COLOR, fontfamily="DejaVu Sans")
        cbar.ax.tick_params(labelsize=8, labelcolor=_TICK_COLOR)
        cbar.outline.set_edgecolor("#aaaaaa")
        cbar.outline.set_linewidth(0.5)

    fig.suptitle(title, fontsize=11, fontweight="regular", color=_LABEL_COLOR, fontfamily="DejaVu Sans", y=1.02)
    fig.subplots_adjust(wspace=0.15, hspace=0.25)

    _save(fig, output_path)


# ── Consensus map ─────────────────────────────────────────────────


def plot_consensus_map(
    consensus_ds: xr.Dataset,
    output_path: Path,
) -> None:
    """Plot consensus zone map (high/medium/low agreement)."""
    import cartopy.crs as ccrs
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(1, 1, figsize=(14, 7), subplot_kw={"projection": ccrs.Robinson()})
    _add_features(ax)

    cmap = ListedColormap(["#d73027", "#fee08b", "#1a9850"])
    im = consensus_ds["consensus_zone"].plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=0.5,
        vmax=3.5,
        add_colorbar=False,
    )
    ax.set_global()
    ax.set_title("Product Consensus Zones", **_TITLE_KW)

    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        fraction=0.025,
        pad=0.06,
        shrink=0.6,
        aspect=35,
    )
    cbar.set_ticks([1, 2, 3])
    cbar.set_ticklabels(["Low", "Medium", "High"])
    cbar.set_label("Agreement Level", fontsize=9, color=_LABEL_COLOR, fontfamily="DejaVu Sans")
    cbar.ax.tick_params(labelsize=8, labelcolor=_TICK_COLOR)
    cbar.outline.set_edgecolor("#aaaaaa")
    cbar.outline.set_linewidth(0.5)

    _save(fig, output_path)

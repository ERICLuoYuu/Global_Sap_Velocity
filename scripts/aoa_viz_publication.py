"""Convert AOA GeoTIFFs to publication PNGs.

Per-file output: {stem}_DI.png/.pdf + {stem}_AOA.png/.pdf
Uses rasterio.plot.show() on cartopy PlateCarree for raster-basemap alignment.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmcrameri.cm as cmc
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import rasterio
from rasterio.plot import show as rio_show

# ── Shared styling ────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "text.color": "#333333",
        "axes.labelcolor": "#333333",
        "axes.titlesize": 11,
        "axes.titleweight": "regular",
        "axes.titlepad": 12,
        "axes.edgecolor": "#aaaaaa",
        "axes.linewidth": 0.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
)

AOA_THRESHOLD = 0.4953

TITLE_MAP = {
    "di_overall": "Overall (Jan\u2013Mar 2016)",
    "di_monthly_2016_01": "January 2016",
    "di_monthly_2016_02": "February 2016",
    "di_monthly_2016_03": "March 2016",
    "di_clim_01": "Climatological January",
    "di_clim_02": "Climatological February",
    "di_clim_03": "Climatological March",
    "di_yearly_2016": "Annual 2016",
}

# ── Colors ────────────────────────────────────────────────────────────
CLR_OCEAN = "#f0f2f5"
CLR_LAND_NODATA = "#e8e4df"
CLR_COAST = "#888888"
CLR_BORDER = "#cccccc"
CLR_GRID = "#dddddd"
CLR_TEXT = "#333333"
CLR_TEXT_LIGHT = "#555555"
CLR_AOA_INSIDE = "#2a9d8f"
CLR_AOA_OUTSIDE = "#e76f51"


# ── Helpers ───────────────────────────────────────────────────────────


def setup_map(fig, pos, bounds):
    """Create a PlateCarree GeoAxes with shared cartographic styling."""
    ax = fig.add_subplot(pos, projection=ccrs.PlateCarree())
    ax.set_global()

    # Background fills
    ax.add_feature(
        cfeature.OCEAN,
        facecolor=CLR_OCEAN,
        edgecolor="none",
        zorder=0,
    )
    ax.add_feature(
        cfeature.LAND,
        facecolor=CLR_LAND_NODATA,
        edgecolor="none",
        zorder=0,
    )

    # Coastlines and borders
    ax.coastlines(linewidth=0.3, color=CLR_COAST, zorder=3)
    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.15,
        edgecolor=CLR_BORDER,
        alpha=0.5,
        zorder=3,
    )

    # Gridlines
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.2,
        color=CLR_GRID,
        alpha=0.3,
        linestyle="--",
        zorder=2,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(range(-180, 181, 30))
    gl.ylocator = mticker.FixedLocator(range(-90, 91, 30))
    gl.xlabel_style = {"size": 7, "color": CLR_TEXT_LIGHT}
    gl.ylabel_style = {"size": 7, "color": CLR_TEXT_LIGHT}

    # Map border
    for spine in ax.spines.values():
        spine.set_edgecolor("#aaaaaa")
        spine.set_linewidth(0.5)

    return ax


def render_di(tif_path: Path, out_dir: Path):
    """Type 2: Continuous Index map with lajolla colormap + horizontal colorbar."""
    title = TITLE_MAP.get(tif_path.stem, tif_path.stem)

    ds = rasterio.open(tif_path)
    di = ds.read(1).astype(np.float64)
    di[di == 0] = np.nan

    fig = plt.figure(figsize=(14, 7))
    ax = setup_map(fig, 111, ds.bounds)

    cmap = cmc.lajolla.copy()
    cmap.set_bad("none")

    rio_show(
        di,
        transform=ds.transform,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=1.5,
        zorder=1,
    )
    ds.close()

    # Horizontal colorbar below the map
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1.5))
    cax = fig.add_axes([0.20, 0.08, 0.60, 0.025])  # [left, bottom, width, height]
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="max")
    cb.set_label(
        "Dissimilarity Index (DI)",
        fontsize=9,
        color=CLR_TEXT,
        labelpad=6,
    )
    cb.ax.tick_params(labelsize=8, colors=CLR_TEXT_LIGHT)
    cb.outline.set_edgecolor("#aaaaaa")
    cb.outline.set_linewidth(0.5)

    # Threshold marker
    cb.ax.axvline(
        AOA_THRESHOLD,
        color="black",
        linewidth=1,
        ymin=0,
        ymax=1,
    )
    cb.ax.text(
        AOA_THRESHOLD,
        -0.6,
        f"threshold = {AOA_THRESHOLD:.3f}",
        transform=cb.ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7,
        fontstyle="italic",
        color=CLR_TEXT,
    )

    ax.set_title(
        f"Dissimilarity Index \u2014 {title}",
        color=CLR_TEXT,
    )

    stem = tif_path.stem
    fig.savefig(out_dir / f"{stem}_DI.png", dpi=300)
    fig.savefig(out_dir / f"{stem}_DI.pdf")
    plt.close(fig)


def render_aoa(tif_path: Path, out_dir: Path):
    """Type 1: Binary Classification map with categorical legend."""
    title = TITLE_MAP.get(tif_path.stem, tif_path.stem)

    ds = rasterio.open(tif_path)
    di = ds.read(1).astype(np.float64)
    aoa = ds.read(2).astype(np.float64)
    di[di == 0] = np.nan
    aoa[np.isnan(di)] = np.nan
    inside_pct = np.nanmean(aoa) * 100

    fig = plt.figure(figsize=(14, 7))
    ax = setup_map(fig, 111, ds.bounds)

    cmap = mcolors.ListedColormap([CLR_AOA_OUTSIDE, CLR_AOA_INSIDE])
    cmap.set_bad("none")

    rio_show(
        aoa,
        transform=ds.transform,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=1,
        zorder=1,
    )
    ds.close()

    # Categorical legend — lower left
    patch_inside = mpatches.Patch(
        facecolor=CLR_AOA_INSIDE,
        edgecolor=CLR_BORDER,
        linewidth=0.5,
        label=f"Inside AOA ({inside_pct:.1f}%)",
    )
    patch_outside = mpatches.Patch(
        facecolor=CLR_AOA_OUTSIDE,
        edgecolor=CLR_BORDER,
        linewidth=0.5,
        label=f"Outside AOA ({100 - inside_pct:.1f}%)",
    )
    leg = ax.legend(
        handles=[patch_inside, patch_outside],
        loc="lower left",
        fancybox=True,
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor=CLR_BORDER,
        fontsize=8,
        handlelength=1.2,
        handleheight=1.0,
        borderpad=0.6,
    )
    leg.get_frame().set_linewidth(0.5)

    ax.set_title(
        f"Area of Applicability \u2014 {title}",
        color=CLR_TEXT,
    )

    stem = tif_path.stem
    fig.savefig(out_dir / f"{stem}_AOA.png", dpi=300)
    fig.savefig(out_dir / f"{stem}_AOA.pdf")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────


def main():
    maps_dir = Path("outputs/aoa/test_aoa/daily/maps")
    out_dir = Path("outputs/aoa/test_aoa/daily/maps_png")
    out_dir.mkdir(parents=True, exist_ok=True)

    tifs = sorted(maps_dir.rglob("*.tif"))
    print(f"Generating publication maps for {len(tifs)} GeoTIFFs...")

    for tif in tifs:
        rel = tif.relative_to(maps_dir)
        sub_dir = out_dir / rel.parent
        sub_dir.mkdir(parents=True, exist_ok=True)

        render_di(tif, sub_dir)
        render_aoa(tif, sub_dir)
        print(f"  {rel} \u2192 DI + AOA")

    print(f"\nAll maps saved to {out_dir}/")


if __name__ == "__main__":
    main()

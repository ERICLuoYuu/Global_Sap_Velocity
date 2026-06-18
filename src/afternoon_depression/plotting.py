"""Figures for the afternoon-depression analysis (Liu et al. 2024 Fig 1 / Fig 2 analogs).

Matplotlib-only, Agg backend, saved as PNG. House style follows
``src/Analyzers/explore_relationship_observations.py`` (PFT colours, dpi, layout).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Aridity index (P/PET) classes, after Liu et al. (2024) §2.3.
_ARIDITY_EDGES = [0.0, 0.05, 0.2, 0.5, 0.75, 1.2, np.inf]
_ARIDITY_LABELS = ["hyper-arid", "arid", "semi-arid", "sub-humid", "humid", "hyper-humid"]

_DRIVERS = [("vpd", "VPD (kPa)"), ("tair", "Tair (°C)"), ("sm", "SM 0–100 cm (m³/m³)")]
_EFFECTS = ["vpd_given_tair", "tair_given_vpd", "vpd_given_sm", "sm_given_vpd"]
_EFFECT_LABELS = ["VPD|Tair", "Tair|VPD", "VPD|SM", "SM|VPD"]

# Liu et al. (2024) Fig 1b/e/h fine-resolution response bins: VPD every 0.1 kPa,
# Tair every 1 °C. SM has no stated width in the paper; 0.02 m³/m³ gives a
# comparable ~25–30-bin resolution over the observed 0–0.6 m³/m³ range.
_DRIVER_BIN_WIDTH = {"vpd": 0.1, "tair": 1.0, "sm": 0.02}
_RESPONSE_COLOR = "#1b7837"


def aridity_class(values: pd.Series) -> pd.Series:
    """Map aridity index (P/PET) to Liu's six classes."""
    return pd.cut(pd.to_numeric(values, errors="coerce"), bins=_ARIDITY_EDGES, labels=_ARIDITY_LABELS)


# Asymptotic standard error of the median is ~1.2533× that of the mean (√(π/2)),
# so the band stays an honest precision estimate when agg="median".
_MEDIAN_SE_FACTOR = 1.2533


def _fixed_width_binned(
    x: pd.Series, y: pd.Series, width: float, min_count: int = 20, agg: str = "mean"
) -> pd.DataFrame:
    """Central response per fixed-width driver bin, with the SE of that statistic.

    ``agg`` selects the central estimate: ``"mean"`` (default) or ``"median"``.
    Fixed-width bins (vs deciles) keep each bin narrow, so within-bin driver
    variation — and the systematic ΔSF trend it carries — no longer inflates the
    band. The band is the standard error of the central estimate (σ/√n for the
    mean, ×1.2533 for the median), not ±1 SD, so it reflects the precision of the
    binned statistic rather than ΔSF's (very wide) sample spread. The median is
    robust to ΔSF's heavy −900 %-bounded tail, which otherwise drags sparse-bin
    means far negative. Bins with fewer than ``min_count`` observations are dropped.
    Columns: ``x`` (bin centre), ``y_center``, ``y_se``, ``n``.
    """
    d = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if d.empty or width <= 0:
        return pd.DataFrame(columns=["x", "y_center", "y_se", "n"])
    centers = (np.floor(d["x"] / width) + 0.5) * width
    g = d.groupby(centers)["y"].agg(["mean", "median", "std", "count"])
    g.index.name = "x"
    g = g.reset_index().rename(columns={"count": "n"})
    g = g[g["n"] >= min_count].copy()
    g["y_center"] = g["median"] if agg == "median" else g["mean"]
    se_factor = _MEDIAN_SE_FACTOR if agg == "median" else 1.0
    g["y_se"] = se_factor * g["std"] / np.sqrt(g["n"])
    return g[["x", "y_center", "y_se", "n"]]


def plot_driver_responses(
    table: pd.DataFrame,
    out_path: Path,
    bin_widths: dict[str, float] | None = None,
    min_count: int = 20,
    title_note: str = "",
    agg: str = "mean",
    resp_label: str = "ΔSF",
    metric_name: str = "transpiration",
) -> None:
    """Fig 1b/e/h analog: ΔSF and diurnal centroid vs fixed-width-binned VPD/Tair/SM.

    Fine fixed-width bins (VPD 0.1 kPa, Tair 1 °C) with a central ± SE band reproduce
    Liu's clean response curves, instead of the few wide deciles whose within-bin
    trend made the ±1 SD band span hundreds of percent. ``agg="median"`` swaps the
    mean for the (tail-robust) median per bin.
    """
    widths = bin_widths or _DRIVER_BIN_WIDTH
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for col, (driver, label) in enumerate(_DRIVERS):
        if driver not in table.columns:
            continue
        width = widths.get(driver, 0.0)
        for row, (resp, ylab) in enumerate([("delta_sf", f"{resp_label} (%)"), ("centroid", "C_SF (h)")]):
            ax = axes[row, col]
            stats = _fixed_width_binned(table[driver], table[resp], width, min_count, agg=agg)
            if not stats.empty:
                ax.plot(stats["x"], stats["y_center"], "-", marker="o", ms=3, color=_RESPONSE_COLOR)
                ax.fill_between(
                    stats["x"],
                    stats["y_center"] - stats["y_se"],
                    stats["y_center"] + stats["y_se"],
                    color=_RESPONSE_COLOR,
                    alpha=0.25,
                    linewidth=0,
                )
            ax.set_xlabel(label)
            ax.set_ylabel(ylab)
            if row == 1:
                ax.axhline(12.0, ls="--", color="grey", lw=1)  # solar noon reference
    vpd_w = widths.get("vpd", 0.0)
    tair_w = widths.get("tair", 0.0)
    base = (
        f"Diurnal {metric_name} metrics vs climate drivers "
        f"(bins: VPD {vpd_w:g} kPa, Tair {tair_w:g} °C; band = {agg} ± SE)"
    )
    fig.suptitle(f"{base}{title_note}", fontsize=12, fontweight="bold")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out_path)


def _percentile_labels(n_bins: int) -> list[str]:
    """Percentile-range tick labels, e.g. n_bins=10 → ['0–10', '10–20', …, '90–100']."""
    return [f"{round(i * 100 / n_bins)}–{round((i + 1) * 100 / n_bins)}" for i in range(n_bins)]


def per_site_percentile_grid(
    table: pd.DataFrame,
    response: str = "delta_sf",
    n_bins: int = 10,
    site_col: str = "site_name",
    min_valid_days: int = 120,
    agg: str = "mean",
) -> pd.DataFrame:
    """Central ``response`` over the VPD × Tair *per-site percentile* bin grid.

    Each site's VPD and Tair are binned into its own deciles (Liu SI Text S2,
    "per pixel"), so a bin means "this site's driest/hottest decile", not an
    absolute value. ΔSF is averaged within each site's (vpd_bin, tair_bin) cell,
    then across sites. Cells that no site populates stay NaN (rendered grey); no
    site-count threshold is applied — matching Liu, who states none.

    Sites with fewer than ``min_valid_days`` valid records are dropped first — the
    SAME record-length filter as the quantitative ``decouple_all_sites`` — so the
    figure and the verdict use the identical site set, and short-record sites (down
    to a handful of days) can't inject unstable per-site deciles into the lines.

    Because VPD and Tair are correlated *within* a site (daily r≈0.78), the
    off-diagonal corners (high-VPD-pct × low-Tair-pct and vice-versa) are populated
    by only a handful of sites (or none — rendered grey), while the diagonal band
    carries ~all of them, recovering Liu's diagonal band (Fig 2c).
    Returns an ``n_bins × n_bins`` frame indexed by VPD bin (rows), Tair bin (cols).
    """
    d = table.dropna(subset=["vpd", "tair", response]).copy()
    if min_valid_days > 0:
        n_per_site = d.groupby(site_col)[response].transform("size")
        d = d[n_per_site >= min_valid_days].copy()
    d["vb"] = d.groupby(site_col)["vpd"].transform(lambda s: pd.qcut(s, q=n_bins, labels=False, duplicates="drop"))
    d["tb"] = d.groupby(site_col)["tair"].transform(lambda s: pd.qcut(s, q=n_bins, labels=False, duplicates="drop"))
    d = d.dropna(subset=["vb", "tb"])
    stat = "median" if agg == "median" else "mean"
    site_cell = d.groupby([site_col, "vb", "tb"])[response].agg(stat).reset_index()
    grid = site_cell.groupby(["vb", "tb"])[response].agg(stat).unstack("tb")
    idx = [float(i) for i in range(n_bins)]
    return grid.reindex(index=idx, columns=idx)


def plot_decoupling_lines(
    table: pd.DataFrame,
    out_path: Path,
    n_bins: int = 10,
    site_col: str = "site_name",
    min_valid_days: int = 120,
    title_note: str = "",
    agg: str = "mean",
    response: Literal["delta_sf", "centroid"] = "delta_sf",
    resp_label: str = "ΔSF",
) -> None:
    """Fig 2a–c analog: ΔSF vs VPD binned by Tair, vs Tair binned by VPD, and the 2-D grid.

    Drivers are binned into ``n_bins`` *per-site* percentile bins (Liu SI Text S2,
    "per pixel"), so axes are percentile bins (0–10th … 90–100th), not absolute
    values. Only sites with ≥ ``min_valid_days`` records contribute (same set as the
    verdict). The within-site VPD–Tair coupling leaves the off-diagonal corners
    empty (Liu Fig 2c: high VPD never co-occurs with low Tair at a site).
    """
    grid = per_site_percentile_grid(table, response, n_bins, site_col, min_valid_days, agg=agg)
    labels = _percentile_labels(n_bins)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), constrained_layout=True)
    cmap = plt.get_cmap("viridis")

    # (a) ΔSF vs VPD percentile, one line per Tair percentile bin (gaps where empty).
    for tb in range(n_bins):
        col = grid[float(tb)]
        axes[0].plot(range(n_bins), col.to_numpy(), marker="o", color=cmap(tb / max(n_bins - 1, 1)), label=labels[tb])
    axes[0].set(
        xlabel="VPD bin (within-site percentile)", ylabel=f"{resp_label} (%)", title=f"{resp_label} vs VPD, by Tair"
    )
    axes[0].set_xticks(range(n_bins))
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes[0].legend(fontsize=6, ncol=2, title="Tair pct")

    # (b) ΔSF vs Tair percentile, one line per VPD percentile bin.
    for vb in range(n_bins):
        row = grid.loc[float(vb)]
        axes[1].plot(range(n_bins), row.to_numpy(), marker="o", color=cmap(vb / max(n_bins - 1, 1)), label=labels[vb])
    axes[1].set(
        xlabel="Tair bin (within-site percentile)", ylabel=f"{resp_label} (%)", title=f"{resp_label} vs Tair, by VPD"
    )
    axes[1].set_xticks(range(n_bins))
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes[1].legend(fontsize=6, ncol=2, title="VPD pct")

    # (c) 2-D mean ΔSF over the per-site percentile grid; empty corners render grey.
    axes[2].set_facecolor("#e6e6e6")  # NaN cells (impossible VPD×Tair combos) show through
    im = axes[2].imshow(grid.to_numpy(), origin="lower", aspect="auto", cmap="YlOrRd")
    axes[2].set(
        xlabel="Tair bin (within-site percentile)",
        ylabel="VPD bin (within-site percentile)",
        title=f"mean {resp_label} (VPD × Tair percentile)",
    )
    axes[2].set_xticks(range(n_bins))
    axes[2].set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    axes[2].set_yticks(range(n_bins))
    axes[2].set_yticklabels(labels, fontsize=6)
    fig.colorbar(im, ax=axes[2], label=f"{resp_label} (%)")
    if title_note:
        fig.suptitle(title_note.strip(" —"), fontsize=12, fontweight="bold")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out_path)


def plot_effect_distributions(
    effects: pd.DataFrame, out_path: Path, resp_label: str = "ΔSF", metric_name: str = "transpiration"
) -> None:
    """Fig 2d analog: per-site distributions of the four decoupled effects."""
    cols = [c for c in _EFFECTS if c in effects.columns]
    data = [effects[c].dropna().to_numpy() for c in cols]
    labels = [_EFFECT_LABELS[_EFFECTS.index(c)] for c in cols]
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    # Box = median + 25/75 (Liu et al. 2024 Fig 2 spec); showfliers=False keeps the
    # ΔSF divide-by-near-zero tail out of the rendered axes.
    ax.boxplot(data, tick_labels=labels, showmeans=True, showfliers=False)
    ax.axhline(0, ls="--", color="grey", lw=1)
    ax.set_ylabel(f"Decoupled effect on {resp_label} (%)")
    ax.set_title(f"Independent driver effects on afternoon depression of {metric_name}")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out_path)


def plot_effects_by_group(effects: pd.DataFrame, group_col: str, out_path: Path, resp_label: str = "ΔSF") -> None:
    """Fig 2g/h analog: VPD|Tair vs Tair|VPD distributions across a grouping column (PFT or aridity)."""
    if group_col not in effects.columns:
        return
    groups = [g for g in effects[group_col].dropna().unique()]
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 1.2), 6), constrained_layout=True)
    width = 0.38
    for k, (eff, colr, lab) in enumerate(
        [("vpd_given_tair", "#d73027", "VPD|Tair"), ("tair_given_vpd", "#4575b4", "Tair|VPD")]
    ):
        positions = np.arange(len(groups)) + (k - 0.5) * width
        data = [effects.loc[effects[group_col] == g, eff].dropna().to_numpy() for g in groups]
        ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.9,
            patch_artist=True,
            showfliers=False,  # hide the ΔSF divide-by-near-zero tail (median + 25/75 box, Liu Fig 2)
            boxprops=dict(facecolor=colr, alpha=0.6),
            medianprops=dict(color="black"),
        )
        ax.plot([], [], color=colr, label=lab)
    ax.axhline(0, ls="--", color="grey", lw=1)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([str(g) for g in groups], rotation=45, ha="right")
    ax.set_ylabel(f"Decoupled effect on {resp_label} (%)")
    ax.set_title(f"VPD vs Tair limitation by {group_col}")
    ax.legend()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out_path)

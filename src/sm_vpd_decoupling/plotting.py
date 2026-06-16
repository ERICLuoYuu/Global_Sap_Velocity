# src/sm_vpd_decoupling/plotting.py
"""Figures for the SM-VPD decoupling analysis (matplotlib, Agg-safe).

All figures honour the 2-D nested-binning structure of the Liu et al. (2020)
estimator: the soil-moisture effect is read WITHIN VPD bins and the VPD effect
WITHIN soil-moisture bins. Labels use the real variable names passed in (the
response column, e.g. ``E_norm``/``Gc_norm``; the SM-variant column, e.g.
``swvl1``/``root_zone_sm``; and ``vpd``) rather than generic placeholders.
"""

from __future__ import annotations

import logging
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.sm_vpd_decoupling.decoupling import MIN_BIN_COUNT, assign_percentile_bins

logger = logging.getLogger(__name__)

_SM_COLOR = "tab:blue"
_VPD_COLOR = "tab:orange"


def _grid_cell_means(df, sm_col, response, n_bins, vpd_col="vpd"):
    """2-D grid (rows = VPD percentile bin, cols = SM percentile bin) of the mean
    response. Cells with < ``MIN_BIN_COUNT`` records are NaN. Reindexed to the
    full ``0..n_bins-1`` range on both axes so the grid is regular."""
    d = df.dropna(subset=[sm_col, vpd_col, response]).copy()
    d["_sm"] = assign_percentile_bins(d[sm_col], n_bins)
    d["_vpd"] = assign_percentile_bins(d[vpd_col], n_bins)
    d = d.dropna(subset=["_sm", "_vpd"])
    grp = d.groupby(["_vpd", "_sm"])[response]
    mean = grp.mean()[grp.size() >= MIN_BIN_COUNT]
    grid = mean.unstack("_sm")
    full = list(range(n_bins))
    return grid.reindex(index=full, columns=full)


def _hi_lo(series):
    """(value at largest non-NaN index, value at smallest non-NaN index)."""
    s = series.dropna()
    if len(s) < 2:
        return np.nan, np.nan
    return float(s.loc[s.index.max()]), float(s.loc[s.index.min()])


def _sm_effect_per_vpd(grid):
    """Per VPD row: low-SM minus high-SM response (SM stressor convention)."""
    out = []
    for r in grid.index:
        hi, lo = _hi_lo(grid.loc[r])  # hi = high SM bin, lo = low SM bin
        out.append(lo - hi)
    return out


def _vpd_effect_per_sm(grid):
    """Per SM column: high-VPD minus low-VPD response."""
    out = []
    for c in grid.columns:
        hi, lo = _hi_lo(grid[c])  # hi = high VPD bin, lo = low VPD bin
        out.append(hi - lo)
    return out


def select_example_sites(table, n=4, sm_col="root_zone_sm"):
    """Pick up to ``n`` most-data sites spread across the aridity gradient.

    Bins sites by aridity (if present) and picks the most-data site per bin so
    the example panel shows how the decoupling shape varies across climates.
    """
    counts = table.groupby("site_name").size().rename("n_days")
    meta = table.groupby("site_name")["aridity"].first() if "aridity" in table.columns else None
    if meta is None or meta.notna().sum() == 0:
        return counts.sort_values(ascending=False).head(n).index.tolist()
    df = pd.concat([counts, meta], axis=1).dropna(subset=["aridity"])
    df["arid_bin"] = pd.qcut(df["aridity"], q=min(n, df["aridity"].nunique()), duplicates="drop")
    picks = df.sort_values("n_days", ascending=False).groupby("arid_bin", observed=True).head(1)
    return picks.sort_values("aridity").index.tolist()


def _plot_within_bin(ax_sm, ax_vpd, grid, response, sm_col, title_prefix=""):
    """Left axis: response vs SM bin, one line per VPD bin (= {sm_col} | vpd).
    Right axis: response vs VPD bin, one line per SM bin (= vpd | {sm_col})."""
    n = max(grid.shape[0], grid.shape[1])
    cmap = plt.get_cmap("viridis")
    denom = max(1, n - 1)
    for i, r in enumerate(grid.index):
        ax_sm.plot(grid.columns, grid.loc[r].values, "o-", color=cmap(i / denom), label=f"{int(r)}")
    ax_sm.set_xlabel(f"{sm_col} percentile bin")
    ax_sm.set_ylabel(response)
    ax_sm.set_title(f"{title_prefix}{response} vs {sm_col} | vpd", fontsize=9)
    for j, c in enumerate(grid.columns):
        ax_vpd.plot(grid.index, grid[c].values, "s-", color=cmap(j / denom), label=f"{int(c)}")
    ax_vpd.set_xlabel("vpd percentile bin")
    ax_vpd.set_title(f"{title_prefix}{response} vs vpd | {sm_col}", fontsize=9)


def plot_example_sites(table, sites, response, sm_col, n_bins, out_path):
    """(a1) Per-site 2-D decoupling as within-bin lines: response vs SM bin with
    one line per VPD bin (= {sm_col} | vpd), and response vs VPD bin with one
    line per SM bin (= vpd | {sm_col})."""
    sites = list(sites) or [table["site_name"].iloc[0]]
    fig, axes = plt.subplots(len(sites), 2, figsize=(9.5, 3.2 * len(sites)), squeeze=False)
    for r, site in enumerate(sites):
        g = table[table["site_name"] == site].dropna(subset=[response, sm_col, "vpd"])
        grid = _grid_cell_means(g, sm_col, response, n_bins)
        _plot_within_bin(axes[r][0], axes[r][1], grid, response, sm_col, title_prefix=f"{site}: ")
    axes[0][0].legend(title="vpd bin", fontsize=7, ncol=2)
    axes[0][1].legend(title=f"{sm_col} bin", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cross_site_aggregate(table, response, sm_col, n_bins, out_path):
    """(a2) Cross-site mean 2-D decoupling grid (heatmap) with marginal effect
    bars: ``{sm_col} | vpd`` per VPD row (right) and ``vpd | {sm_col}`` per SM
    column (top)."""
    grids = []
    for _site, g in table.groupby("site_name"):
        gg = g.dropna(subset=[response, sm_col, "vpd"])
        if len(gg) < MIN_BIN_COUNT * 2:
            continue
        grids.append(_grid_cell_means(gg, sm_col, response, n_bins).values)

    fig = plt.figure(figsize=(7.5, 6.8))
    if not grids:
        fig.text(0.5, 0.5, "no populated cells", ha="center")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_grid = pd.DataFrame(np.nanmean(np.dstack(grids), axis=2))  # rows = vpd, cols = sm

    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1.1], height_ratios=[1.1, 4], hspace=0.08, wspace=0.08)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    im = ax_main.imshow(mean_grid.values, origin="lower", aspect="auto", cmap="viridis")
    ax_main.set_xlabel(f"{sm_col} percentile bin")
    ax_main.set_ylabel("vpd percentile bin")
    fig.colorbar(im, ax=ax_main, orientation="horizontal", fraction=0.046, pad=0.16, label=f"mean {response}")

    vpd_eff = _vpd_effect_per_sm(mean_grid)
    ax_top.bar(range(len(vpd_eff)), vpd_eff, color=_VPD_COLOR)
    ax_top.axhline(0, color="grey", lw=0.6)
    ax_top.set_xticks([])
    ax_top.set_ylabel(f"vpd | {sm_col}", fontsize=8)
    ax_top.set_title(f"{response}: {sm_col} x vpd decoupling grid", fontsize=10)

    sm_eff = _sm_effect_per_vpd(mean_grid)
    ax_right.barh(range(len(sm_eff)), sm_eff, color=_SM_COLOR)
    ax_right.axvline(0, color="grey", lw=0.6)
    ax_right.set_yticks([])
    ax_right.set_xlabel(f"{sm_col} | vpd", fontsize=8)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_lines(table, response, sm_col, n_bins, out_path):
    """(a2-lines) Cross-site MEAN 2-D decoupling rendered as within-bin lines
    (mean +/- SE across sites per cell): response vs SM bin with one line per VPD
    bin (= {sm_col} | vpd), and response vs VPD bin with one line per SM bin
    (= vpd | {sm_col}). The line analogue of the aggregate heatmap."""
    grids = []
    for _site, g in table.groupby("site_name"):
        gg = g.dropna(subset=[response, sm_col, "vpd"])
        if len(gg) < MIN_BIN_COUNT * 2:
            continue
        grids.append(_grid_cell_means(gg, sm_col, response, n_bins).values)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    if not grids:
        fig.text(0.5, 0.5, "no populated cells", ha="center")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    stack = np.dstack(grids)  # axis0 = vpd bin, axis1 = sm bin, axis2 = site
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(stack, axis=2)
        n_obs = np.sum(~np.isnan(stack), axis=2)
        se = np.nanstd(stack, axis=2) / np.sqrt(np.maximum(n_obs, 1))

    nb = mean.shape[0]
    xs = np.arange(nb)
    cmap = plt.get_cmap("viridis")
    denom = max(1, nb - 1)

    # left: response vs SM bin, one line per VPD bin (rows of the grid)
    for i in range(nb):
        c = cmap(i / denom)
        axes[0].plot(xs, mean[i, :], "o-", color=c, label=f"{i}")
        axes[0].fill_between(xs, mean[i, :] - se[i, :], mean[i, :] + se[i, :], color=c, alpha=0.15)
    axes[0].set_xlabel(f"{sm_col} percentile bin")
    axes[0].set_ylabel(f"{response} (cross-site mean)")
    axes[0].set_title(f"{response} vs {sm_col} | vpd  (mean +/- SE across sites)", fontsize=9)
    axes[0].legend(title="vpd bin", fontsize=7, ncol=2)

    # right: response vs VPD bin, one line per SM bin (columns of the grid)
    for j in range(nb):
        c = cmap(j / denom)
        axes[1].plot(xs, mean[:, j], "s-", color=c, label=f"{j}")
        axes[1].fill_between(xs, mean[:, j] - se[:, j], mean[:, j] + se[:, j], color=c, alpha=0.15)
    axes[1].set_xlabel("vpd percentile bin")
    axes[1].set_title(f"{response} vs vpd | {sm_col}  (mean +/- SE across sites)", fontsize=9)
    axes[1].legend(title=f"{sm_col} bin", fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_depth_dominance(depth_df, response, out_path):
    """(b) % of sites where SM dominates, across SM depth variants."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(depth_df["sm_variant"], depth_df["pct_sm_dominant"])
    for x in range(len(depth_df)):
        pct = float(depth_df["pct_sm_dominant"].iloc[x])
        n = depth_df["n_sites"].iloc[x]
        ax.text(x, pct + 1, f"n={n}", ha="center", fontsize=8)
    ax.axhline(50, color="grey", ls=":")
    ax.set_ylabel("% sites SM-dominant")
    ax.set_title(f"soil-moisture vs vpd dominance by depth ({response})")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _require_legs(df, where):
    for col in ("sm_given_vpd", "vpd_given_sm"):
        if col not in df.columns:
            raise ValueError(f"{where} missing required leg column: {col}")


def plot_gradient_groups(effects, group_col, response, sm_col, out_path):
    """(c) Per-group distribution of BOTH decoupled legs: ``{sm_col} | vpd``
    (sm_given_vpd) and ``vpd | {sm_col}`` (vpd_given_sm)."""
    _require_legs(effects, "effects")
    df = effects.dropna(subset=[group_col]).copy()
    if group_col in ("aridity", "canopy_height"):
        df[group_col] = pd.cut(df[group_col], bins=5).astype(str)
    groups = sorted(str(g) for g in df[group_col].dropna().unique())
    fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.4), 4.5))
    for i, gname in enumerate(groups):
        sub = df[df[group_col].astype(str) == gname]
        sm = sub["sm_given_vpd"].dropna().values
        vp = sub["vpd_given_sm"].dropna().values
        if len(sm):
            ax.boxplot(
                sm,
                positions=[i - 0.2],
                widths=0.34,
                patch_artist=True,
                boxprops=dict(facecolor=_SM_COLOR, alpha=0.6),
                medianprops=dict(color="black"),
                showfliers=False,
            )
        if len(vp):
            ax.boxplot(
                vp,
                positions=[i + 0.2],
                widths=0.34,
                patch_artist=True,
                boxprops=dict(facecolor=_VPD_COLOR, alpha=0.6),
                medianprops=dict(color="black"),
                showfliers=False,
            )
    ax.axhline(0, color="grey", ls=":")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylabel(f"{response} effect (high-low)")
    ax.set_title(f"{response}: decoupled effects by {group_col}  ({sm_col})")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_SM_COLOR, alpha=0.6),
        plt.Rectangle((0, 0), 1, 1, color=_VPD_COLOR, alpha=0.6),
    ]
    ax.legend(handles, [f"{sm_col} | vpd", f"vpd | {sm_col}"], fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_leg_comparison_box(combined, response, out_path):
    """Signed paired boxplots across sites: sm_given_vpd (``soil moisture | vpd``)
    vs vpd_given_sm (``vpd | soil moisture``) at each SM variant. ``combined``
    has columns [sm_variant, sm_given_vpd, vpd_given_sm]."""
    _require_legs(combined, "combined")
    if "sm_variant" not in combined.columns:
        raise ValueError("combined missing required column: sm_variant")
    variants = list(dict.fromkeys(combined["sm_variant"]))
    fig, ax = plt.subplots(figsize=(max(7, len(variants) * 1.5), 4.5))
    for i, v in enumerate(variants):
        sub = combined[combined["sm_variant"] == v]
        sm = sub["sm_given_vpd"].dropna().values
        vp = sub["vpd_given_sm"].dropna().values
        if len(sm):
            ax.boxplot(
                sm,
                positions=[i - 0.2],
                widths=0.34,
                patch_artist=True,
                boxprops=dict(facecolor=_SM_COLOR, alpha=0.6),
                medianprops=dict(color="black"),
                showfliers=False,
            )
        if len(vp):
            ax.boxplot(
                vp,
                positions=[i + 0.2],
                widths=0.34,
                patch_artist=True,
                boxprops=dict(facecolor=_VPD_COLOR, alpha=0.6),
                medianprops=dict(color="black"),
                showfliers=False,
            )
    ax.axhline(0, color="grey", ls=":")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=20, ha="right")
    ax.set_ylabel(f"{response} effect (per-site, high-low)")
    ax.set_title(f"{response}: soil-moisture vs vpd decoupled effects across sites")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_SM_COLOR, alpha=0.6),
        plt.Rectangle((0, 0), 1, 1, color=_VPD_COLOR, alpha=0.6),
    ]
    ax.legend(handles, ["soil moisture | vpd", "vpd | soil moisture"], fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_leg_comparison_scatter(effects, response, sm_col, out_path):
    """Per-site dominance scatter: ``|{sm_col} | vpd|`` (y) vs ``|vpd | {sm_col}|``
    (x) with the y=x line. Points above the line are SM-dominant."""
    _require_legs(effects, "effects")
    d = effects.dropna(subset=["sm_given_vpd", "vpd_given_sm"])
    x = d["vpd_given_sm"].abs().to_numpy()
    y = d["sm_given_vpd"].abs().to_numpy()
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.scatter(x, y, s=28, alpha=0.7, color=_SM_COLOR, edgecolor="k", linewidth=0.3)
    top = max(float(x.max()) if len(x) else 0.0, float(y.max()) if len(y) else 0.0, 1e-6) * 1.05
    ax.plot([0, top], [0, top], "k--", lw=1)
    n_sm = int((y > x).sum())
    n_vpd = int((x >= y).sum())
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel(f"| vpd | {sm_col} |")
    ax.set_ylabel(f"| {sm_col} | vpd |")
    ax.set_title(f"{response} ({sm_col}): SM-dominant={n_sm}, VPD-dominant={n_vpd}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

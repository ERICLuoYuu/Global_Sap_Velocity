# src/sm_vpd_decoupling/plotting.py
"""Figures for the SM-VPD decoupling analysis (matplotlib, Agg-safe)."""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.sm_vpd_decoupling.decoupling import MIN_BIN_COUNT, assign_percentile_bins

logger = logging.getLogger(__name__)


def _binned_curve(df, axis_col, response, n_bins):
    """Mean response per percentile bin of ``axis_col`` (one site)."""
    bins = assign_percentile_bins(df[axis_col], n_bins)
    g = pd.DataFrame({"bin": bins, "resp": df[response]}).dropna()
    return g.groupby("bin")["resp"].mean()


def select_example_sites(table, n=4, sm_col="root_zone_sm"):
    """Pick up to ``n`` most-data sites spread across the aridity gradient.

    Bins sites by aridity (if present) and picks the most-data site per bin so
    the (a1) panel shows how the decoupling shape varies across climates.
    """
    counts = table.groupby("site_name").size().rename("n_days")
    meta = table.groupby("site_name")["aridity"].first() if "aridity" in table.columns else None
    if meta is None or meta.notna().sum() == 0:
        return counts.sort_values(ascending=False).head(n).index.tolist()
    df = pd.concat([counts, meta], axis=1).dropna(subset=["aridity"])
    df["arid_bin"] = pd.qcut(df["aridity"], q=min(n, df["aridity"].nunique()), duplicates="drop")
    picks = df.sort_values("n_days", ascending=False).groupby("arid_bin", observed=True).head(1)
    return picks.sort_values("aridity").index.tolist()


def plot_example_sites(table, sites, response, sm_col, n_bins, out_path):
    """(a1) Per-site small-multiples: Resp-vs-VPD binned by SM, and Resp-vs-SM
    binned by VPD, for the chosen example sites."""
    sites = list(sites) or [table["site_name"].iloc[0]]
    fig, axes = plt.subplots(len(sites), 2, figsize=(9, 3.2 * len(sites)), squeeze=False)
    for r, site in enumerate(sites):
        g = table[table["site_name"] == site].dropna(subset=[response, sm_col, "vpd"])
        vpd_by_sm = _binned_curve(g, sm_col, response, n_bins)
        sm_by_vpd = _binned_curve(g, "vpd", response, n_bins)
        axes[r][0].plot(vpd_by_sm.index, vpd_by_sm.values, "o-")
        axes[r][0].set_ylabel(f"{site}\n{response}")
        axes[r][0].set_xlabel("SM percentile bin")
        axes[r][1].plot(sm_by_vpd.index, sm_by_vpd.values, "s-", color="tab:orange")
        axes[r][1].set_xlabel("VPD percentile bin")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cross_site_aggregate(table, response, sm_col, n_bins, out_path):
    """(a2) Mean & median across sites of response per SM/VPD percentile bin."""
    sm_curves, vpd_curves = [], []
    for _site, g in table.groupby("site_name"):
        gg = g.dropna(subset=[response, sm_col, "vpd"])
        if len(gg) < MIN_BIN_COUNT * n_bins:
            continue
        sm_curves.append(_binned_curve(gg, sm_col, response, n_bins))
        vpd_curves.append(_binned_curve(gg, "vpd", response, n_bins))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, curves, label in ((axes[0], sm_curves, "SM percentile bin"), (axes[1], vpd_curves, "VPD percentile bin")):
        if curves:
            mat = pd.concat(curves, axis=1)
            ax.plot(mat.index, mat.mean(axis=1), "o-", label="mean")
            ax.plot(mat.index, mat.median(axis=1), "s--", label="median")
        ax.set_xlabel(label)
        ax.set_ylabel(f"{response} (normalized)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_depth_dominance(depth_df, response, out_path):
    """(b) % of sites where SM dominates, across SM depth variants."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(depth_df["sm_variant"], depth_df["pct_sm_dominant"])
    for x, (pct, n) in enumerate(zip(depth_df["pct_sm_dominant"], depth_df["n_sites"])):
        ax.text(x, pct + 1, f"n={n}", ha="center", fontsize=8)
    ax.axhline(50, color="grey", ls=":")
    ax.set_ylabel("% sites SM-dominant")
    ax.set_title(f"SM-vs-VPD dominance by depth ({response})")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_gradient_violins(effects, group_col, out_path):
    """(c) Violin of sm_given_vpd grouped by a categorical/binned column."""
    df = effects.dropna(subset=["sm_given_vpd", group_col]).copy()
    if group_col in ("aridity", "canopy_height"):
        df[group_col] = pd.cut(df[group_col], bins=5).astype(str)
    groups = sorted(df[group_col].unique())
    data = [df.loc[df[group_col] == gname, "sm_given_vpd"].values for gname in groups]
    fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.2), 4))
    if any(len(d) > 0 for d in data):
        ax.violinplot([d for d in data if len(d) > 0], showmedians=True)
        ax.set_xticks(range(1, len([d for d in data if len(d) > 0]) + 1))
        ax.set_xticklabels([g for g, d in zip(groups, data) if len(d) > 0], rotation=30, ha="right")
    ax.axhline(0, color="grey", ls=":")
    ax.set_ylabel("ΔResp(SM|VPD)")
    ax.set_title(f"SM limitation by {group_col}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

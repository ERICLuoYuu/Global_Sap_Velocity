# src/sm_vpd_decoupling/tests/test_plotting.py
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from src.sm_vpd_decoupling.plotting import (
    plot_cross_site_aggregate,
    plot_depth_dominance,
    plot_example_sites,
    plot_gradient_groups,
    plot_leg_comparison_box,
    plot_leg_comparison_scatter,
    select_example_sites,
)


def _table(seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for site in ["A", "B"]:
        sm = rng.uniform(0.1, 0.35, 600)
        vpd = rng.uniform(0.5, 2.5, 600)
        frames.append(
            pd.DataFrame(
                {
                    "site_name": site,
                    "vpd": vpd,
                    "swvl1": sm,
                    "E_norm": (sm - 0.1) / 0.25 + 0.1 * (vpd - 0.5),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _effects(seed=0, n=30):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "site_name": [f"s{i}" for i in range(n)],
            "sm_given_vpd": rng.normal(-0.15, 0.08, n),
            "vpd_given_sm": rng.normal(0.10, 0.08, n),
            "pft": (["ENF"] * (n // 2)) + (["DBF"] * (n - n // 2)),
            "biome": ["temperate"] * n,
            "aridity": rng.uniform(0.1, 1.5, n),
            "canopy_height": rng.uniform(5, 35, n),
        }
    )


def test_plot_cross_site_aggregate_heatmap(tmp_path):
    out = tmp_path / "grid.png"
    plot_cross_site_aggregate(_table(), response="E_norm", sm_col="swvl1", n_bins=5, out_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_select_and_plot_example_sites(tmp_path):
    table = _table()
    table["aridity"] = np.where(table["site_name"] == "A", 0.3, 1.2)
    sites = select_example_sites(table, n=2)
    assert 1 <= len(sites) <= 2
    out = tmp_path / "examples.png"
    plot_example_sites(table, sites=sites, response="E_norm", sm_col="swvl1", n_bins=5, out_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_depth_dominance(tmp_path):
    depth = pd.DataFrame(
        {
            "sm_variant": ["swvl1", "swvl2", "swvl3", "swvl4", "root_zone_sm"],
            "pct_sm_dominant": [60.0, 65.0, 70.0, 72.0, 68.0],
            "n_sites": [40, 40, 40, 38, 40],
        }
    )
    out = tmp_path / "depth.png"
    plot_depth_dominance(depth, response="E_norm", out_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_gradient_groups_both_legs(tmp_path):
    out = tmp_path / "grad.png"
    plot_gradient_groups(_effects(), group_col="pft", response="E_norm", sm_col="root_zone_sm", out_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_gradient_groups_requires_both_legs(tmp_path):
    eff = _effects().drop(columns=["vpd_given_sm"])  # drop the VPD leg
    with pytest.raises(ValueError):
        plot_gradient_groups(
            eff, group_col="pft", response="E_norm", sm_col="root_zone_sm", out_path=str(tmp_path / "x.png")
        )


def test_plot_leg_comparison_box(tmp_path):
    combined = pd.concat(
        [_effects(0).assign(sm_variant="swvl1"), _effects(1).assign(sm_variant="root_zone_sm")],
        ignore_index=True,
    )
    out = tmp_path / "box.png"
    plot_leg_comparison_box(combined, response="E_norm", out_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_leg_comparison_box_requires_variant(tmp_path):
    with pytest.raises(ValueError):
        plot_leg_comparison_box(_effects(), response="E_norm", out_path=str(tmp_path / "x.png"))


def test_plot_leg_comparison_scatter(tmp_path):
    out = tmp_path / "scatter.png"
    plot_leg_comparison_scatter(_effects(), response="E_norm", sm_col="root_zone_sm", out_path=str(out))
    assert out.exists() and out.stat().st_size > 0

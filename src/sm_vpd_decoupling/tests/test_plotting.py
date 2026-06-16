# src/sm_vpd_decoupling/tests/test_plotting.py
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from src.sm_vpd_decoupling.plotting import (
    plot_cross_site_aggregate,
    plot_depth_dominance,
    plot_example_sites,
    plot_gradient_violins,
    select_example_sites,
)


def _table(seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for site in ["A", "B"]:
        sm = rng.uniform(0.1, 0.35, 400)
        vpd = rng.uniform(0.5, 2.5, 400)
        frames.append(
            pd.DataFrame(
                {
                    "site_name": site,
                    "vpd": vpd,
                    "swvl1": sm,
                    "E_norm": (sm - 0.1) / 0.25,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_plot_cross_site_aggregate(tmp_path):
    out = tmp_path / "agg.png"
    plot_cross_site_aggregate(_table(), response="E_norm", sm_col="swvl1", n_bins=5, out_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_select_and_plot_example_sites(tmp_path):
    table = _table()
    # add an aridity column so selection spans a climate gradient
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


def test_plot_gradient_violins(tmp_path):
    effects = pd.DataFrame(
        {
            "sm_given_vpd": np.random.default_rng(0).normal(-0.3, 0.1, 30),
            "aridity": np.random.default_rng(1).uniform(0.1, 1.5, 30),
            "pft": (["ENF"] * 15) + (["DBF"] * 15),
            "biome": (["temperate"] * 30),
            "canopy_height": np.random.default_rng(2).uniform(5, 35, 30),
        }
    )
    out = tmp_path / "grad.png"
    plot_gradient_violins(effects, group_col="pft", out_path=str(out))
    assert out.exists() and out.stat().st_size > 0

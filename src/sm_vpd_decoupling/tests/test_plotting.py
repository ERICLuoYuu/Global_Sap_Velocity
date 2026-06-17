# src/sm_vpd_decoupling/tests/test_plotting.py
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from src.sm_vpd_decoupling.plotting import (
    _grid_cell_means,
    _hi_lo,
    _sm_effect_per_vpd,
    _vpd_effect_per_sm,
    plot_aggregate_lines,
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


def test_plot_aggregate_lines(tmp_path):
    out = tmp_path / "aggline.png"
    plot_aggregate_lines(_table(), response="E_norm", sm_col="swvl1", n_bins=5, out_path=str(out))
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


# --- helper-level tests: lock the 2-D decoupling SEMANTICS, not just file output ---
# These guard against the "plots render but mean the wrong thing" class of bug:
# smoke tests (file written) cannot catch a flipped axis or a dropped leg, but the
# pure helpers that encode the math can be asserted directly.


def _grid(rows):
    """Build a grid DataFrame: index = VPD bin, columns = SM bin."""
    n = len(rows[0])
    return pd.DataFrame(rows, index=list(range(len(rows))), columns=list(range(n)))


def test_hi_lo_keys_off_index_not_position():
    # _hi_lo returns (value at LARGEST index, value at SMALLEST index).
    hi, lo = _hi_lo(pd.Series([7.0, 9.0], index=[5, 1]))
    assert hi == 7.0  # index 5 is largest
    assert lo == 9.0  # index 1 is smallest


def test_hi_lo_needs_two_points():
    hi, lo = _hi_lo(pd.Series([np.nan, 5.0], index=[0, 1]))
    assert np.isnan(hi) and np.isnan(lo)


def test_sm_effect_is_low_minus_high_within_vpd_rows():
    # Response rises with SM (columns) and is flat across VPD (rows):
    #   SM leg = low-SM minus high-SM -> strongly NEGATIVE in every VPD row;
    #   VPD leg -> ~0 (no VPD dependence).
    sm_driven = _grid([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    assert _sm_effect_per_vpd(sm_driven) == [-2.0, -2.0, -2.0]
    assert _vpd_effect_per_sm(sm_driven) == [0.0, 0.0, 0.0]


def test_vpd_effect_is_high_minus_low_within_sm_cols():
    # Response rises with VPD (rows) and is flat across SM (columns):
    #   VPD leg = high-VPD minus low-VPD -> POSITIVE in every SM column;
    #   SM leg -> ~0.
    vpd_driven = _grid([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    assert _vpd_effect_per_sm(vpd_driven) == [2.0, 2.0, 2.0]
    assert _sm_effect_per_vpd(vpd_driven) == [0.0, 0.0, 0.0]


def test_grid_cell_means_shape_and_orientation():
    # response == swvl1 exactly -> within each VPD row, response must INCREASE
    # left (low-SM bin) to right (high-SM bin); grid is a regular n_bins x n_bins.
    rng = np.random.default_rng(3)
    n = 600
    sm = rng.uniform(0.1, 0.4, n)
    df = pd.DataFrame({"swvl1": sm, "vpd": rng.uniform(0.5, 2.5, n), "E_norm": sm})
    grid = _grid_cell_means(df, "swvl1", "E_norm", 5)
    assert isinstance(grid, pd.DataFrame)
    assert grid.shape == (5, 5)
    assert list(grid.index) == [0, 1, 2, 3, 4]  # VPD bins
    assert list(grid.columns) == [0, 1, 2, 3, 4]  # SM bins
    for r in grid.index:
        row = grid.loc[r].dropna()
        assert row.loc[row.index.min()] < row.loc[row.index.max()]  # rises with SM
    # On a realistic SM-driven grid the SM leg is negative in every populated row.
    assert all(e < 0 for e in _sm_effect_per_vpd(grid))

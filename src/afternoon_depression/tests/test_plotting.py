"""Regression tests: figures must stay readable when a few sites carry extreme
decoupled effects (the ΔSF divide-by-near-zero tail). Box plots with
``showfliers=False`` (Liu et al. 2024 Fig 2 spec: median + 25/75 box) keep the
abnormal values out of the rendered axes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.afternoon_depression.plotting import (
    _fixed_width_binned,
    per_site_percentile_grid,
    plot_decoupling_lines,
    plot_driver_responses,
    plot_effect_distributions,
    plot_effects_by_group,
)

_EFFECT_COLS = ["vpd_given_tair", "tair_given_vpd", "vpd_given_sm", "sm_given_vpd"]


def _effects_with_outlier() -> pd.DataFrame:
    rng = np.arange(1, 21, dtype=float)
    df = pd.DataFrame({c: rng + i for i, c in enumerate(_EFFECT_COLS)})
    # inject pathological extremes like the boreal divide-by-near-zero sites
    df.loc[0, "vpd_given_tair"] = 5.0e5
    df.loc[1, "sm_given_vpd"] = -4.0e5
    df["pft"] = (["ENF"] * 10) + (["EBF"] * 10)
    return df


@pytest.mark.unit
def test_effect_distributions_renders_with_extremes(tmp_path) -> None:
    out = tmp_path / "fig2d.png"
    plot_effect_distributions(_effects_with_outlier(), out)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.unit
def test_effects_by_group_renders_with_extremes(tmp_path) -> None:
    out = tmp_path / "fig2g.png"
    plot_effects_by_group(_effects_with_outlier(), "pft", out)
    assert out.exists() and out.stat().st_size > 0


# ── Fig 1: fixed-width binning + standard-error band (Liu 0.1 kPa / 1 °C) ────
def _site_day_table(n: int = 2000) -> pd.DataFrame:
    """Synthetic site-day table where ΔSF rises with VPD (the dominant driver)."""
    rng = np.random.RandomState(1)
    vpd = rng.uniform(0.2, 4.0, n)
    tair = rng.uniform(6.0, 35.0, n)
    sm = rng.uniform(0.08, 0.42, n)
    return pd.DataFrame(
        {
            "site_name": np.repeat([f"S{i}" for i in range(n // 100)], 100),
            "vpd": vpd,
            "tair": tair,
            "sm": sm,
            "delta_sf": 20.0 * vpd + rng.normal(0, 5.0, n),
            "centroid": 12.0 + 0.3 * vpd + rng.normal(0, 0.2, n),
        }
    )


@pytest.mark.unit
def test_fixed_width_binned_centers_and_se() -> None:
    # x in [0,1) → centre 0.5; x in [1,2) → centre 1.5. SE = std/sqrt(n).
    x = pd.Series([0.1, 0.9, 0.5, 1.2, 1.8])
    y = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0])
    out = _fixed_width_binned(x, y, width=1.0, min_count=1).set_index("x")
    assert set(out.index) == {0.5, 1.5}
    assert out.loc[0.5, "y_center"] == pytest.approx(2.0)  # mean(1,2,3)
    assert out.loc[0.5, "n"] == 3
    assert out.loc[0.5, "y_se"] == pytest.approx(1.0 / np.sqrt(3))  # std(1,2,3)=1


@pytest.mark.unit
def test_fixed_width_binned_median_resists_tail() -> None:
    # one bin, heavy positive outlier: median(1,2,3,99)=2.5 vs mean=26.25 → median is robust
    x = pd.Series([0.1, 0.2, 0.3, 0.4])
    y = pd.Series([1.0, 2.0, 3.0, 99.0])
    mean_out = _fixed_width_binned(x, y, width=1.0, min_count=1, agg="mean").iloc[0]
    med_out = _fixed_width_binned(x, y, width=1.0, min_count=1, agg="median").iloc[0]
    assert mean_out["y_center"] == pytest.approx(26.25)
    assert med_out["y_center"] == pytest.approx(2.5)


@pytest.mark.unit
def test_per_site_grid_median_differs_from_mean_when_skewed() -> None:
    grid_mean = per_site_percentile_grid(_multi_site(coupled=False), n_bins=5, agg="mean")
    grid_med = per_site_percentile_grid(_multi_site(coupled=False), n_bins=5, agg="median")
    # same shape/axes, but the central estimates are not identical for skewed cells
    assert grid_mean.shape == grid_med.shape == (5, 5)
    assert not np.allclose(grid_mean.to_numpy(), grid_med.to_numpy(), equal_nan=True)


@pytest.mark.unit
def test_fixed_width_binned_drops_sparse_bins() -> None:
    x = pd.Series([0.1, 0.2, 0.3, 5.0])  # the lone x=5 bin has n=1
    y = pd.Series([1.0, 2.0, 3.0, 99.0])
    out = _fixed_width_binned(x, y, width=1.0, min_count=2)
    assert (out["n"] >= 2).all()
    assert 5.5 not in set(out["x"])


@pytest.mark.unit
def test_driver_responses_renders(tmp_path) -> None:
    out = tmp_path / "fig1.png"
    plot_driver_responses(_site_day_table(), out)
    assert out.exists() and out.stat().st_size > 0


# ── Fig 2: per-site percentile binning (Liu "per pixel") + empty corners ─────
def _multi_site(coupled: bool, n_sites: int = 12, per: int = 240) -> pd.DataFrame:
    """Multi-site table; within each site Tair is either tightly coupled to VPD
    (coupled=True) or independent of it (coupled=False)."""
    rng = np.random.RandomState(7)
    frames = []
    for si in range(n_sites):
        vpd = rng.uniform(0.3, 4.0, per)
        tair = 6.0 + 7.0 * vpd + rng.normal(0, 0.05, per) if coupled else rng.uniform(6.0, 35.0, per)
        frames.append(
            pd.DataFrame(
                {"site_name": f"S{si}", "vpd": vpd, "tair": tair, "delta_sf": 10.0 * vpd + rng.normal(0, 1.0, per)}
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.mark.unit
def test_per_site_grid_corners_empty_when_drivers_coupled() -> None:
    grid = per_site_percentile_grid(_multi_site(coupled=True), n_bins=10)
    # high-VPD-pct × low-Tair-pct (and the mirror) cannot co-occur within a site
    assert np.isnan(grid.loc[9.0, 0.0]) and np.isnan(grid.loc[0.0, 9.0])
    # the diagonal (where VPD and Tair percentiles coincide) is populated
    assert np.isfinite(grid.loc[0.0, 0.0]) and np.isfinite(grid.loc[9.0, 9.0])


@pytest.mark.unit
def test_per_site_grid_corners_fill_when_drivers_independent() -> None:
    grid = per_site_percentile_grid(_multi_site(coupled=False), n_bins=10)
    assert np.isfinite(grid.loc[9.0, 0.0]) and np.isfinite(grid.loc[0.0, 9.0])


@pytest.mark.unit
def test_per_site_grid_axes_are_percentile_bins() -> None:
    grid = per_site_percentile_grid(_multi_site(coupled=True), n_bins=10)
    assert list(grid.index) == [float(i) for i in range(10)]
    assert list(grid.columns) == [float(i) for i in range(10)]


@pytest.mark.unit
def test_per_site_grid_respects_min_valid_days() -> None:
    # Guards the (table, response, n_bins, site_col, min_valid_days, agg) positional
    # contract that plot_decoupling_lines relies on: a 200-day site is kept below its
    # record length and dropped above it. A mis-wired arg would break this.
    df = _multi_site(coupled=False, n_sites=1, per=200)
    populated = per_site_percentile_grid(df, n_bins=5, min_valid_days=100)
    dropped = per_site_percentile_grid(df, n_bins=5, min_valid_days=300)
    assert np.isfinite(populated.to_numpy()).any()
    assert np.isnan(dropped.to_numpy()).all()


@pytest.mark.unit
def test_decoupling_lines_renders_with_ten_bins(tmp_path) -> None:
    out = tmp_path / "fig2abc.png"
    plot_decoupling_lines(_multi_site(coupled=True), out, n_bins=10)
    assert out.exists() and out.stat().st_size > 0

"""Smoke tests for the Fu-style figures (Task 5): each writes a non-empty PNG."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.fu_ann_sensitivity.plotting import (
    plot_dual_legs,
    plot_pft_panels,
    plot_sensitivity_heatmap,
)


def _leg_table(n_bins=5, axis_name="sm_bin"):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            axis_name: np.arange(n_bins),
            "d_sm_median": rng.normal(size=n_bins),
            "d_sm_q25": rng.normal(size=n_bins) - 0.5,
            "d_sm_q75": rng.normal(size=n_bins) + 0.5,
            "d_sm_sig": [True, False, True, False, True][:n_bins],
            "d_vpd_median": rng.normal(size=n_bins),
            "d_vpd_q25": rng.normal(size=n_bins) - 0.5,
            "d_vpd_q75": rng.normal(size=n_bins) + 0.5,
            "d_vpd_sig": [False, True, False, True, False][:n_bins],
        }
    )


def test_heatmap_writes_png(tmp_path):
    rng = np.random.default_rng(1)
    m = rng.normal(size=(5, 5))
    sig = rng.random((5, 5)) > 0.5
    out = tmp_path / "heat.png"
    plot_sensitivity_heatmap(m, sig, "Sensitivity of E to SWC", out)
    assert out.exists() and out.stat().st_size > 0


def test_heatmap_handles_all_nan(tmp_path):
    out = tmp_path / "heat_nan.png"
    plot_sensitivity_heatmap(np.full((5, 5), np.nan), np.zeros((5, 5), bool), "empty", out)
    assert out.exists() and out.stat().st_size > 0


def test_dual_legs_writes_png(tmp_path):
    out = tmp_path / "dual.png"
    plot_dual_legs(_leg_table(axis_name="sm_bin"), _leg_table(axis_name="vpd_bin"), "E", out)
    assert out.exists() and out.stat().st_size > 0


def test_pft_panels_writes_png(tmp_path):
    out = tmp_path / "pft.png"
    per_pft = {"ENF": _leg_table(), "DBF": _leg_table(), "EBF": _leg_table()}
    plot_pft_panels(per_pft, "E", out)
    assert out.exists() and out.stat().st_size > 0

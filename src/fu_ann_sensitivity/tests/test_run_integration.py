"""End-to-end integration test for the Fu ANN sensitivity CLI (Task 6).

Writes tiny synthetic per-site daily CSVs (sibling-loader input schema), runs the
analysis for one response x one SM variant across BOTH bin counts, and checks the
full output set (per-site CSV, cross-site maps tagged _nbins5/_nbins10, dual-leg
tables, performance table, and at least one figure per tag).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.fu_ann_sensitivity.run_fu_ann_sensitivity import run_analysis


def _write_site_csv(path, site, seed):
    rng = np.random.default_rng(seed)
    n = 130
    swvl1 = rng.uniform(0.10, 0.40, n)
    vpd = rng.uniform(0.6, 2.5, n)
    sap = 1.0 + 3.0 * swvl1 + 0.8 * vpd + 0.15 * rng.normal(size=n)  # rises with SM & VPD
    df = pd.DataFrame(
        {
            "TIMESTAMP": pd.date_range("2015-05-01", periods=n, freq="D"),
            "site_name": site,
            "sap_velocity": sap,
            "ta": rng.uniform(16, 30, n),
            "vpd": vpd,
            "ppfd_in": rng.uniform(600, 1100, n),
            "volumetric_soil_water_layer_1": swvl1,
            "volumetric_soil_water_layer_2": swvl1 + rng.uniform(0, 0.05, n),
            "volumetric_soil_water_layer_3": swvl1 + rng.uniform(0, 0.08, n),
            "volumetric_soil_water_layer_4": swvl1 + rng.uniform(0, 0.10, n),
            "elevation": 150.0,
            "pft": "ENF",
        }
    )
    df.to_csv(path, index=False)


def test_run_analysis_end_to_end(tmp_path):
    data_dir = tmp_path / "daily"
    data_dir.mkdir()
    for i, site in enumerate(["s1", "s2", "s3"]):
        _write_site_csv(data_dir / f"{site}.csv", site, seed=i)

    out_dir = tmp_path / "out"
    run_analysis(
        data_dir=str(data_dir),
        out_dir=str(out_dir),
        sm_variants=["swvl1"],
        responses=["E"],
        n_bins_list=[5, 10],
        n_repeats=2,
        min_valid_days=20,
        make_figures=True,
    )

    assert (out_dir / "per_site_E_swvl1.csv").exists()
    assert (out_dir / "performance.csv").exists()
    for n in (5, 10):
        assert (out_dir / f"maps_E_swvl1_nbins{n}.csv").exists()
        assert (out_dir / f"by_sm_bin_E_swvl1_nbins{n}.csv").exists()
        assert (out_dir / f"by_vpd_bin_E_swvl1_nbins{n}.csv").exists()
        assert list(out_dir.glob(f"*_nbins{n}.png"))  # >=1 figure per bin count

    # performance table records the fitted sites.
    perf = pd.read_csv(out_dir / "performance.csv")
    assert perf.loc[0, "n_sites"] >= 1

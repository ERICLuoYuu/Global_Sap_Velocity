# src/sm_vpd_decoupling/tests/test_run_integration.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.sm_vpd_decoupling.run_sm_vpd_decoupling import run_analysis


def _write_site_csv(path, site, seed):
    rng = np.random.default_rng(seed)
    n = 500
    sm = rng.uniform(0.1, 0.35, n)
    vpd = rng.uniform(0.4, 2.5, n)
    df = pd.DataFrame(
        {
            "site_name": [site] * n,
            "TIMESTAMP": pd.date_range("2007-01-01", periods=n, freq="D"),
            "sap_velocity": 6.0 * sm + rng.normal(0, 0.05, n),
            "ta": rng.uniform(12.0, 28.0, n),
            "vpd": vpd,
            "ppfd_in": rng.uniform(400.0, 1300.0, n),
            "sw_in": rng.uniform(100.0, 500.0, n),
            "surface_solar_radiation_downwards_hourly": rng.uniform(100.0, 500.0, n),
            "volumetric_soil_water_layer_1": sm,
            "volumetric_soil_water_layer_2": sm * 0.9,
            "volumetric_soil_water_layer_3": sm * 0.8,
            "volumetric_soil_water_layer_4": sm * 0.7,
            "temperature_2m": rng.uniform(12.0, 28.0, n) + 273.15,
            "dewpoint_2m": rng.uniform(5.0, 15.0, n) + 273.15,
            "elevation": [150.0] * n,
            "pft": ["ENF"] * n,
            "biome": ["temperate"] * n,
            "prcip/PET": [0.7] * n,
            "canopy_height": [20.0] * n,
            "latitude_x": [46.0] * n,
            "longitude_x": [8.0] * n,
        }
    )
    df.to_csv(path, index=False)


def test_run_analysis_end_to_end(tmp_path):
    daily = tmp_path / "daily"
    daily.mkdir()
    for i, s in enumerate(["A", "B", "C"]):
        _write_site_csv(daily / f"{s}_daily.csv", s, seed=i)
    out_dir = tmp_path / "out"
    run_analysis(
        data_dir=str(daily),
        out_dir=str(out_dir),
        climate_source="site",
        tair_min=15.0,
        n_bins_list=[5],
        min_valid_days_list=[120],
        make_figures=False,
    )
    # Depth-profile dissociation table exists and covers both responses + 5 SM variants.
    depth = pd.read_csv(out_dir / "depth_profile.csv")
    assert set(depth["response"].unique()) == {"E_norm", "Gc_norm"}
    assert set(depth["sm_variant"].unique()) == {"swvl1", "swvl2", "swvl3", "swvl4", "root_zone_sm"}
    # Per-site CSV written for at least one combination.
    assert (out_dir / "per_site_E_norm_swvl1_nbins5_mvd120.csv").exists()
    # Attrition table exists AND decomposes the Liu day filter into its funnel
    # (input -> Tair -> +VPD -> +PPFD) plus the min_valid_days threshold.
    attrition = pd.read_csv(out_dir / "attrition.csv")
    stages = list(attrition["stage"])
    assert any("input" in s for s in stages)
    assert any(s.startswith("Tair>") for s in stages)
    assert any("VPD" in s for s in stages)
    assert any("PPFD" in s for s in stages)
    assert any("min_valid_days>=120" in s for s in stages)
    # Funnel is monotonically non-increasing in surviving rows.
    funnel = attrition[~attrition["stage"].str.startswith("min_valid_days")]
    assert list(funnel["n_rows"]) == sorted(funnel["n_rows"], reverse=True)

# src/sm_vpd_decoupling/tests/test_loader_load.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.sm_vpd_decoupling.loader import load_table, resolve_daily_dir


def _write_site_csv(path, site, n=400, seed=0):
    rng = np.random.default_rng(seed)
    sm = rng.uniform(0.1, 0.35, n)
    vpd = rng.uniform(0.3, 2.5, n)
    df = pd.DataFrame(
        {
            "site_name": [site] * n,
            "TIMESTAMP": pd.date_range("2008-01-01", periods=n, freq="D"),
            "sap_velocity": 5.0 * sm + rng.normal(0, 0.05, n),
            "ta": rng.uniform(12.0, 28.0, n),
            "vpd": vpd,
            "ppfd_in": rng.uniform(400.0, 1200.0, n),
            "sw_in": rng.uniform(100.0, 500.0, n),
            "surface_solar_radiation_downwards_hourly": rng.uniform(100.0, 500.0, n),
            "volumetric_soil_water_layer_1": sm,
            "volumetric_soil_water_layer_2": sm,
            "volumetric_soil_water_layer_3": sm,
            "volumetric_soil_water_layer_4": sm,
            "temperature_2m": rng.uniform(12.0, 28.0, n) + 273.15,
            "dewpoint_2m": rng.uniform(5.0, 15.0, n) + 273.15,
            "elevation": [200.0] * n,
            "pft": ["ENF"] * n,
            "biome": ["temperate"] * n,
            "prcip/PET": [0.7] * n,
            "canopy_height": [18.0] * n,
            "latitude_x": [46.0] * n,
            "longitude_x": [8.0] * n,
        }
    )
    df.to_csv(path, index=False)


def test_resolve_daily_dir_explicit(tmp_path):
    d = tmp_path / "daily"
    d.mkdir()
    _write_site_csv(d / "S1_daily.csv", "S1")
    assert resolve_daily_dir(str(d)) == d


def test_load_table_builds_normalized_responses(tmp_path):
    d = tmp_path / "daily"
    d.mkdir()
    _write_site_csv(d / "S1_daily.csv", "S1", seed=1)
    _write_site_csv(d / "S2_daily.csv", "S2", seed=2)
    table = load_table(str(d), climate_source="site", tair_min=15.0)
    assert {"E_norm", "Gc_norm", "swvl1", "root_zone_sm"}.issubset(table.columns)
    assert set(table["site_name"].unique()) == {"S1", "S2"}
    # Normalized responses are positive and finite where E is present.
    assert table["E_norm"].dropna().gt(0).all()


def test_load_table_raises_on_empty_dir(tmp_path):
    d = tmp_path / "daily"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        load_table(str(d), climate_source="site", tair_min=15.0)

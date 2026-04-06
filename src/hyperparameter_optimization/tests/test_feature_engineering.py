"""Tests for feature_engineering extracted module."""

import numpy as np
import pandas as pd
import pytest

from src.hyperparameter_optimization.feature_engineering import (
    add_sap_flow_features,
    apply_feature_engineering,
    calculate_soil_hydraulics_sr2006,
)


class TestAddSapFlowFeatures:
    """Test add_sap_flow_features column creation."""

    @pytest.fixture
    def base_df(self):
        rng = np.random.RandomState(42)
        n = 100
        return pd.DataFrame(
            {
                "vpd": rng.uniform(0.1, 3.0, n),
                "sw_in": rng.uniform(0, 800, n),
                "ta": rng.uniform(5, 35, n),
                "precip": rng.uniform(0, 20, n),
                "ws": rng.uniform(0, 10, n),
                "LAI": rng.uniform(0.5, 6.0, n),
                "volumetric_soil_water_layer_1": rng.uniform(0.1, 0.5, n),
                "canopy_height": rng.uniform(5, 30, n),
                "soil_temperature_level_1": rng.uniform(273, 300, n),
                "ppfd_in": rng.uniform(0, 2000, n),
            }
        )

    def test_returns_dataframe(self, base_df):
        result = add_sap_flow_features(base_df)
        assert isinstance(result, pd.DataFrame)

    def test_adds_interaction_features(self, base_df):
        result = add_sap_flow_features(base_df)
        assert "vpd_x_sw_in" in result.columns

    def test_does_not_mutate_input(self, base_df):
        original_cols = set(base_df.columns)
        add_sap_flow_features(base_df)
        assert set(base_df.columns) == original_cols

    def test_handles_missing_columns(self):
        df = pd.DataFrame({"ta": [1.0, 2.0, 3.0]})
        result = add_sap_flow_features(df)
        assert isinstance(result, pd.DataFrame)


class TestApplyFeatureEngineering:
    """Test apply_feature_engineering with groups."""

    @pytest.fixture
    def grouped_df(self):
        rng = np.random.RandomState(42)
        n = 60
        df = pd.DataFrame(
            {
                "vpd": rng.uniform(0.1, 3.0, n),
                "sw_in": rng.uniform(0, 800, n),
                "ta": rng.uniform(5, 35, n),
                "precip": rng.uniform(0, 20, n),
                "ws": rng.uniform(0, 10, n),
                "LAI": rng.uniform(0.5, 6.0, n),
                "volumetric_soil_water_layer_1": rng.uniform(0.1, 0.5, n),
                "canopy_height": rng.uniform(5, 30, n),
                "soil_temperature_level_1": rng.uniform(273, 300, n),
                "ppfd_in": rng.uniform(0, 2000, n),
                "sap_velocity": rng.uniform(0, 50, n),
            }
        )
        groups = np.array([0] * 20 + [1] * 20 + [2] * 20)
        return df, groups

    def test_returns_tuple(self, grouped_df):
        df, groups = grouped_df
        result_df, new_features = apply_feature_engineering(df, groups)
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(new_features, list)

    def test_nan_handling(self, grouped_df):
        df, groups = grouped_df
        result_df, _ = apply_feature_engineering(df, groups)
        # Rolling features will have NaN at edges — that's expected
        # but no all-NaN columns should be created
        all_nan_cols = [c for c in result_df.columns if result_df[c].isna().all()]
        assert len(all_nan_cols) == 0


class TestCalculateSoilHydraulicsSR2006:
    """Test Saxton & Rawls (2006) pedotransfer functions."""

    def test_sandy_soil_returns_tuple(self):
        wp, fc, sat = calculate_soil_hydraulics_sr2006(0.9, 0.05, 1.0, 0)
        assert isinstance(wp, float)
        assert isinstance(fc, float)
        assert isinstance(sat, float)

    def test_ordering_wp_lt_fc_lt_sat(self):
        """Wilting point < field capacity < saturation for any valid soil."""
        wp, fc, sat = calculate_soil_hydraulics_sr2006(0.4, 0.3, 2.0, 0)
        assert wp < fc < sat

    def test_clay_soil_higher_wp_than_sand(self):
        wp_clay, _, _ = calculate_soil_hydraulics_sr2006(0.1, 0.6, 1.0, 0)
        wp_sand, _, _ = calculate_soil_hydraulics_sr2006(0.9, 0.05, 1.0, 0)
        assert wp_clay > wp_sand

    def test_coarse_fragments_reduce_all_values(self):
        wp0, fc0, sat0 = calculate_soil_hydraulics_sr2006(0.4, 0.3, 2.0, 0)
        wp50, fc50, sat50 = calculate_soil_hydraulics_sr2006(0.4, 0.3, 2.0, 50)
        assert wp50 == pytest.approx(wp0 * 0.5)
        assert fc50 == pytest.approx(fc0 * 0.5)
        assert sat50 == pytest.approx(sat0 * 0.5)

    def test_zero_coarse_fragments(self):
        wp, fc, sat = calculate_soil_hydraulics_sr2006(0.4, 0.3, 2.0, 0)
        assert wp > 0
        assert fc > 0
        assert sat > 0

    def test_known_loam_values(self):
        """Loam soil (40% sand, 20% clay, 2% OM) should give realistic values."""
        wp, fc, sat = calculate_soil_hydraulics_sr2006(0.4, 0.2, 2.0, 0)
        # Typical loam: WP ~0.08-0.15, FC ~0.20-0.35, Sat ~0.40-0.55
        assert 0.05 < wp < 0.20
        assert 0.15 < fc < 0.40
        assert 0.35 < sat < 0.60


class TestRootZoneSWC:
    """Test root_zone_swc feature group."""

    @pytest.fixture
    def soil_df(self):
        rng = np.random.RandomState(42)
        n = 50
        return pd.DataFrame(
            {
                "volumetric_soil_water_layer_2": rng.uniform(0.1, 0.4, n),
                "volumetric_soil_water_layer_3": rng.uniform(0.1, 0.4, n),
                "volumetric_soil_water_layer_4": rng.uniform(0.1, 0.4, n),
                "soil_temperature_level_2": rng.uniform(275, 295, n),
                "soil_temperature_level_3": rng.uniform(275, 295, n),
                "soil_temperature_level_4": rng.uniform(275, 295, n),
                "sap_velocity": rng.uniform(0, 50, n),
            }
        )

    def test_adds_normalized_swc(self, soil_df):
        result, feats = apply_feature_engineering(soil_df, ["root_zone_swc"])
        for layer in [2, 3, 4]:
            assert f"swc_layer{layer}_norm" in feats

    def test_includes_soil_temp(self, soil_df):
        result, feats = apply_feature_engineering(soil_df, ["root_zone_swc"])
        for layer in [2, 3, 4]:
            assert f"soil_temperature_level_{layer}" in feats

    def test_no_all_nan_columns(self, soil_df):
        result, feats = apply_feature_engineering(soil_df, ["root_zone_swc"])
        for f in feats:
            if f in result.columns:
                assert not result[f].isna().all()


class TestREW:
    """Test rew (Relative Extractable Water) feature group."""

    @pytest.fixture
    def rew_df(self):
        rng = np.random.RandomState(42)
        n = 50
        return pd.DataFrame(
            {
                "soil_sand": [40.0] * n,  # 40% sand
                "soil_clay": [20.0] * n,  # 20% clay
                "soil_soc": [15.0] * n,  # g/kg SOC
                "soil_cfvo": [5.0] * n,  # 5% coarse fragments
                "volumetric_soil_water_layer_2": rng.uniform(0.1, 0.4, n),
                "sap_velocity": rng.uniform(0, 50, n),
            }
        )

    def test_creates_rew_column(self, rew_df):
        result, feats = apply_feature_engineering(rew_df, ["rew"])
        assert "rew" in feats
        assert "rew" in result.columns

    def test_rew_bounded(self, rew_df):
        result, _ = apply_feature_engineering(rew_df, ["rew"])
        assert result["rew"].max() <= 1.5
        assert result["rew"].min() >= 0.0

    def test_rew_skipped_without_soil_data(self):
        df = pd.DataFrame(
            {
                "volumetric_soil_water_layer_2": [0.2, 0.3, 0.25],
                "sap_velocity": [10, 20, 15],
            }
        )
        result, feats = apply_feature_engineering(df, ["rew"])
        assert "rew" not in feats


class TestET0:
    """Test et0 (FAO-56 Penman-Monteith) feature group."""

    @pytest.fixture
    def et0_df(self):
        n = 30
        return pd.DataFrame(
            {
                "ta": np.full(n, 20.0),
                "ta_max": np.full(n, 25.0),
                "ta_min": np.full(n, 15.0),
                "rh": np.full(n, 60.0),
                "ws": np.full(n, 2.0),
                "sw_in": np.full(n, 200.0),
                "ext_rad": np.full(n, 350.0),
                "surface_pressure": np.full(n, 101300.0),
                "elevation": np.full(n, 100.0),
                "sap_velocity": np.full(n, 15.0),
            }
        )

    def test_creates_et0_column(self, et0_df):
        result, feats = apply_feature_engineering(et0_df, ["et0"])
        assert "et0" in feats
        assert "et0" in result.columns

    def test_et0_non_negative(self, et0_df):
        result, _ = apply_feature_engineering(et0_df, ["et0"])
        assert (result["et0"] >= 0).all()

    def test_et0_realistic_range(self, et0_df):
        """Typical daily ET0 is 0-15 mm/day."""
        result, _ = apply_feature_engineering(et0_df, ["et0"])
        assert result["et0"].mean() < 20

    def test_et0_skipped_without_required_cols(self):
        df = pd.DataFrame(
            {
                "ta": [20.0, 22.0],
                "ws": [2.0, 3.0],
                "sap_velocity": [10, 20],
            }
        )
        result, feats = apply_feature_engineering(df, ["et0"])
        assert "et0" not in feats


class TestPsiSoil:
    """Test psi_soil (Campbell 1974 + Cosby 1984) feature group."""

    @pytest.fixture
    def psi_df(self):
        n = 30
        return pd.DataFrame(
            {
                "soil_sand": [40.0] * n,
                "soil_clay": [20.0] * n,
                "volumetric_soil_water_layer_2": np.linspace(0.1, 0.4, n),
                "sap_velocity": np.full(n, 15.0),
            }
        )

    def test_creates_psi_soil_column(self, psi_df):
        result, feats = apply_feature_engineering(psi_df, ["psi_soil"])
        assert "psi_soil" in feats
        assert "psi_soil" in result.columns

    def test_psi_soil_bounded(self, psi_df):
        result, _ = apply_feature_engineering(psi_df, ["psi_soil"])
        assert result["psi_soil"].max() <= 0
        assert result["psi_soil"].min() >= -10

    def test_wetter_soil_less_negative_psi(self, psi_df):
        result, _ = apply_feature_engineering(psi_df, ["psi_soil"])
        # Last rows have higher SWC → less negative psi
        assert result["psi_soil"].iloc[-1] > result["psi_soil"].iloc[0]

    def test_psi_skipped_without_soil_texture(self):
        df = pd.DataFrame(
            {
                "volumetric_soil_water_layer_2": [0.2, 0.3],
                "sap_velocity": [10, 20],
            }
        )
        result, feats = apply_feature_engineering(df, ["psi_soil"])
        assert "psi_soil" not in feats


class TestCWD:
    """Test cwd (Cumulative Water Deficit) feature group."""

    @pytest.fixture
    def cwd_df(self):
        n = 30
        return pd.DataFrame(
            {
                # PET in m (negative by ERA5 convention), precip in m
                "potential_evaporation_hourly_sum": np.full(n, -0.001),  # -1mm
                "total_precipitation_hourly_sum": np.full(n, 0.0005),  # 0.5mm
                "sap_velocity": np.full(n, 15.0),
            }
        )

    def test_creates_cwd_column(self, cwd_df):
        result, feats = apply_feature_engineering(cwd_df, ["cwd"])
        assert "cwd" in feats
        assert "cwd" in result.columns

    def test_cwd_non_negative(self, cwd_df):
        result, _ = apply_feature_engineering(cwd_df, ["cwd"])
        assert (result["cwd"] >= 0).all()

    def test_cwd_accumulates(self, cwd_df):
        """With constant deficit, CWD should increase monotonically."""
        result, _ = apply_feature_engineering(cwd_df, ["cwd"])
        cwd_vals = result["cwd"].values
        assert all(cwd_vals[i] <= cwd_vals[i + 1] for i in range(len(cwd_vals) - 1))

    def test_cwd_skipped_without_required_cols(self):
        df = pd.DataFrame(
            {
                "precip": [1.0, 2.0],
                "sap_velocity": [10, 20],
            }
        )
        result, feats = apply_feature_engineering(df, ["cwd"])
        assert "cwd" not in feats

    def test_cwd_resets_with_surplus(self):
        """CWD should not accumulate when precip > PET."""
        n = 10
        df = pd.DataFrame(
            {
                "potential_evaporation_hourly_sum": np.full(n, -0.0005),  # 0.5mm PET
                "total_precipitation_hourly_sum": np.full(n, 0.001),  # 1mm precip (surplus)
                "sap_velocity": np.full(n, 15.0),
            }
        )
        result, _ = apply_feature_engineering(df, ["cwd"])
        # With surplus (precip > PET), deficit is negative → max(0, running + neg) → stays 0
        assert (result["cwd"] == 0).all()

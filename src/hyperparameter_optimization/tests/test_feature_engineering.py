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
    def base_df_with_sap(self):
        rng = np.random.RandomState(42)
        n = 60
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
                "sap_velocity": rng.uniform(0, 50, n),
            }
        )

    def test_returns_tuple(self, base_df_with_sap):
        result_df, new_features = apply_feature_engineering(base_df_with_sap, ["rolling_3d"])
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(new_features, list)
        assert len(new_features) > 0

    def test_rolling_creates_features(self, base_df_with_sap):
        result_df, new_features = apply_feature_engineering(base_df_with_sap, ["rolling_3d"])
        assert "ta_roll3d_mean" in new_features
        assert "vpd_roll3d_std" in new_features

    def test_no_internal_dropna(self, base_df_with_sap):
        """FE must NOT drop rows internally — callers handle dropna."""
        n_before = len(base_df_with_sap)
        result_df, _ = apply_feature_engineering(base_df_with_sap, ["rolling_3d", "lags_1d"])
        assert len(result_df) == n_before

    def test_no_internal_dropna_with_precip_memory(self, base_df_with_sap):
        n_before = len(base_df_with_sap)
        result_df, _ = apply_feature_engineering(base_df_with_sap, ["precip_memory"])
        assert len(result_df) == n_before

    def test_nan_at_edges_expected(self, base_df_with_sap):
        """Rolling/lag features have NaN at edges — that's expected, not dropped."""
        result_df, new_features = apply_feature_engineering(base_df_with_sap, ["rolling_3d", "lags_1d"])
        # First row should have NaN in lag features
        lag_feats = [f for f in new_features if "lag" in f]
        if lag_feats:
            assert result_df[lag_feats[0]].isna().any()


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


class TestTreeMetadata:
    """Test tree_metadata feature group."""

    @pytest.fixture
    def tree_df(self):
        n = 20
        return pd.DataFrame(
            {
                "pl_dbh": [15.0, 20.0, np.nan, 25.0] * 5,
                "pl_sens_meth": ["HD", "CHP", "HR", "HD"] * 5,
                "pl_species": [
                    "Pinus sylvestris",
                    "Fagus sylvatica",
                    "Quercus robur",
                    "Unknown",
                ]
                * 5,
                "sap_velocity": np.full(n, 15.0),
            }
        )

    def test_creates_dbh_feature(self, tree_df):
        result, feats = apply_feature_engineering(tree_df, ["tree_metadata"])
        assert "pl_dbh" in feats

    def test_creates_method_dummies(self, tree_df):
        result, feats = apply_feature_engineering(tree_df, ["tree_metadata"])
        assert "meth_HD" in feats
        assert "meth_CHP" in feats
        assert result["meth_HD"].dtype == np.float32

    def test_creates_genus_dummies(self, tree_df):
        result, feats = apply_feature_engineering(tree_df, ["tree_metadata"])
        assert "genus_Pinus" in feats
        assert "genus_Fagus" in feats
        assert "genus_Quercus" in feats
        assert "genus_Other" in feats

    def test_genus_other_for_unknown(self, tree_df):
        result, _ = apply_feature_engineering(tree_df, ["tree_metadata"])
        # "Unknown" species should map to genus_Other
        assert result["genus_Other"].sum() > 0

    def test_skipped_without_columns(self):
        df = pd.DataFrame({"sap_velocity": [10, 20, 30]})
        result, feats = apply_feature_engineering(df, ["tree_metadata"])
        assert "pl_dbh" not in feats
        assert "meth_HD" not in feats


# =====================================================================
# Expanded tests — edge cases and untested groups
# =====================================================================


class TestInteractionsGroup:
    """Test interactions feature group."""

    @pytest.fixture
    def interaction_df(self):
        n = 20
        return pd.DataFrame(
            {
                "vpd": np.full(n, 2.0),
                "sw_in": np.full(n, 400.0),
                "ta": np.full(n, 20.0),
                "canopy_height": np.full(n, 15.0),
                "ws": np.full(n, 3.0),
                "prcip/PET": np.full(n, 0.5),
            }
        )

    def test_creates_all_interactions(self, interaction_df):
        result, feats = apply_feature_engineering(interaction_df, ["interactions"])
        expected = ["vpd_x_sw_in", "vpd_squared", "ta_x_vpd", "height_x_vpd", "wind_x_vpd", "demand_x_supply"]
        for f in expected:
            assert f in feats, f"Missing interaction feature: {f}"

    def test_vpd_squared_correct(self, interaction_df):
        result, _ = apply_feature_engineering(interaction_df, ["interactions"])
        assert result["vpd_squared"].iloc[0] == pytest.approx(4.0)

    def test_interactions_with_missing_cols(self):
        df = pd.DataFrame({"vpd": [1.0, 2.0], "ta": [20.0, 25.0]})
        result, feats = apply_feature_engineering(df, ["interactions"])
        assert "vpd_squared" in feats
        assert "ta_x_vpd" in feats
        # sw_in missing → vpd_x_sw_in not created
        assert "vpd_x_sw_in" not in feats


class TestLags1dGroup:
    """Test lags_1d feature group."""

    def test_creates_lag_features(self):
        n = 10
        df = pd.DataFrame(
            {
                "ta": np.arange(n, dtype=float),
                "vpd": np.arange(n, dtype=float),
                "sw_in": np.arange(n, dtype=float),
                "precip": np.arange(n, dtype=float),
                "rh": np.arange(n, dtype=float),
            }
        )
        result, feats = apply_feature_engineering(df, ["lags_1d"])
        for col in ["ta", "vpd", "sw_in", "precip", "rh"]:
            assert f"{col}_lag1d" in feats

    def test_lag_values_correct(self):
        df = pd.DataFrame({"ta": [10.0, 20.0, 30.0]})
        result, _ = apply_feature_engineering(df, ["lags_1d"])
        assert np.isnan(result["ta_lag1d"].iloc[0])
        assert result["ta_lag1d"].iloc[1] == pytest.approx(10.0)
        assert result["ta_lag1d"].iloc[2] == pytest.approx(20.0)

    def test_no_rows_dropped(self):
        df = pd.DataFrame({"ta": np.arange(5, dtype=float)})
        result, _ = apply_feature_engineering(df, ["lags_1d"])
        assert len(result) == 5


class TestPhysicsGroup:
    """Test physics feature group."""

    def test_clear_sky_index_bounded(self):
        df = pd.DataFrame(
            {
                "sw_in": [0.0, 200.0, 500.0, 900.0],
                "ext_rad": [300.0, 300.0, 300.0, 300.0],
                "ta": [10.0, 20.0, 30.0, 40.0],
                "LAI": [2.0, 3.0, 4.0, 5.0],
            }
        )
        result, feats = apply_feature_engineering(df, ["physics"])
        assert "clear_sky_index" in feats
        assert result["clear_sky_index"].max() <= 1.0
        assert result["clear_sky_index"].min() >= 0.0

    def test_gdd_non_negative(self):
        df = pd.DataFrame({"ta": [-5.0, 0.0, 5.0, 10.0, 20.0]})
        result, feats = apply_feature_engineering(df, ["physics"])
        assert "gdd" in feats
        assert (result["gdd"] >= 0).all()
        assert result["gdd"].iloc[0] == 0.0  # -5 < 5 → clipped to 0
        assert result["gdd"].iloc[4] == pytest.approx(15.0)  # 20-5=15

    def test_absorbed_radiation(self):
        df = pd.DataFrame({"sw_in": [300.0], "ta": [20.0], "LAI": [3.0]})
        result, feats = apply_feature_engineering(df, ["physics"])
        assert "absorbed_radiation" in feats
        assert result["absorbed_radiation"].iloc[0] > 0


class TestPrecipMemoryGroup:
    """Test precip_memory feature group."""

    def test_creates_precip_features(self):
        df = pd.DataFrame(
            {
                "precip": np.concatenate([np.full(10, 5.0), np.zeros(10), np.full(10, 2.0)]),
            }
        )
        result, feats = apply_feature_engineering(df, ["precip_memory"])
        assert "precip_sum_3d" in feats
        assert "precip_sum_7d" in feats
        assert "days_since_rain" in feats

    def test_days_since_rain_resets(self):
        # Rain, no rain, no rain, rain, no rain
        df = pd.DataFrame({"precip": [5.0, 0.0, 0.0, 3.0, 0.0]})
        result, _ = apply_feature_engineering(df, ["precip_memory"])
        dsr = result["days_since_rain"].values
        assert dsr[0] == 0  # rain day
        assert dsr[1] == 1  # 1 day since
        assert dsr[2] == 2  # 2 days since
        assert dsr[3] == 0  # rain again


class TestIndicatorsGroup:
    """Test indicators feature group."""

    def test_vpd_high_threshold(self):
        df = pd.DataFrame({"vpd": [1.0, 2.0, 2.5, 3.0, 4.0]})
        result, feats = apply_feature_engineering(df, ["indicators"])
        assert "vpd_high" in feats
        assert result["vpd_high"].iloc[0] == 0.0  # 1.0 < 2.5
        assert result["vpd_high"].iloc[3] == 1.0  # 3.0 > 2.5

    def test_soil_moisture_rel_with_swc(self):
        n = 20
        df = pd.DataFrame(
            {
                "vpd": np.full(n, 2.0),
                "volumetric_soil_water_layer_1": np.linspace(0.05, 0.45, n),
            }
        )
        result, feats = apply_feature_engineering(df, ["indicators"])
        assert "soil_moisture_rel" in feats
        assert "soil_dry" in feats
        assert result["soil_moisture_rel"].min() >= 0.0
        assert result["soil_moisture_rel"].max() <= 1.0


class TestStaticEnrichGroup:
    """Test static_enrich feature group."""

    def test_includes_existing_static_cols(self):
        n = 5
        df = pd.DataFrame(
            {
                "stand_age": np.full(n, 50.0),
                "slope": np.full(n, 10.0),
                "mean_annual_temp": np.full(n, 12.0),
            }
        )
        result, feats = apply_feature_engineering(df, ["static_enrich"])
        assert "stand_age" in feats
        assert "slope" in feats
        assert "mean_annual_temp" in feats

    def test_skips_missing_static_cols(self):
        df = pd.DataFrame({"ta": [20.0, 25.0]})
        result, feats = apply_feature_engineering(df, ["static_enrich"])
        assert len(feats) == 0


class TestRootZoneSWCEdgeCases:
    """Edge cases for root_zone_swc normalization."""

    def test_constant_swc_produces_zero(self):
        """Constant SWC → std=0 → guard returns 0.0."""
        n = 20
        df = pd.DataFrame(
            {
                "volumetric_soil_water_layer_2": np.full(n, 0.25),
                "sap_velocity": np.full(n, 15.0),
            }
        )
        result, feats = apply_feature_engineering(df, ["root_zone_swc"])
        assert "swc_layer2_norm" in feats
        # Constant input → std ≈ 0 → guard sets 0.0
        assert (result["swc_layer2_norm"] == 0.0).all()

    def test_zscore_mean_near_zero(self):
        """Z-score normalized values should have mean ≈ 0."""
        rng = np.random.RandomState(42)
        n = 100
        df = pd.DataFrame(
            {
                "volumetric_soil_water_layer_2": rng.uniform(0.1, 0.4, n),
            }
        )
        result, _ = apply_feature_engineering(df, ["root_zone_swc"])
        assert abs(result["swc_layer2_norm"].mean()) < 0.1

    def test_zscore_std_near_one(self):
        """Z-score normalized values should have std ≈ 1."""
        rng = np.random.RandomState(42)
        n = 200
        df = pd.DataFrame(
            {
                "volumetric_soil_water_layer_3": rng.uniform(0.1, 0.4, n),
            }
        )
        result, _ = apply_feature_engineering(df, ["root_zone_swc"])
        assert abs(result["swc_layer3_norm"].std() - 1.0) < 0.1


class TestET0EdgeCases:
    """Edge cases for ET0 computation."""

    def test_et0_zero_elevation(self):
        """ET0 should work at sea level (elevation=0)."""
        n = 10
        df = pd.DataFrame(
            {
                "ta": np.full(n, 25.0),
                "ta_max": np.full(n, 30.0),
                "ta_min": np.full(n, 20.0),
                "rh": np.full(n, 50.0),
                "ws": np.full(n, 2.0),
                "sw_in": np.full(n, 250.0),
                "ext_rad": np.full(n, 400.0),
                "elevation": np.full(n, 0.0),
            }
        )
        result, feats = apply_feature_engineering(df, ["et0"])
        assert "et0" in feats
        assert (result["et0"] >= 0).all()

    def test_et0_high_elevation(self):
        """ET0 should work at high altitude (4000m) with lower pressure."""
        n = 10
        df = pd.DataFrame(
            {
                "ta": np.full(n, 10.0),
                "ta_max": np.full(n, 15.0),
                "ta_min": np.full(n, 5.0),
                "rh": np.full(n, 70.0),
                "ws": np.full(n, 3.0),
                "sw_in": np.full(n, 300.0),
                "ext_rad": np.full(n, 450.0),
                "elevation": np.full(n, 4000.0),
            }
        )
        result, feats = apply_feature_engineering(df, ["et0"])
        assert "et0" in feats
        assert (result["et0"] >= 0).all()

    def test_et0_without_surface_pressure(self):
        """ET0 should NOT require surface_pressure — derives pressure from elevation."""
        n = 5
        df = pd.DataFrame(
            {
                "ta": np.full(n, 20.0),
                "ta_max": np.full(n, 25.0),
                "ta_min": np.full(n, 15.0),
                "rh": np.full(n, 60.0),
                "ws": np.full(n, 2.0),
                "sw_in": np.full(n, 200.0),
                "ext_rad": np.full(n, 350.0),
                "elevation": np.full(n, 100.0),
                # No surface_pressure column!
            }
        )
        result, feats = apply_feature_engineering(df, ["et0"])
        assert "et0" in feats

    def test_et0_zero_wind_speed(self):
        """ET0 should handle zero wind speed gracefully."""
        n = 5
        df = pd.DataFrame(
            {
                "ta": np.full(n, 20.0),
                "ta_max": np.full(n, 25.0),
                "ta_min": np.full(n, 15.0),
                "rh": np.full(n, 60.0),
                "ws": np.full(n, 0.0),
                "sw_in": np.full(n, 200.0),
                "ext_rad": np.full(n, 350.0),
                "elevation": np.full(n, 100.0),
            }
        )
        result, feats = apply_feature_engineering(df, ["et0"])
        assert "et0" in feats
        assert not result["et0"].isna().any()


class TestCWDEdgeCases:
    """Edge cases for CWD computation."""

    def test_cwd_with_nan_precip(self):
        """NaN in precipitation should produce NaN in CWD (not crash)."""
        df = pd.DataFrame(
            {
                "potential_evaporation_hourly_sum": [-0.001, -0.001, -0.001, -0.001, -0.001],
                "total_precipitation_hourly_sum": [0.0005, np.nan, 0.0005, 0.0005, np.nan],
            }
        )
        result, feats = apply_feature_engineering(df, ["cwd"])
        assert "cwd" in feats
        # NaN rows should propagate NaN
        assert result["cwd"].isna().any()

    def test_cwd_all_zero(self):
        """No PET and no precip → CWD stays 0."""
        n = 10
        df = pd.DataFrame(
            {
                "potential_evaporation_hourly_sum": np.zeros(n),
                "total_precipitation_hourly_sum": np.zeros(n),
            }
        )
        result, _ = apply_feature_engineering(df, ["cwd"])
        assert (result["cwd"] == 0).all()


class TestREWEdgeCases:
    """Edge cases for REW."""

    def test_rew_skipped_when_wp_equals_fc(self):
        """When WP ≈ FC (fc - wp < 0.01), REW should not be created."""
        n = 10
        # Very extreme soil: try to make fc ≈ wp
        # Pure sand: sand=1.0, clay=0.0, om=0
        df = pd.DataFrame(
            {
                "soil_sand": [100.0] * n,  # extreme sand
                "soil_clay": [0.0] * n,
                "soil_soc": [0.0] * n,
                "soil_cfvo": [0.0] * n,
                "volumetric_soil_water_layer_2": np.full(n, 0.2),
            }
        )
        result, feats = apply_feature_engineering(df, ["rew"])
        # Either rew is created (fc-wp > 0.01) or skipped — no crash either way
        # We test it doesn't crash
        assert isinstance(result, pd.DataFrame)

    def test_rew_with_nan_soil_data(self):
        """NaN soil texture → REW skipped."""
        n = 5
        df = pd.DataFrame(
            {
                "soil_sand": [np.nan] * n,
                "soil_clay": [np.nan] * n,
                "volumetric_soil_water_layer_2": np.full(n, 0.2),
            }
        )
        result, feats = apply_feature_engineering(df, ["rew"])
        assert "rew" not in feats


class TestSoilHydraulicsEdgeCases:
    """Edge cases for soil hydraulics."""

    def test_100_percent_coarse_fragments(self):
        """100% coarse fragments → all values zero."""
        wp, fc, sat = calculate_soil_hydraulics_sr2006(0.4, 0.3, 2.0, 100)
        assert wp == pytest.approx(0.0)
        assert fc == pytest.approx(0.0)
        assert sat == pytest.approx(0.0)

    def test_extreme_clay_soil(self):
        """Very high clay content should still produce valid results."""
        wp, fc, sat = calculate_soil_hydraulics_sr2006(0.05, 0.8, 1.0, 0)
        assert isinstance(wp, float)
        assert isinstance(fc, float)
        assert isinstance(sat, float)

    def test_zero_organic_matter(self):
        wp, fc, sat = calculate_soil_hydraulics_sr2006(0.5, 0.2, 0.0, 0)
        assert wp > 0
        assert fc > wp


class TestMultipleGroupsCombined:
    """Test combining multiple feature groups."""

    def test_all_groups_no_crash(self):
        """Applying many groups simultaneously should not crash."""
        rng = np.random.RandomState(42)
        n = 60
        df = pd.DataFrame(
            {
                "vpd": rng.uniform(0.5, 3.0, n),
                "sw_in": rng.uniform(50, 600, n),
                "ta": rng.uniform(5, 35, n),
                "ta_max": rng.uniform(25, 40, n),
                "ta_min": rng.uniform(0, 15, n),
                "precip": rng.uniform(0, 10, n),
                "ws": rng.uniform(0.5, 8, n),
                "rh": rng.uniform(30, 90, n),
                "LAI": rng.uniform(1, 5, n),
                "ext_rad": rng.uniform(200, 450, n),
                "canopy_height": rng.uniform(5, 30, n),
                "elevation": np.full(n, 200.0),
                "prcip/PET": rng.uniform(0.2, 2.0, n),
                "soil_sand": np.full(n, 40.0),
                "soil_clay": np.full(n, 20.0),
                "soil_soc": np.full(n, 15.0),
                "soil_cfvo": np.full(n, 5.0),
                "volumetric_soil_water_layer_1": rng.uniform(0.1, 0.4, n),
                "volumetric_soil_water_layer_2": rng.uniform(0.1, 0.4, n),
                "volumetric_soil_water_layer_3": rng.uniform(0.1, 0.4, n),
                "soil_temperature_level_2": rng.uniform(275, 295, n),
                "soil_temperature_level_3": rng.uniform(275, 295, n),
                "potential_evaporation_hourly_sum": np.full(n, -0.001),
                "total_precipitation_hourly_sum": rng.uniform(0, 0.002, n),
                "pl_dbh": rng.uniform(10, 40, n),
                "pl_sens_meth": ["HD"] * n,
                "pl_species": ["Pinus sylvestris"] * n,
                "stand_age": np.full(n, 80.0),
                "sap_velocity": rng.uniform(0, 50, n),
            }
        )
        all_groups = [
            "interactions",
            "lags_1d",
            "rolling_3d",
            "rolling_7d",
            "physics",
            "precip_memory",
            "indicators",
            "static_enrich",
            "root_zone_swc",
            "rew",
            "et0",
            "psi_soil",
            "cwd",
            "tree_metadata",
        ]
        result, feats = apply_feature_engineering(df, all_groups)
        assert len(feats) > 30  # Should create many features
        assert len(result) == n  # No rows dropped

    def test_no_mutation_of_input(self):
        """apply_feature_engineering must not mutate the input DataFrame."""
        rng = np.random.RandomState(42)
        n = 20
        df = pd.DataFrame(
            {
                "vpd": rng.uniform(0.5, 3.0, n),
                "sw_in": rng.uniform(50, 600, n),
                "ta": rng.uniform(5, 35, n),
            }
        )
        original_cols = set(df.columns)
        original_len = len(df)
        apply_feature_engineering(df, ["interactions", "lags_1d", "rolling_3d"])
        assert set(df.columns) == original_cols
        assert len(df) == original_len

    def test_unknown_group_ignored(self):
        """Unknown group names should be silently ignored."""
        df = pd.DataFrame({"ta": [20.0, 25.0, 30.0]})
        result, feats = apply_feature_engineering(df, ["nonexistent_group"])
        assert len(feats) == 0
        assert len(result) == 3


class TestTreeMetadataEdgeCases:
    """Edge cases for tree_metadata group."""

    def test_all_nan_species(self):
        """All-NaN species should create genus_Other = 1 everywhere."""
        n = 5
        df = pd.DataFrame(
            {
                "pl_species": [np.nan] * n,
                "sap_velocity": np.full(n, 15.0),
            }
        )
        result, feats = apply_feature_engineering(df, ["tree_metadata"])
        assert "genus_Other" in feats
        # NaN → str "nan" → not in genera list → genus_Other
        assert (result["genus_Other"] == 1.0).all()

    def test_all_nan_dbh_skipped(self):
        """All-NaN pl_dbh should not be included in features."""
        n = 5
        df = pd.DataFrame(
            {
                "pl_dbh": [np.nan] * n,
                "sap_velocity": np.full(n, 15.0),
            }
        )
        result, feats = apply_feature_engineering(df, ["tree_metadata"])
        assert "pl_dbh" not in feats

    def test_single_word_species(self):
        """Species with single word (no space) should extract as genus."""
        n = 3
        df = pd.DataFrame(
            {
                "pl_species": ["Pinus", "Fagus", "Quercus"],
                "sap_velocity": np.full(n, 15.0),
            }
        )
        result, feats = apply_feature_engineering(df, ["tree_metadata"])
        assert "genus_Pinus" in feats

"""Tests for feature_engineering extracted module."""

import numpy as np
import pandas as pd
import pytest

from src.hyperparameter_optimization.feature_engineering import (
    apply_all_feature_engineering,
    apply_feature_engineering,
    calculate_soil_hydraulics_sr2006,
)


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
        # Pre-computed Saxton-Rawls values for 40% sand, 20% clay, 15 g/kg SOC, 5% cfvo
        return pd.DataFrame(
            {
                "soil_sand": [40.0] * n,
                "soil_clay": [20.0] * n,
                "soil_soc": [15.0] * n,
                "soil_cfvo": [5.0] * n,
                "soil_theta_wp": [0.130640] * n,
                "soil_theta_fc": [0.266469] * n,
                "soil_theta_sat": [0.438513] * n,
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
        df = pd.DataFrame(
            {
                "soil_theta_wp": [0.10] * n,
                "soil_theta_fc": [0.105] * n,  # fc - wp = 0.005 < 0.01
                "volumetric_soil_water_layer_2": np.full(n, 0.2),
            }
        )
        result, feats = apply_feature_engineering(df, ["rew"])
        assert "rew" not in feats

    def test_rew_with_nan_soil_data(self):
        """NaN soil hydraulic params → REW skipped."""
        n = 5
        df = pd.DataFrame(
            {
                "soil_theta_wp": [np.nan] * n,
                "soil_theta_fc": [np.nan] * n,
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
                "volumetric_soil_water_layer_4": rng.uniform(0.15, 0.3, n),
                "soil_temperature_level_1": rng.uniform(275, 300, n),
                "soil_temperature_level_2": rng.uniform(275, 295, n),
                "soil_temperature_level_3": rng.uniform(275, 295, n),
                "soil_temperature_level_4": rng.uniform(278, 290, n),
                "soil_theta_wp": np.full(n, 0.130640),
                "soil_theta_fc": np.full(n, 0.266469),
                "potential_evaporation_hourly_sum": np.full(n, -0.001),
                "total_precipitation_hourly_sum": rng.uniform(0, 0.002, n),
                "vpd_max": rng.uniform(2.0, 4.0, n),
                "vpd_min": rng.uniform(0.2, 1.0, n),
                "mean_annual_temp": np.full(n, 12.0),
                "mean_annual_precip": np.full(n, 750.0),
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
            "soil_hydraulics_extended",
            "atm_demand_extended",
            "plant_hydraulics",
            "swc_memory",
            "temporal_anomalies",
            "cross_interactions",
            "bioclimatic",
            "tree_metadata",
        ]
        result, feats = apply_feature_engineering(df, all_groups)
        assert len(feats) > 50  # Should create many features with new groups
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


class TestApplyAllFeatureEngineering:
    """Test the unified apply_all_feature_engineering entry point."""

    @pytest.fixture
    def full_df(self):
        rng = np.random.RandomState(42)
        n = 60
        return pd.DataFrame(
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
                "ppfd_in": rng.uniform(0, 2000, n),
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
                "latitude": np.full(n, 51.0),
                "sap_velocity": rng.uniform(0, 50, n),
            }
        )

    def test_returns_tuple(self, full_df):
        result_df, new_features = apply_all_feature_engineering(full_df)
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(new_features, list)
        assert len(new_features) > 30

    def test_includes_all_group_features(self, full_df):
        _, feats = apply_all_feature_engineering(full_df)
        assert "vpd_x_sw_in" in feats  # interactions
        assert "ta_lag1d" in feats  # lags_1d
        assert "ta_roll3d_mean" in feats  # rolling_3d
        assert "clear_sky_index" in feats  # physics

    def test_shared_scalar_features(self, full_df):
        _, feats = apply_all_feature_engineering(full_df)
        assert "vpd_log" in feats
        assert "dew_point" in feats
        assert "dew_point_depression" in feats
        assert "tropical" in feats
        assert "boreal" in feats
        assert "southern_hemisphere" in feats

    def test_daily_scale_aware_names(self, full_df):
        _, feats = apply_all_feature_engineering(full_df, time_scale="daily")
        # Daily lags: _lag1d
        assert "ta_lag1d" in feats
        assert "vpd_lag1d" in feats
        # Daily rolling: _roll{3,7,14}d
        assert "ta_roll3d_mean" in feats
        assert "vpd_roll7d_std" in feats
        assert "sw_in_roll14d_mean" in feats
        # Daily precip: _3d/_7d, days_since_rain
        assert "precip_sum_3d" in feats
        assert "precip_sum_7d" in feats
        assert "days_since_rain" in feats
        # Daily extras
        assert "ta_change_1d" in feats
        assert "sw_in_cumsum_7d" in feats
        assert "vpd_cumsum_3d" in feats
        # Must NOT have hourly names
        assert "ta_lag1h" not in feats
        assert "ta_roll3h_mean" not in feats

    def test_hourly_scale_aware_names(self, full_df):
        _, feats = apply_all_feature_engineering(full_df, time_scale="hourly")
        # Hourly lags: 1h, 3h, 6h, 12h, 24h
        for lag in [1, 3, 6, 12, 24]:
            assert f"ta_lag{lag}h" in feats
            assert f"vpd_lag{lag}h" in feats
        # Hourly rolling: 3h, 6h, 12h, 24h
        for w in [3, 6, 12, 24]:
            assert f"ta_roll{w}h_mean" in feats
            assert f"vpd_roll{w}h_std" in feats
        # Hourly precip: 24h/72h, hours_since_rain
        assert "precip_sum_24h" in feats
        assert "precip_sum_72h" in feats
        assert "hours_since_rain" in feats
        # Hourly extras
        assert "ta_change_1h" in feats
        assert "sw_in_cumsum_24h" in feats
        assert "vpd_cumsum_6h" in feats
        # Must NOT have daily names
        assert "ta_lag1d" not in feats
        assert "ta_roll3d_mean" not in feats
        assert "precip_sum_3d" not in feats

    def test_no_rows_dropped(self, full_df):
        n_before = len(full_df)
        result_df, _ = apply_all_feature_engineering(full_df)
        assert len(result_df) == n_before

    def test_binary_flags(self, full_df):
        result_df, _ = apply_all_feature_engineering(full_df)
        for col in ["tropical", "boreal", "southern_hemisphere"]:
            assert set(result_df[col].unique()).issubset({0.0, 1.0})

    def test_daily_no_is_daytime(self, full_df):
        """Daily scale must NOT include is_daytime."""
        _, feats = apply_all_feature_engineering(full_df, time_scale="daily")
        assert "is_daytime" not in feats

    def test_hourly_has_is_daytime(self, full_df):
        """Hourly scale must include is_daytime."""
        _, feats = apply_all_feature_engineering(full_df, time_scale="hourly")
        assert "is_daytime" in feats

    def test_symmetric_feature_categories(self, full_df):
        """Daily and hourly share the same non-temporal features; temporal names differ by scale."""
        _, daily_feats = apply_all_feature_engineering(full_df, time_scale="daily")
        _, hourly_feats = apply_all_feature_engineering(full_df, time_scale="hourly")
        # Non-temporal features identical across scales
        non_temporal = {
            "vpd_log",
            "dew_point",
            "dew_point_depression",
            "tropical",
            "boreal",
            "southern_hemisphere",
            "vpd_x_sw_in",
            "vpd_squared",
            "clear_sky_index",
            "gdd",
        }
        for feat in non_temporal:
            assert feat in daily_feats, f"{feat} missing from daily"
            assert feat in hourly_feats, f"{feat} missing from hourly"
        # is_daytime only in hourly
        assert "is_daytime" not in daily_feats
        assert "is_daytime" in hourly_feats

    def test_works_without_optional_cols(self):
        """Should not crash when optional columns (rh, latitude) are missing."""
        df = pd.DataFrame(
            {
                "vpd": [1.0, 2.0, 3.0],
                "ta": [20.0, 25.0, 30.0],
                "sw_in": [200.0, 300.0, 400.0],
            }
        )
        result_df, feats = apply_all_feature_engineering(df)
        assert isinstance(result_df, pd.DataFrame)
        assert "vpd_log" in feats
        assert "ta_change_1d" in feats
        assert "sw_in_cumsum_7d" in feats
        assert "dew_point" not in feats  # no rh
        assert "tropical" not in feats  # no latitude


# =====================================================================
# Tests for new feature groups (feature expansion)
# =====================================================================


class TestSoilHydraulicsExtended:
    """Test soil_hydraulics_extended feature group."""

    @pytest.fixture
    def hydro_df(self):
        rng = np.random.RandomState(42)
        n = 50
        _swc2 = rng.uniform(0.1, 0.4, n)
        return pd.DataFrame(
            {
                "soil_theta_wp": [0.130640] * n,
                "soil_theta_fc": [0.266469] * n,
                "soil_theta_sat": [0.438513] * n,
                "volumetric_soil_water_layer_1": rng.uniform(0.1, 0.4, n),
                "volumetric_soil_water_layer_2": _swc2 / _swc2.std(),  # variance-normalised
                "volumetric_soil_water_layer_2_raw": _swc2,  # physical m³/m³
                "volumetric_soil_water_layer_3": rng.uniform(0.15, 0.35, n),
                "volumetric_soil_water_layer_4": rng.uniform(0.2, 0.3, n),
                "soil_temperature_level_1": rng.uniform(275, 300, n),  # Kelvin
                "soil_temperature_level_4": rng.uniform(278, 290, n),
            }
        )

    def test_creates_awc(self, hydro_df):
        result, feats = apply_feature_engineering(hydro_df, ["soil_hydraulics_extended"])
        assert "awc" in feats
        # AWC = (fc - wp) * 1000 = (0.266469 - 0.130640) * 1000 ≈ 135.8
        assert result["awc"].iloc[0] == pytest.approx(135.829, abs=0.1)

    def test_creates_available_water_and_deficit(self, hydro_df):
        result, feats = apply_feature_engineering(hydro_df, ["soil_hydraulics_extended"])
        assert "available_water" in feats
        assert "soil_water_deficit" in feats
        assert (result["available_water"] >= 0).all()
        assert (result["soil_water_deficit"] >= 0).all()

    def test_creates_root_zone_weighted(self, hydro_df):
        result, feats = apply_feature_engineering(hydro_df, ["soil_hydraulics_extended"])
        assert "root_zone_swc_weighted" in feats
        # Weighted mean should be between min and max of layers
        _min = min(hydro_df[f"volumetric_soil_water_layer_{i}"].min() for i in range(1, 5))
        _max = max(hydro_df[f"volumetric_soil_water_layer_{i}"].max() for i in range(1, 5))
        assert result["root_zone_swc_weighted"].min() >= _min - 0.01
        assert result["root_zone_swc_weighted"].max() <= _max + 0.01

    def test_creates_soil_temp_gradient(self, hydro_df):
        result, feats = apply_feature_engineering(hydro_df, ["soil_hydraulics_extended"])
        assert "soil_temp_gradient" in feats

    def test_soil_frozen_kelvin(self, hydro_df):
        result, feats = apply_feature_engineering(hydro_df, ["soil_hydraulics_extended"])
        assert "soil_frozen" in feats
        # All temps > 273.15 → frozen should be 0
        assert result["soil_frozen"].sum() == 0

    def test_soil_frozen_detects_freezing(self):
        n = 10
        df = pd.DataFrame(
            {"soil_temperature_level_1": [270.0] * n}  # Below 273.15 K
        )
        result, feats = apply_feature_engineering(df, ["soil_hydraulics_extended"])
        assert "soil_frozen" in feats
        assert (result["soil_frozen"] == 1.0).all()

    def test_skipped_without_soil_params(self):
        n = 10
        df = pd.DataFrame({"volumetric_soil_water_layer_2": np.full(n, 0.25)})
        result, feats = apply_feature_engineering(df, ["soil_hydraulics_extended"])
        assert "awc" not in feats
        assert "available_water" not in feats

    def test_available_water_skipped_without_raw_column(self):
        """available_water requires _raw SWC (m³/m³), not variance-normalised."""
        n = 10
        df = pd.DataFrame(
            {
                "soil_theta_wp": [0.130640] * n,
                "soil_theta_fc": [0.266469] * n,
                "volumetric_soil_water_layer_2": np.full(n, 3.0),  # normalised (x/σ)
                # No _raw column!
            }
        )
        result, feats = apply_feature_engineering(df, ["soil_hydraulics_extended"])
        assert "awc" in feats  # AWC only needs θ_fc - θ_wp
        assert "available_water" not in feats  # requires _raw
        assert "soil_water_deficit" not in feats  # requires _raw


class TestAtmDemandExtended:
    """Test atm_demand_extended feature group."""

    @pytest.fixture
    def atm_df(self):
        n = 30
        return pd.DataFrame(
            {
                "ta": np.full(n, 20.0),
                "ta_max": np.full(n, 25.0),
                "ta_min": np.full(n, 15.0),
                "rh": np.full(n, 60.0),
                "sw_in": np.full(n, 200.0),
                "ext_rad": np.full(n, 350.0),
                "elevation": np.full(n, 100.0),
                "vpd_max": np.full(n, 2.5),
                "vpd_min": np.full(n, 0.5),
            }
        )

    def test_creates_net_radiation(self, atm_df):
        result, feats = apply_feature_engineering(atm_df, ["atm_demand_extended"])
        assert "net_radiation" in feats

    def test_creates_priestley_taylor(self, atm_df):
        result, feats = apply_feature_engineering(atm_df, ["atm_demand_extended"])
        assert "priestley_taylor_pet" in feats
        # PT PET should be non-negative
        assert (result["priestley_taylor_pet"] >= 0).all()
        # Realistic range: 0-15 mm/day for daily
        assert result["priestley_taylor_pet"].mean() < 20

    def test_creates_diurnal_ranges_daily(self, atm_df):
        result, feats = apply_feature_engineering(atm_df, ["atm_demand_extended"], time_scale="daily")
        assert "diurnal_temp_range" in feats
        assert result["diurnal_temp_range"].iloc[0] == pytest.approx(10.0)
        assert "vpd_diurnal_range" in feats
        assert result["vpd_diurnal_range"].iloc[0] == pytest.approx(2.0)

    def test_no_diurnal_ranges_hourly(self, atm_df):
        result, feats = apply_feature_engineering(atm_df, ["atm_demand_extended"], time_scale="hourly")
        assert "diurnal_temp_range" not in feats
        assert "vpd_diurnal_range" not in feats

    def test_skipped_without_rh(self):
        n = 10
        df = pd.DataFrame(
            {
                "ta": np.full(n, 20.0),
                "sw_in": np.full(n, 200.0),
                "ext_rad": np.full(n, 350.0),
                "elevation": np.full(n, 100.0),
            }
        )
        result, feats = apply_feature_engineering(df, ["atm_demand_extended"])
        assert "net_radiation" not in feats
        assert "priestley_taylor_pet" not in feats


class TestPlantHydraulics:
    """Test plant_hydraulics feature group."""

    @pytest.fixture
    def plant_df(self):
        n = 30
        return pd.DataFrame(
            {
                "LAI": np.linspace(0.5, 6.0, n),
                "sw_in": np.full(n, 300.0),
            }
        )

    def test_creates_fAPAR(self, plant_df):
        result, feats = apply_feature_engineering(plant_df, ["plant_hydraulics"])
        assert "fAPAR" in feats
        # fAPAR bounded [0, 1)
        assert result["fAPAR"].min() >= 0
        assert result["fAPAR"].max() < 1.0

    def test_fAPAR_increases_with_lai(self, plant_df):
        result, _ = apply_feature_engineering(plant_df, ["plant_hydraulics"])
        assert result["fAPAR"].iloc[-1] > result["fAPAR"].iloc[0]

    def test_creates_radiation_per_leaf(self, plant_df):
        result, feats = apply_feature_engineering(plant_df, ["plant_hydraulics"])
        assert "radiation_per_leaf" in feats
        # Higher LAI → less radiation per leaf
        assert result["radiation_per_leaf"].iloc[-1] < result["radiation_per_leaf"].iloc[0]

    def test_creates_lai_change_rate(self, plant_df):
        result, feats = apply_feature_engineering(plant_df, ["plant_hydraulics"])
        assert "lai_change_rate" in feats
        # Linearly increasing LAI → constant positive change
        assert result["lai_change_rate"].iloc[1:].mean() > 0


class TestSwcMemory:
    """Test swc_memory feature group."""

    def test_daily_lags(self):
        n = 20
        df = pd.DataFrame({"volumetric_soil_water_layer_1": np.arange(n, dtype=float)})
        result, feats = apply_feature_engineering(df, ["swc_memory"], time_scale="daily")
        assert "swc_lag1d" in feats
        assert "swc_lag3d" in feats
        assert "swc_lag7d" in feats
        assert "swc_change_1d" in feats
        # Verify lag value
        assert result["swc_lag1d"].iloc[1] == pytest.approx(0.0)

    def test_hourly_lags(self):
        n = 30
        df = pd.DataFrame({"volumetric_soil_water_layer_1": np.arange(n, dtype=float)})
        result, feats = apply_feature_engineering(df, ["swc_memory"], time_scale="hourly")
        assert "swc_lag1h" in feats
        assert "swc_lag6h" in feats
        assert "swc_lag24h" in feats
        assert "swc_change_1h" in feats

    def test_no_rows_dropped(self):
        n = 10
        df = pd.DataFrame({"volumetric_soil_water_layer_1": np.arange(n, dtype=float)})
        result, _ = apply_feature_engineering(df, ["swc_memory"])
        assert len(result) == n


class TestTemporalAnomalies:
    """Test temporal_anomalies feature group."""

    def test_creates_anomalies(self):
        rng = np.random.RandomState(42)
        n = 100
        df = pd.DataFrame(
            {
                "ta": rng.uniform(10, 30, n),
                "vpd": rng.uniform(0.5, 3.0, n),
                "volumetric_soil_water_layer_1": rng.uniform(0.1, 0.4, n),
            }
        )
        result, feats = apply_feature_engineering(df, ["temporal_anomalies"])
        assert "ta_anomaly" in feats
        assert "vpd_anomaly" in feats
        assert "swc_anomaly" in feats
        assert "cumulative_gdd" in feats

    def test_anomalies_mean_near_zero(self):
        """Over a long enough window, anomalies should center near 0."""
        rng = np.random.RandomState(42)
        n = 200
        df = pd.DataFrame({"ta": rng.uniform(15, 25, n)})
        result, _ = apply_feature_engineering(df, ["temporal_anomalies"])
        # After burn-in period, anomaly mean should be close to 0
        assert abs(result["ta_anomaly"].iloc[50:].mean()) < 2.0

    def test_cumulative_gdd_non_decreasing(self):
        n = 30
        df = pd.DataFrame({"ta": np.full(n, 15.0)})  # GDD = 10 per day
        result, _ = apply_feature_engineering(df, ["temporal_anomalies"])
        gdd_vals = result["cumulative_gdd"].values
        assert all(gdd_vals[i] <= gdd_vals[i + 1] for i in range(len(gdd_vals) - 1))

    def test_cumulative_gdd_zero_below_base(self):
        n = 10
        df = pd.DataFrame({"ta": np.full(n, 3.0)})  # 3 < 5°C base
        result, _ = apply_feature_engineering(df, ["temporal_anomalies"])
        assert (result["cumulative_gdd"] == 0).all()


class TestCrossInteractions:
    """Test cross_interactions feature group."""

    @pytest.fixture
    def cross_df(self):
        n = 20
        return pd.DataFrame(
            {
                "vpd": np.full(n, 2.0),
                "LAI": np.full(n, 3.0),
                "sw_in": np.full(n, 300.0),
                "ta": np.full(n, 20.0),
                "volumetric_soil_water_layer_1": np.full(n, 0.25),
                "rew": np.full(n, 0.6),
                "et0": np.full(n, 4.0),
            }
        )

    def test_creates_all_cross_terms(self, cross_df):
        result, feats = apply_feature_engineering(cross_df, ["cross_interactions"])
        expected = [
            "vpd_x_swc",
            "vpd_x_rew",
            "lai_x_vpd",
            "lai_x_sw_in",
            "ta_x_swc",
            "et0_x_rew",
        ]
        for f in expected:
            assert f in feats, f"Missing cross-interaction: {f}"

    def test_vpd_x_swc_value(self, cross_df):
        result, _ = apply_feature_engineering(cross_df, ["cross_interactions"])
        assert result["vpd_x_swc"].iloc[0] == pytest.approx(2.0 * 0.25)

    def test_et0_x_rew_value(self, cross_df):
        result, _ = apply_feature_engineering(cross_df, ["cross_interactions"])
        assert result["et0_x_rew"].iloc[0] == pytest.approx(4.0 * 0.6)

    def test_skipped_without_rew(self):
        n = 10
        df = pd.DataFrame(
            {
                "vpd": np.full(n, 2.0),
                "volumetric_soil_water_layer_1": np.full(n, 0.25),
            }
        )
        result, feats = apply_feature_engineering(df, ["cross_interactions"])
        assert "vpd_x_swc" in feats
        assert "vpd_x_rew" not in feats  # no rew column


class TestBioclimatic:
    """Test bioclimatic feature group."""

    def test_creates_de_martonne(self):
        n = 10
        df = pd.DataFrame(
            {
                "mean_annual_temp": np.full(n, 15.0),
                "mean_annual_precip": np.full(n, 800.0),
            }
        )
        result, feats = apply_feature_engineering(df, ["bioclimatic"])
        assert "de_martonne_aridity" in feats
        # I_DM = 800 / (15 + 10) = 32.0
        assert result["de_martonne_aridity"].iloc[0] == pytest.approx(32.0)

    def test_skipped_when_mat_below_minus10(self):
        n = 5
        df = pd.DataFrame(
            {
                "mean_annual_temp": np.full(n, -11.0),
                "mean_annual_precip": np.full(n, 200.0),
            }
        )
        result, feats = apply_feature_engineering(df, ["bioclimatic"])
        # MAT + 10 = -1 < 0 → guard skips
        assert "de_martonne_aridity" not in feats

    def test_skipped_without_climate_data(self):
        df = pd.DataFrame({"ta": [20.0, 25.0]})
        result, feats = apply_feature_engineering(df, ["bioclimatic"])
        assert "de_martonne_aridity" not in feats


class TestApplyAllNewFeatures:
    """Test that apply_all_feature_engineering includes new features."""

    @pytest.fixture
    def comprehensive_df(self):
        rng = np.random.RandomState(42)
        n = 60
        return pd.DataFrame(
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
                "ppfd_in": rng.uniform(0, 2000, n),
                "canopy_height": rng.uniform(5, 30, n),
                "elevation": np.full(n, 200.0),
                "prcip/PET": rng.uniform(0.2, 2.0, n),
                "soil_sand": np.full(n, 40.0),
                "soil_clay": np.full(n, 20.0),
                "soil_soc": np.full(n, 15.0),
                "soil_cfvo": np.full(n, 5.0),
                "soil_theta_wp": np.full(n, 0.130640),
                "soil_theta_fc": np.full(n, 0.266469),
                "soil_theta_sat": np.full(n, 0.438513),
                "volumetric_soil_water_layer_1": rng.uniform(0.1, 0.4, n),
                "volumetric_soil_water_layer_2": rng.uniform(0.1, 0.4, n),
                "volumetric_soil_water_layer_3": rng.uniform(0.1, 0.4, n),
                "volumetric_soil_water_layer_4": rng.uniform(0.15, 0.3, n),
                "soil_temperature_level_1": rng.uniform(275, 300, n),
                "soil_temperature_level_2": rng.uniform(275, 295, n),
                "soil_temperature_level_3": rng.uniform(275, 295, n),
                "soil_temperature_level_4": rng.uniform(278, 290, n),
                "potential_evaporation_hourly_sum": np.full(n, -0.001),
                "total_precipitation_hourly_sum": rng.uniform(0, 0.002, n),
                "latitude": np.full(n, 51.0),
                "vpd_max": rng.uniform(2.0, 4.0, n),
                "vpd_min": rng.uniform(0.2, 1.0, n),
                "mean_annual_temp": np.full(n, 12.0),
                "mean_annual_precip": np.full(n, 750.0),
                "sap_velocity": rng.uniform(0, 50, n),
            }
        )

    def test_new_features_present(self, comprehensive_df):
        _, feats = apply_all_feature_engineering(comprehensive_df)
        # soil_hydraulics_extended
        assert "awc" in feats
        assert "root_zone_swc_weighted" in feats
        assert "soil_frozen" in feats
        # atm_demand_extended
        assert "net_radiation" in feats
        assert "priestley_taylor_pet" in feats
        assert "diurnal_temp_range" in feats
        # plant_hydraulics
        assert "fAPAR" in feats
        assert "radiation_per_leaf" in feats
        # swc_memory (daily)
        assert "swc_lag1d" in feats
        assert "swc_change_1d" in feats
        # temporal_anomalies
        assert "ta_anomaly" in feats
        assert "cumulative_gdd" in feats
        # cross_interactions
        assert "vpd_x_swc" in feats
        assert "lai_x_vpd" in feats
        # bioclimatic
        assert "de_martonne_aridity" in feats
        # derived scalars
        assert "vpd_change_1d" in feats
        assert "soil_atm_temp_diff" in feats

    def test_total_feature_count_increased(self, comprehensive_df):
        _, feats = apply_all_feature_engineering(comprehensive_df)
        # Previously ~55-65 features, now should be 80+
        assert len(feats) > 75

    def test_no_rows_dropped(self, comprehensive_df):
        n_before = len(comprehensive_df)
        result, _ = apply_all_feature_engineering(comprehensive_df)
        assert len(result) == n_before

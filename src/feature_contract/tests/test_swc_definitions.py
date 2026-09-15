"""Soil moisture has one definition per quantity per layer: a level column and an anomaly column."""

import pytest

import src.feature_contract.definitions as definitions
from src.feature_contract.definitions import (
    RETIRED_SWC_FEATURES,
    SWC_DERIVED_SOURCE,
    SWC_LAYERS,
    assert_no_retired_swc_features,
    swc_anomaly_col,
    swc_level_col,
)

CANONICAL = [swc_level_col(layer) for layer in SWC_LAYERS] + [swc_anomaly_col(layer) for layer in SWC_LAYERS]


class TestColumnNames:
    @pytest.mark.parametrize("layer", [1, 2, 3, 4])
    def test_level_and_anomaly_columns(self, layer):
        assert swc_level_col(layer) == "volumetric_soil_water_layer_%d_raw" % layer
        assert swc_anomaly_col(layer) == "volumetric_soil_water_layer_%d_zscore" % layer

    @pytest.mark.parametrize("layer", [0, 5, "1"])
    def test_an_unknown_layer_raises(self, layer):
        with pytest.raises(ValueError, match="layer"):
            swc_level_col(layer)
        with pytest.raises(ValueError, match="layer"):
            swc_anomaly_col(layer)

    def test_derived_features_read_the_top_layer_level(self):
        assert SWC_DERIVED_SOURCE == "volumetric_soil_water_layer_1_raw"

    def test_the_scale_switch_is_gone(self):
        for name in ("SWC_SCALES", "DEFAULT_SWC_SCALE", "SWC_SCALE_COMPANION", "rescale_swc_levels", "validate_swc_scale"):
            assert not hasattr(definitions, name), name


class TestRetiredNames:
    @pytest.mark.parametrize("layer", [1, 2, 3, 4])
    def test_the_bare_column_is_retired_with_both_replacements_named(self, layer):
        bare = "volumetric_soil_water_layer_%d" % layer
        with pytest.raises(ValueError) as err:
            assert_no_retired_swc_features([bare], "features.json")
        message = str(err.value)
        assert swc_level_col(layer) in message
        assert swc_anomaly_col(layer) in message
        assert "features.json" in message

    @pytest.mark.parametrize("suffix", ["_min", "_max", "_sum"])
    def test_the_bare_daily_statistics_are_retired(self, suffix):
        with pytest.raises(ValueError, match="volumetric_soil_water_layer_2_raw" + suffix):
            assert_no_retired_swc_features(["volumetric_soil_water_layer_2" + suffix], "list")

    @pytest.mark.parametrize("layer", [2, 3, 4])
    def test_swc_layer_norm_is_retired_for_the_anomaly_column(self, layer):
        with pytest.raises(ValueError, match=swc_anomaly_col(layer)):
            assert_no_retired_swc_features(["swc_layer%d_norm" % layer], "list")

    def test_every_offending_name_is_reported_at_once(self):
        with pytest.raises(ValueError) as err:
            assert_no_retired_swc_features(["vpd", "swc_layer3_norm", "volumetric_soil_water_layer_1"], "list")
        assert "swc_layer3_norm ->" in str(err.value)
        assert "volumetric_soil_water_layer_1 ->" in str(err.value)

    def test_canonical_and_unrelated_names_pass(self):
        names = CANONICAL + [col + stat for col in CANONICAL for stat in ("_min", "_max", "_sum")]
        names += ["vpd", "swc_lag1d", "swc_anomaly", "vpd_x_swc", "rew", "root_zone_swc_weighted"]
        assert assert_no_retired_swc_features(names, "list") is None

    def test_no_canonical_name_is_retired(self):
        """A canonical name in the retired table would refuse every feature list, including correct ones."""
        assert not set(CANONICAL) & set(RETIRED_SWC_FEATURES)

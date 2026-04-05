"""Tests for shap_constants extracted module."""

import pytest

from src.hyperparameter_optimization.shap_constants import (
    FEATURE_UNITS,
    PFT_COLORS,
    PFT_COLUMNS,
    PFT_FULL_NAMES,
    SHAP_UNITS,
    get_feature_label,
    get_feature_unit,
    get_shap_label,
)


class TestPFTConstants:
    """Test PFT constant integrity."""

    def test_pft_columns_count(self):
        assert len(PFT_COLUMNS) == 8

    def test_pft_full_names_keys_match_columns(self):
        assert set(PFT_FULL_NAMES.keys()) == set(PFT_COLUMNS)

    def test_pft_colors_keys_match_columns(self):
        assert set(PFT_COLORS.keys()) == set(PFT_COLUMNS)

    def test_pft_colors_are_hex(self):
        for color in PFT_COLORS.values():
            assert color.startswith("#")
            assert len(color) == 7


class TestFeatureUnits:
    """Test FEATURE_UNITS dictionary."""

    def test_all_pfts_have_units(self):
        for pft in PFT_COLUMNS:
            assert pft in FEATURE_UNITS

    def test_key_variables_present(self):
        required = ["sap_velocity", "ta", "vpd", "sw_in", "LAI", "canopy_height"]
        for var in required:
            assert var in FEATURE_UNITS, f"Missing {var}"

    def test_shap_units_matches_sap_velocity(self):
        assert FEATURE_UNITS["sap_velocity"] == SHAP_UNITS


class TestGetFeatureUnit:
    """Test get_feature_unit with base and windowed names."""

    def test_base_name(self):
        assert get_feature_unit("vpd") == "kPa"
        assert get_feature_unit("ta") == "°C"

    def test_windowed_name(self):
        assert get_feature_unit("vpd_t-0") == "kPa"
        assert get_feature_unit("ta_t-2") == "°C"
        assert get_feature_unit("sw_in_t-1") == "W m⁻²"

    def test_unknown_feature(self):
        assert get_feature_unit("nonexistent_feature") == ""

    def test_pft_unitless(self):
        assert get_feature_unit("ENF") == ""


class TestGetFeatureLabel:
    """Test get_feature_label formatting."""

    def test_with_unit(self):
        assert get_feature_label("vpd") == "vpd (kPa)"

    def test_without_unit_flag(self):
        assert get_feature_label("vpd", include_unit=False) == "vpd"

    def test_unitless_feature(self):
        # PFT columns have empty unit strings — should NOT add parens
        assert get_feature_label("ENF") == "ENF"


class TestGetShapLabel:
    """Test get_shap_label formatting."""

    def test_with_unit(self):
        label = get_shap_label()
        assert "SHAP Value" in label
        assert SHAP_UNITS in label

    def test_without_unit(self):
        label = get_shap_label(include_unit=False)
        assert label == "SHAP Value"

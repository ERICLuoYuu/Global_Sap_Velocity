"""
SHAP analysis constants: PFT definitions, feature units, and label helpers.

Extracted from test_hyperparameter_tuning_ML_spatial_stratified.py for reuse
across training, prediction, and standalone SHAP analysis scripts.
"""

import re

# =============================================================================
# PFT (Plant Functional Type) DEFINITIONS
# =============================================================================

PFT_COLUMNS = ["MF", "DNF", "ENF", "EBF", "WSA", "WET", "DBF", "SAV"]

PFT_FULL_NAMES = {
    "MF": "Mixed Forest",
    "DNF": "Deciduous Needleleaf Forest",
    "ENF": "Evergreen Needleleaf Forest",
    "EBF": "Evergreen Broadleaf Forest",
    "WSA": "Woody Savanna",
    "WET": "Wetland",
    "DBF": "Deciduous Broadleaf Forest",
    "SAV": "Savanna",
}

PFT_COLORS = {
    "MF": "#1f77b4",  # blue
    "DNF": "#ff7f0e",  # orange
    "ENF": "#2ca02c",  # green
    "EBF": "#d62728",  # red
    "WSA": "#9467bd",  # purple
    "WET": "#8c564b",  # brown
    "DBF": "#e377c2",  # pink
    "SAV": "#7f7f7f",  # gray
}

# =============================================================================
# FEATURE UNITS
# =============================================================================

FEATURE_UNITS = {
    # Target variable
    "sap_velocity": "cm³ cm⁻² h⁻¹",
    # Meteorological variables
    "sw_in": "W m⁻²",
    "ppfd_in": "µmol m⁻² s⁻¹",
    "ext_rad": "W m⁻²",
    "ta": "°C",
    "vpd": "kPa",
    "ws": "m s⁻¹",
    "rh": "%",
    "precip": "mm",
    "precipitation": "mm",
    # Soil variables
    "volumetric_soil_water_layer_1": "m³ m⁻³",
    "soil_temperature_level_1": "K",
    "swc": "m³ m⁻³",
    # Vegetation/Site characteristics
    "canopy_height": "m",
    "elevation": "m",
    "LAI": "m² m⁻²",
    "latitude": "°",
    "longitude": "°",
    "prcip/PET": "dimensionless",
    # PFT categories (one-hot encoded, unitless)
    "MF": "",
    "DNF": "",
    "ENF": "",
    "EBF": "",
    "WSA": "",
    "WET": "",
    "DBF": "",
    "SAV": "",
    # Time features (cyclical, unitless)
    "Day sin": "",
    "Day cos": "",
    "Year sin": "",
    "Year cos": "",
    "Week sin": "",
    "Week cos": "",
    "Month sin": "",
    "Month cos": "",
}

# SHAP value units (same as target variable)
SHAP_UNITS = "cm³ cm⁻² h⁻¹"


# =============================================================================
# LABEL HELPERS
# =============================================================================


def get_feature_unit(feature_name: str) -> str:
    """
    Get the unit for a feature, handling windowed feature names.

    Parameters:
    -----------
    feature_name : str
        Feature name, potentially with time suffix (e.g., 'vpd_t-0', 'ta_t-2')

    Returns:
    --------
    str
        Unit string for the feature
    """
    # Remove time suffix if present (e.g., '_t-0', '_t-1', '_t-2')
    base_name = re.sub(r"_t-\d+$", "", feature_name)
    return FEATURE_UNITS.get(base_name, "")


def get_feature_label(feature_name: str, include_unit: bool = True) -> str:
    """
    Get a formatted label for a feature including its unit.

    Parameters:
    -----------
    feature_name : str
        Feature name
    include_unit : bool
        Whether to include the unit in parentheses

    Returns:
    --------
    str
        Formatted label like 'VPD (kPa)' or just 'VPD'
    """
    unit = get_feature_unit(feature_name)
    if include_unit and unit:
        return f"{feature_name} ({unit})"
    return feature_name


def get_shap_label(include_unit: bool = True) -> str:
    """
    Get the label for SHAP values with units.

    Parameters:
    -----------
    include_unit : bool
        Whether to include the unit

    Returns:
    --------
    str
        Label like 'SHAP Value (cm³ cm⁻² h⁻¹)'
    """
    if include_unit:
        return f"SHAP Value ({SHAP_UNITS})"
    return "SHAP Value"

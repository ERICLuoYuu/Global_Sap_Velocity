"""Unified feature definitions for forward feature selection.

Merges features from add_sap_flow_features() and apply_feature_engineering()
into a single registry.  Each selectable unit maps to one or more column names
in the pre-computed feature matrix.

PFT one-hot (8 columns) is a single group; everything else is 1:1.
The API supports arbitrary grouping — swap in multi-column groups later.
"""

from __future__ import annotations

from collections import OrderedDict

# ---------------------------------------------------------------------------
# Mandatory features — always included, never candidates for selection
# ---------------------------------------------------------------------------
MANDATORY_FEATURES: list[str] = [
    "sw_in",
    "ppfd_in",
    "ta",
    "vpd",
    "ws",
    "ext_rad",
]

# ---------------------------------------------------------------------------
# PFT one-hot column names (always treated as a single group)
# ---------------------------------------------------------------------------
PFT_ONEHOT_COLS: list[str] = [
    "MF",
    "DNF",
    "ENF",
    "EBF",
    "WSA",
    "WET",
    "DBF",
    "SAV",
]

# ---------------------------------------------------------------------------
# All feature engineering group names to request from apply_feature_engineering
# ---------------------------------------------------------------------------
ALL_FEATURE_ENGINEERING_GROUPS: list[str] = [
    "interactions",
    "lags_1d",
    "rolling_3d",
    "rolling_7d",
    "rolling_14d",
    "physics",
    "precip_memory",
    "indicators",
    "root_zone_swc",
    "rew",
    "et0",
    "psi_soil",
    "cwd",
]

# ---------------------------------------------------------------------------
# Additional features to request during data loading (--additional_features)
# ---------------------------------------------------------------------------
ADDITIONAL_FEATURES: list[str] = [
    # Static
    "slope",
    "aspect_sin",
    "aspect_cos",
    "stand_age",
    "soil_sand",
    "soil_clay",
    "soil_soc",
    "soil_cfvo",
    "temp_seasonality",
    "precip_seasonality",
    "mean_annual_temp",
    "mean_annual_precip",
    # Dynamic — deeper soil layers + ERA5-Land extras
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "soil_temperature_level_4",
    "rh",
    "rh_max",
    "rh_min",
    "surface_pressure",
    "potential_evaporation_hourly_sum",
    "total_precipitation_hourly_sum",
]

# ---------------------------------------------------------------------------
# Candidate features — ordered dict: group_name -> list of column names
# Each group is one selectable unit in SFS.
# ---------------------------------------------------------------------------
CANDIDATE_FEATURES: OrderedDict[str, list[str]] = OrderedDict(
    [
        # -- Base features (individual) --
        ("precip", ["precip"]),
        ("ta_max", ["ta_max"]),
        ("ta_min", ["ta_min"]),
        ("vpd_max", ["vpd_max"]),
        ("vpd_min", ["vpd_min"]),
        ("canopy_height", ["canopy_height"]),
        ("elevation", ["elevation"]),
        ("LAI", ["LAI"]),
        ("prcip/PET", ["prcip/PET"]),
        ("volumetric_soil_water_layer_1", ["volumetric_soil_water_layer_1"]),
        ("soil_temperature_level_1", ["soil_temperature_level_1"]),
        ("day_length", ["day_length"]),
        # PFT one-hot = 1 group of 8 columns
        ("pft", PFT_ONEHOT_COLS),
        # -- Time features (8 individual) --
        ("Day sin", ["Day sin"]),
        ("Day cos", ["Day cos"]),
        ("Week sin", ["Week sin"]),
        ("Week cos", ["Week cos"]),
        ("Month sin", ["Month sin"]),
        ("Month cos", ["Month cos"]),
        ("Year sin", ["Year sin"]),
        ("Year cos", ["Year cos"]),
        # -- Additional static --
        ("slope", ["slope"]),
        ("aspect_sin", ["aspect_sin"]),
        ("aspect_cos", ["aspect_cos"]),
        ("stand_age", ["stand_age"]),
        ("soil_sand", ["soil_sand"]),
        ("soil_clay", ["soil_clay"]),
        ("temp_seasonality", ["temp_seasonality"]),
        ("precip_seasonality", ["precip_seasonality"]),
        ("mean_annual_temp", ["mean_annual_temp"]),
        ("mean_annual_precip", ["mean_annual_precip"]),
        # -- Additional dynamic --
        ("volumetric_soil_water_layer_2", ["volumetric_soil_water_layer_2"]),
        ("volumetric_soil_water_layer_3", ["volumetric_soil_water_layer_3"]),
        ("volumetric_soil_water_layer_4", ["volumetric_soil_water_layer_4"]),
        ("soil_temperature_level_2", ["soil_temperature_level_2"]),
        ("soil_temperature_level_3", ["soil_temperature_level_3"]),
        ("soil_temperature_level_4", ["soil_temperature_level_4"]),
        ("rh", ["rh"]),
        ("rh_max", ["rh_max"]),
        ("rh_min", ["rh_min"]),
        ("surface_pressure", ["surface_pressure"]),
        ("potential_evaporation_hourly_sum", ["potential_evaporation_hourly_sum"]),
        ("total_precipitation_hourly_sum", ["total_precipitation_hourly_sum"]),
        # -- Feature engineering: rolling statistics (ta/vpd/sw_in/rh) --
        ("ta_roll3d_mean", ["ta_roll3d_mean"]),
        ("ta_roll3d_std", ["ta_roll3d_std"]),
        ("vpd_roll3d_mean", ["vpd_roll3d_mean"]),
        ("vpd_roll3d_std", ["vpd_roll3d_std"]),
        ("sw_in_roll3d_mean", ["sw_in_roll3d_mean"]),
        ("sw_in_roll3d_std", ["sw_in_roll3d_std"]),
        ("rh_roll3d_mean", ["rh_roll3d_mean"]),
        ("rh_roll3d_std", ["rh_roll3d_std"]),
        ("ta_roll7d_mean", ["ta_roll7d_mean"]),
        ("ta_roll7d_std", ["ta_roll7d_std"]),
        ("vpd_roll7d_mean", ["vpd_roll7d_mean"]),
        ("vpd_roll7d_std", ["vpd_roll7d_std"]),
        ("sw_in_roll7d_mean", ["sw_in_roll7d_mean"]),
        ("sw_in_roll7d_std", ["sw_in_roll7d_std"]),
        ("rh_roll7d_mean", ["rh_roll7d_mean"]),
        ("rh_roll7d_std", ["rh_roll7d_std"]),
        ("ta_roll14d_mean", ["ta_roll14d_mean"]),
        ("ta_roll14d_std", ["ta_roll14d_std"]),
        ("vpd_roll14d_mean", ["vpd_roll14d_mean"]),
        ("vpd_roll14d_std", ["vpd_roll14d_std"]),
        ("sw_in_roll14d_mean", ["sw_in_roll14d_mean"]),
        ("sw_in_roll14d_std", ["sw_in_roll14d_std"]),
        ("rh_roll14d_mean", ["rh_roll14d_mean"]),
        ("rh_roll14d_std", ["rh_roll14d_std"]),
        # -- Feature engineering: eco-hydro --
        ("swc_layer2_norm", ["swc_layer2_norm"]),
        ("swc_layer3_norm", ["swc_layer3_norm"]),
        ("swc_layer4_norm", ["swc_layer4_norm"]),
        ("rew", ["rew"]),
        ("et0", ["et0"]),
        ("psi_soil", ["psi_soil"]),
        ("cwd", ["cwd"]),
        # -- Feature engineering: interactions --
        ("vpd_x_sw_in", ["vpd_x_sw_in"]),
        ("vpd_squared", ["vpd_squared"]),
        ("ta_x_vpd", ["ta_x_vpd"]),
        ("height_x_vpd", ["height_x_vpd"]),
        ("wind_x_vpd", ["wind_x_vpd"]),
        ("demand_x_supply", ["demand_x_supply"]),
        # -- Feature engineering: physics --
        ("clear_sky_index", ["clear_sky_index"]),
        ("gdd", ["gdd"]),
        ("absorbed_radiation", ["absorbed_radiation"]),
        # -- Feature engineering: precip memory --
        ("precip_sum_3d", ["precip_sum_3d"]),
        ("precip_sum_7d", ["precip_sum_7d"]),
        ("days_since_rain", ["days_since_rain"]),
        # -- Feature engineering: indicators --
        ("vpd_high", ["vpd_high"]),
        ("soil_moisture_rel", ["soil_moisture_rel"]),
        ("soil_dry", ["soil_dry"]),
        # -- Feature engineering: lags --
        ("ta_lag1d", ["ta_lag1d"]),
        ("vpd_lag1d", ["vpd_lag1d"]),
        ("sw_in_lag1d", ["sw_in_lag1d"]),
        ("precip_lag1d", ["precip_lag1d"]),
        ("rh_lag1d", ["rh_lag1d"]),
        # -- add_sap_flow_features unique outputs --
        ("is_daytime", ["is_daytime"]),
        ("southern_hemisphere", ["southern_hemisphere"]),
        ("vpd_mean_6h", ["vpd_mean_6h"]),
        ("vpd_log", ["vpd_log"]),
        ("sw_in_cumsum_day", ["sw_in_cumsum_day"]),
        ("vpd_cumsum_6h", ["vpd_cumsum_6h"]),
        ("ta_change_1h", ["ta_change_1h"]),
        ("tropical", ["tropical"]),
        ("boreal", ["boreal"]),
        ("dew_point", ["dew_point"]),
        ("dew_point_depression", ["dew_point_depression"]),
        ("precip_sum_24h", ["precip_sum_24h"]),
        ("precip_sum_72h", ["precip_sum_72h"]),
        ("hours_since_rain", ["hours_since_rain"]),
    ]
)


def build_feature_groups(
    feature_names: list[str],
) -> tuple[list[int], list[list[int]]]:
    """Convert the registry into column-index lists for mlxtend SFS.

    Parameters
    ----------
    feature_names : list[str]
        Ordered column names of the pre-computed feature matrix X
        (excludes target column).

    Returns
    -------
    mandatory_idx : list[int]
        Column indices of mandatory features (for ``fixed_features``).
    candidate_groups : list[list[int]]
        Each inner list holds column indices for one selectable group
        (for ``feature_groups``).
    """
    name_to_idx = {name: i for i, name in enumerate(feature_names)}

    # Mandatory indices
    mandatory_idx: list[int] = []
    for feat in MANDATORY_FEATURES:
        if feat in name_to_idx:
            mandatory_idx.append(name_to_idx[feat])

    # Candidate groups — only include groups whose columns actually exist
    candidate_groups: list[list[int]] = []
    for _group_name, col_names in CANDIDATE_FEATURES.items():
        indices = [name_to_idx[c] for c in col_names if c in name_to_idx]
        if indices:
            candidate_groups.append(indices)

    return mandatory_idx, candidate_groups


def get_candidate_group_names(feature_names: list[str]) -> list[str]:
    """Return the names of candidate groups that exist in the feature matrix.

    Parameters
    ----------
    feature_names : list[str]
        Column names of the feature matrix.

    Returns
    -------
    list[str]
        Group names whose columns are present.
    """
    name_set = set(feature_names)
    names: list[str] = []
    for group_name, col_names in CANDIDATE_FEATURES.items():
        if any(c in name_set for c in col_names):
            names.append(group_name)
    return names


def get_all_expected_columns() -> list[str]:
    """Return a flat list of all expected column names (mandatory + candidates).

    Useful for verifying the cached feature matrix covers the full registry.
    """
    cols = list(MANDATORY_FEATURES)
    for col_names in CANDIDATE_FEATURES.values():
        cols.extend(col_names)
    return cols

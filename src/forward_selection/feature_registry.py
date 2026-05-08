"""Unified feature definitions for forward feature selection.

Defines all candidate features produced by apply_all_feature_engineering().
Both daily and hourly feature names are included — build_feature_groups()
silently skips names not present in the cached data, so only the correct
scale's features participate in selection.

PFT one-hot (8 columns) is a single group; everything else is 1:1.
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
# Additional raw columns to load from CSV (beyond base features)
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
    # Pre-computed Saxton-Rawls hydraulic params (from merge pipeline)
    "soil_theta_wp",
    "soil_theta_fc",
    "soil_theta_sat",
    # Dynamic — deeper soil layers + ERA5-Land extras
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    # Raw SWC (m³/m³) — needed for physics-based features (REW, ψ_soil)
    "volumetric_soil_water_layer_1_raw",
    "volumetric_soil_water_layer_2_raw",
    "volumetric_soil_water_layer_3_raw",
    "volumetric_soil_water_layer_4_raw",
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

# Columns loaded only as inputs to feature engineering — drop before building X
INTERMEDIATE_ONLY: list[str] = [
    "soil_theta_wp",
    "soil_theta_fc",
    "soil_theta_sat",
    "volumetric_soil_water_layer_1_raw",
    "volumetric_soil_water_layer_2_raw",
    "volumetric_soil_water_layer_3_raw",
    "volumetric_soil_water_layer_4_raw",
]


# ---------------------------------------------------------------------------
# Candidate features — ordered dict: group_name -> list of column names
# Each group is one selectable unit in SFS.
# ---------------------------------------------------------------------------
def _pairs(*names: str) -> list[tuple[str, list[str]]]:
    """Helper: one-to-one name→[name] entries."""
    return [(n, [n]) for n in names]


CANDIDATE_FEATURES: OrderedDict[str, list[str]] = OrderedDict(
    [
        # -- Base features (individual, daily only marked) --
        ("precip", ["precip"]),
        ("ta_max", ["ta_max"]),  # daily only
        ("ta_min", ["ta_min"]),  # daily only
        ("vpd_max", ["vpd_max"]),  # daily only
        ("vpd_min", ["vpd_min"]),  # daily only
        ("canopy_height", ["canopy_height"]),
        ("elevation", ["elevation"]),
        ("LAI", ["LAI"]),
        ("prcip/PET", ["prcip/PET"]),
        ("volumetric_soil_water_layer_1", ["volumetric_soil_water_layer_1"]),
        ("soil_temperature_level_1", ["soil_temperature_level_1"]),
        ("day_length", ["day_length"]),  # daily only
        # PFT one-hot = 1 group of 8 columns
        ("pft", PFT_ONEHOT_COLS),
        # -- Time features (8 individual) --
        *_pairs("Day sin", "Day cos", "Week sin", "Week cos", "Month sin", "Month cos", "Year sin", "Year cos"),
        # -- Additional static --
        *_pairs(
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
        ),
        # -- Additional dynamic --
        *_pairs(
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
        ),
        # == Feature engineering: interactions (both scales) ==
        *_pairs("vpd_x_sw_in", "vpd_squared", "ta_x_vpd", "height_x_vpd", "wind_x_vpd", "demand_x_supply"),
        # == Feature engineering: physics (both scales) ==
        *_pairs("clear_sky_index", "gdd", "absorbed_radiation"),
        # == Feature engineering: indicators (both scales) ==
        *_pairs("vpd_high", "soil_moisture_rel", "soil_dry"),
        # == Feature engineering: eco-hydro (both scales) ==
        *_pairs("swc_layer2_norm", "swc_layer3_norm", "swc_layer4_norm", "rew", "et0", "psi_soil", "cwd"),
        # == Derived scalar features (both scales) ==
        *_pairs("vpd_log", "dew_point", "dew_point_depression", "tropical", "boreal", "southern_hemisphere"),
        # == Feature engineering: soil_hydraulics_extended (both scales) ==
        *_pairs(
            "awc",
            "available_water",
            "soil_water_deficit",
            "root_zone_swc_weighted",
            "soil_temp_gradient",
            "soil_frozen",
        ),
        # == Feature engineering: atm_demand_extended (both scales) ==
        *_pairs("net_radiation", "priestley_taylor_pet"),
        # == Feature engineering: plant_hydraulics (both scales) ==
        *_pairs("fAPAR", "lai_change_rate", "radiation_per_leaf"),
        # == Feature engineering: temporal_anomalies (both scales) ==
        *_pairs("ta_anomaly", "vpd_anomaly", "swc_anomaly", "cumulative_gdd"),
        # == Feature engineering: cross_interactions (both scales) ==
        *_pairs("vpd_x_swc", "vpd_x_rew", "lai_x_vpd", "lai_x_sw_in", "ta_x_swc", "et0_x_rew"),
        # == Feature engineering: bioclimatic (both scales) ==
        *_pairs("de_martonne_aridity"),
        # == Derived: soil-atmosphere temp diff (both scales) ==
        *_pairs("soil_atm_temp_diff"),
        # =====================================================================
        # DAILY-specific temporal features
        # =====================================================================
        # -- Daily lags --
        *_pairs("ta_lag1d", "vpd_lag1d", "sw_in_lag1d", "precip_lag1d", "rh_lag1d"),
        # -- Daily rolling (3d, 7d, 14d) --
        *[
            (f"{v}_roll{w}d_{s}", [f"{v}_roll{w}d_{s}"])
            for v in ("ta", "vpd", "sw_in", "rh")
            for w in (3, 7, 14)
            for s in ("mean", "std")
        ],
        # -- Daily precip memory --
        *_pairs("precip_sum_3d", "precip_sum_7d", "days_since_rain"),
        # -- Daily temporal extras --
        *_pairs("ta_change_1d", "sw_in_cumsum_7d", "vpd_cumsum_3d"),
        # -- Daily vpd_change + swc_memory + diurnal ranges --
        *_pairs("vpd_change_1d", "swc_lag1d", "swc_lag3d", "swc_lag7d", "swc_change_1d"),
        *_pairs("diurnal_temp_range", "vpd_diurnal_range"),
        # =====================================================================
        # HOURLY-specific temporal features
        # =====================================================================
        # -- Hourly lags (1h, 3h, 6h, 12h, 24h) --
        *[
            (f"{v}_lag{h}h", [f"{v}_lag{h}h"])
            for v in ("ta", "vpd", "sw_in", "precip", "rh")
            for h in (1, 3, 6, 12, 24)
        ],
        # -- Hourly rolling (3h, 6h, 12h, 24h) --
        *[
            (f"{v}_roll{h}h_{s}", [f"{v}_roll{h}h_{s}"])
            for v in ("ta", "vpd", "sw_in", "rh")
            for h in (3, 6, 12, 24)
            for s in ("mean", "std")
        ],
        # -- Hourly precip memory --
        *_pairs("precip_sum_24h", "precip_sum_72h", "hours_since_rain"),
        # -- Hourly temporal extras --
        *_pairs("ta_change_1h", "sw_in_cumsum_24h", "vpd_cumsum_6h"),
        # -- Hourly vpd_change + swc_memory --
        *_pairs("vpd_change_1h", "swc_lag1h", "swc_lag6h", "swc_lag24h", "swc_change_1h"),
        # -- Hourly only --
        ("is_daytime", ["is_daytime"]),
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

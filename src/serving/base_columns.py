"""The base columns of a served daily frame: the merged-tree columns that feature engineering reads.

Every name is a column of ``merged/<sweep>/<config>/daily/*_daily.csv``, except ``latitude`` and
``longitude``, which the trainer writes as the medians of ``latitude_x`` and ``longitude_x``.
:data:`SOURCE` says how a prediction cell produces each one.

:data:`SENSOR_SUBSTITUTED` names the columns that were a site SENSOR in training and are ERA5-Land at a
cell: the same name and unit, a different measurement. That is the one train/serve difference serving
cannot remove, only measure (:mod:`src.serving.site_parity`).
"""

from __future__ import annotations

from src.feature_contract.definitions import SWC_LAYERS, swc_anomaly_col, swc_level_col

TIME_COLUMN = "solar_TIMESTAMP"
DAY_COLUMN = "TIMESTAMP"
PFT_COLUMN = "pft"
CELL_COLUMNS = ("latitude", "longitude", PFT_COLUMN)

SOIL_TEMPERATURE_COLUMNS = tuple(f"soil_temperature_level_{layer}" for layer in SWC_LAYERS)

ERA5_COLUMNS = (
    "ta",
    "ta_min",
    "ta_max",
    "vpd",
    "vpd_min",
    "vpd_max",
    "rh",
    "rh_min",
    "rh_max",
    "ws",
    "sw_in",
    "ppfd_in",
    "total_precipitation_hourly",
    "total_precipitation_hourly_sum",
    "potential_evaporation_hourly_sum",
    "prcip/PET",
    "surface_pressure",
    *SOIL_TEMPERATURE_COLUMNS,
    *(swc_level_col(layer) for layer in SWC_LAYERS),
    *(swc_anomaly_col(layer) for layer in SWC_LAYERS),
)

SOLAR_COLUMNS = ("ext_rad", "day_length")

LAI_COLUMNS = ("LAI",)

STATIC_COLUMNS = (
    "elevation",
    "canopy_height",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "stand_age",
    "soil_sand",
    "soil_clay",
    "soil_soc",
    "soil_cfvo",
    "soil_theta_wp",
    "soil_theta_fc",
    "soil_theta_sat",
    "mean_annual_temp",
    "mean_annual_precip",
    "temp_seasonality",
    "precip_seasonality",
)

SENSOR_SUBSTITUTED = frozenset(
    {"ta", "ta_min", "ta_max", "vpd", "vpd_min", "vpd_max", "rh", "rh_min", "rh_max", "ws", "sw_in", "ppfd_in"}
)

BASE_COLUMNS = (TIME_COLUMN, *CELL_COLUMNS, *ERA5_COLUMNS, *SOLAR_COLUMNS, *LAI_COLUMNS, *STATIC_COLUMNS)

SOURCE = {
    TIME_COLUMN: "calendar",
    **{name: "cell" for name in CELL_COLUMNS},
    **{name: "era5_land" for name in ERA5_COLUMNS},
    **{name: "solar_geometry" for name in SOLAR_COLUMNS},
    **{name: "globmap_lai" for name in LAI_COLUMNS},
    **{name: "static_layer" for name in STATIC_COLUMNS},
}


def source_of(name: str) -> str:
    """The serving source of base column ``name``; ``KeyError`` for a name that is not a base column."""
    if name not in SOURCE:
        raise KeyError(f"{name!r} is not a base column of a served daily frame")
    return SOURCE[name]

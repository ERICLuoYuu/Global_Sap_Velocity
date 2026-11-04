"""Fan 2017 rooting depth and hydrology: today's reader unchanged, with its fallback turned into a reason code."""

from __future__ import annotations

import pandas as pd
from src.static_features.results import NEIGHBOUR, NO_DATA, OK, ExtractionFailure, ProductResult, RunContext

FEATURES = (
    "root_depth",
    "infiltration_depth",
    "deep_drainage",
    "drainage_frequency",
    "regolith_flush_rate",
    "gw_residence_time",
)
DETAILS = (
    "root_depth_mask_valid",
    "root_depth_fallback_px",
    "root_depth_continent",
    "root_depth_lat_pixel",
    "root_depth_lon_pixel",
)
COLUMNS = FEATURES + DETAILS
PROVENANCE = {
    "source": "http://thredds-gfnl.usc.es/thredds/dodsC/INFILTRATION_DATA",
    "version": "Fan et al. 2017, PNAS, 30 arc-seconds by continent",
    "checksum": None,
    "time_window": None,
    "method": (
        "Native pixel with the reader's own mask-aware search over 3 pixels, in the first continent file whose box "
        "holds the site and whose mask has a valid pixel within reach: boxes overlap, and French Guiana lies in "
        "NAMERICA's, whose file is empty there. Fallback 0 is ok, above 0 neighbour, -1 no_data."
    ),
}


def status_from_fallback(radius: int) -> str:
    if radius < 0:
        return NO_DATA
    return OK if radius == 0 else NEIGHBOUR


def _row(result) -> tuple[dict, dict]:
    """(values, status) of one site. A detail column is ok wherever it holds a value, whatever the fallback."""
    values = {
        **{name: result.values.get(name) for name in FEATURES},
        "root_depth_mask_valid": result.mask_valid,
        "root_depth_fallback_px": result.fallback_radius_px,
        "root_depth_continent": result.continent,
        "root_depth_lat_pixel": result.lat_pixel,
        "root_depth_lon_pixel": result.lon_pixel,
    }
    feature_status = status_from_fallback(result.fallback_radius_px)
    status = {}
    for name, value in values.items():
        if pd.isna(value):
            status[name] = NO_DATA
        else:
            status[name] = feature_status if name in FEATURES else OK
    return values, status


def extract(context: RunContext) -> ProductResult:
    from src.Extractors import extract_root_depth_fan2017 as reader

    cache: dict = {}
    value_rows, status_rows = [], []
    try:
        for site in context.sites.itertuples(index=False):
            try:
                result = reader.fetch_all_variables_at_point(site.lat, site.lon, site.site_name, dataset_cache=cache)
            except Exception as error:
                raise ExtractionFailure(f"fan {site.site_name}: {error}") from error
            values, status = _row(result)
            value_rows.append(values)
            status_rows.append(status)
    finally:
        for datasets in cache.values():
            for dataset in datasets.values():
                dataset.close()

    index = pd.Index(context.sites["site_name"], name="site_name")
    return ProductResult(
        product="fan",
        values=pd.DataFrame(value_rows, index=index, columns=list(COLUMNS)).astype(dict.fromkeys(FEATURES, float)),
        status=pd.DataFrame(status_rows, index=index, columns=list(COLUMNS)),
        coverage=pd.DataFrame(index=index),
        by_year=None,
        extras={},
        provenance=PROVENANCE,
    )

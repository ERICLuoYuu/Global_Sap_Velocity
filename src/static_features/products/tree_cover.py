"""MOD44B tree cover: the cell mean of every year, and the mean over 2010-2012 as the feature."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.static_features.earth_engine import initialise, mean_and_count, reduce_cells, with_all_band
from src.static_features.results import BY_YEAR_COLUMNS, NO_DATA, OK, ProductResult, RunContext

ASSET = "MODIS/061/MOD44B"
BAND = "Percent_Tree_Cover"
COLUMN = "tree_cover_pct"
FIRST_YEAR = 2000
LAST_YEAR = 2025
WINDOW = (2010, 2011, 2012)
PROVENANCE = {
    "source": ASSET,
    "version": "MOD44B.061",
    "checksum": None,
    "time_window": "2010-2012",
    "method": (
        "Unweighted mean over pixels centred in the 30 arc-second cell, in the product's own projection, per year; "
        "then the mean of the window's years with pixels. 0 is a value; water and fill are masked in the asset, so "
        "coverage is the share of the cell's pixels holding a value, averaged over the window's years."
    ),
}


def window_result(yearly: pd.DataFrame) -> ProductResult:
    """yearly has one row per site and year, with the band's mean and count, and all_count."""
    index = pd.Index(yearly["site_name"].drop_duplicates(), name="site_name")
    mean, count = yearly[f"{BAND}_mean"], yearly[f"{BAND}_count"]
    in_window = yearly.assign(share=count / yearly["all_count"])[yearly["year"].isin(WINDOW)].groupby("site_name")
    values = in_window[f"{BAND}_mean"].mean().reindex(index)
    by_year = yearly.assign(column=COLUMN, value=mean, status=np.where(count > 0, OK, NO_DATA))
    return ProductResult(
        product="tree_cover",
        values=pd.DataFrame({COLUMN: values}),
        status=pd.DataFrame({COLUMN: np.where(values.notna(), OK, NO_DATA)}, index=index),
        coverage=pd.DataFrame({COLUMN: in_window["share"].mean().reindex(index)}),
        by_year=by_year[list(BY_YEAR_COLUMNS)],
        extras={},
        provenance=PROVENANCE,
    )


def extract(context: RunContext) -> ProductResult:
    import ee

    initialise(context.project)
    collection = ee.ImageCollection(ASSET).select(BAND)
    frames = []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        image = collection.filterDate(f"{year}-01-01", f"{year + 1}-01-01").first()
        cells = reduce_cells(
            with_all_band(image),
            mean_and_count(),
            image.projection(),
            context.sites,
            [f"{BAND}_count", "all_count"],
            [f"{BAND}_mean"],
            f"tree cover {year}",
        )
        frames.append(cells.reset_index().assign(year=year))
    return window_result(pd.concat(frames, ignore_index=True))

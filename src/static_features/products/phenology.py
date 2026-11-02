"""MCD12Q2 phenology: every pixel-year in a cell is stored, and summarised over land pixels, then over 2001-2020."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.static_features.earth_engine import initialise, land_mask, sample_cells
from src.static_features.results import BY_YEAR_COLUMNS, NO_DATA, OK, ExtractionFailure, ProductResult, RunContext

ASSET = "MODIS/061/MCD12Q2"
FIRST_PRODUCT_YEAR = 2001
LAST_PRODUCT_YEAR = 2025
FIRST_YEAR = 2001
LAST_YEAR = 2020
FILL = -9999
DAYS_PER_YEAR = 365.25
EVI2_SCALE = 0.0001
EVI2_AREA_SCALE = 0.1
PIXEL_KEY = ["longitude", "latitude"]
PIXEL_KEY_DECIMALS = 6
LAND_BATCH_SIZE = 1
DATE_BANDS = {
    "Greenup": "phen_greenup",
    "MidGreenup": "phen_midgreenup",
    "Maturity": "phen_maturity",
    "Peak": "phen_peak",
    "Senescence": "phen_senescence",
    "MidGreendown": "phen_midgreendown",
    "Dormancy": "phen_dormancy",
}
CYCLE_BANDS = (*DATE_BANDS, "EVI_Minimum", "EVI_Amplitude", "EVI_Area", "QA_Overall")
BANDS = ["NumCycles"] + [f"{band}_{cycle}" for cycle in (1, 2) for band in CYCLE_BANDS]
CAUTION = (
    "The MCD12Q2 user guide warns that Greenup and Dormancy can be anomalous at high latitude with low amplitude, and "
    "recommends the mid-season dates there."
)


def with_land(pixels: pd.DataFrame, land: pd.DataFrame) -> pd.DataFrame:
    """Each pixel-year with its land value, in the order given, matched on site and coordinates, never on row order.

    A pixel found on only one side stops the run: a pixel with no land value would drop out of every average.
    """
    keys = ["site_name", *PIXEL_KEY]
    decimals = dict.fromkeys(PIXEL_KEY, PIXEL_KEY_DECIMALS)
    year, mask = pixels.round(decimals), land.round(decimals)[[*keys, "land"]]
    sides = year[keys].merge(mask[keys], on=keys, how="outer", indicator=True)
    unmatched = sorted(sides.loc[sides["_merge"] != "both", "site_name"].unique())
    if unmatched:
        raise ExtractionFailure(f"phenology: the year and land samples hold different pixels at {unmatched}")
    return year.merge(mask, on=keys, validate="one_to_one")


def main_cycle(pixels: pd.DataFrame) -> pd.DataFrame:
    """One row per pixel-year with the bands of its larger cycle, the first on a tie, and fill turned into missing."""
    has_cycle = pixels["NumCycles"] != FILL
    second = pixels["EVI_Amplitude_2"] > pixels["EVI_Amplitude_1"]
    picked = {}
    for band in CYCLE_BANDS:
        chosen = pixels[f"{band}_1"].where(~second, pixels[f"{band}_2"])
        picked[band] = chosen.where(has_cycle & (chosen != FILL)).astype(float)
    return pd.DataFrame(
        {
            "site_name": pixels["site_name"],
            "year": pixels["year"],
            "land": pixels["land"] == 1,
            "has_cycle": has_cycle,
            "num_cycles": pixels["NumCycles"].where(has_cycle, 0),
            **picked,
        }
    )


def _measures(cycle: pd.DataFrame) -> pd.DataFrame:
    """Every averaged measure, per pixel-year, so lengths never mix dates from different pixels."""
    columns = {}
    for band, name in DATE_BANDS.items():
        day_of_year = pd.to_datetime(cycle[band], unit="D").dt.dayofyear
        angle = 2 * np.pi * day_of_year / DAYS_PER_YEAR
        columns[f"{name}_sin"] = np.sin(angle)
        columns[f"{name}_cos"] = np.cos(angle)
    columns["phen_season_days"] = cycle["Dormancy"] - cycle["Greenup"]
    columns["phen_midseason_days"] = cycle["MidGreendown"] - cycle["MidGreenup"]
    columns["phen_evi2_min"] = cycle["EVI_Minimum"] * EVI2_SCALE
    columns["phen_evi2_amplitude"] = cycle["EVI_Amplitude"] * EVI2_SCALE
    columns["phen_evi2_area"] = cycle["EVI_Area"] * EVI2_AREA_SCALE
    columns["phen_num_cycles"] = cycle["num_cycles"].astype(float)
    columns["phen_cycle_share"] = cycle["has_cycle"].astype(float)
    return pd.DataFrame(columns, index=cycle.index)


def _by_year(index: pd.Index, per_year: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    every_year = pd.MultiIndex.from_product([index, years], names=["site_name", "year"])
    table = per_year.set_index(["site_name", "year"]).reindex(every_year).reset_index()
    long = table.melt(id_vars=["site_name", "year"], var_name="column", value_name="value")
    return long.assign(status=np.where(long["value"].notna(), OK, NO_DATA))[list(BY_YEAR_COLUMNS)]


def summarise(pixels: pd.DataFrame, first_year: int = FIRST_YEAR, last_year: int = LAST_YEAR) -> ProductResult:
    index = pd.Index(pixels["site_name"].drop_duplicates(), name="site_name")
    cycle = main_cycle(pixels)
    measures = _measures(cycle)
    columns = list(measures.columns)
    land_rows = pd.concat([cycle[["site_name", "year"]], measures], axis=1)[cycle["land"]]

    per_year = land_rows.groupby(["site_name", "year"], as_index=False)[columns].mean()
    window = per_year[per_year["year"].between(first_year, last_year)]
    with_cycle = (window["phen_cycle_share"] > 0).groupby(window["site_name"]).sum()
    values = window.groupby("site_name")[columns].mean().reindex(index).assign(phen_years_used=with_cycle)

    pixel_years = cycle[cycle["year"].between(first_year, last_year)].groupby("site_name").size()
    held = land_rows[land_rows["year"].between(first_year, last_year)].groupby("site_name")[columns].count()
    return ProductResult(
        product="phenology",
        values=values,
        status=pd.DataFrame(np.where(values.notna(), OK, NO_DATA), index=index, columns=values.columns),
        coverage=held.div(pixel_years, axis=0).reindex(index).fillna(0.0),
        by_year=_by_year(index, per_year, sorted(pixels["year"].unique())),
        extras={"phenology_pixels": pixels},
        provenance={
            "source": ASSET,
            "version": "MCD12Q2.061",
            "checksum": None,
            "time_window": f"{first_year}-{last_year}",
            "method": (
                "Every pixel centred in the 30 arc-second cell, on the product's own grid. Each pixel-year uses the "
                "cycle with the larger EVI2 amplitude. Dates become the sine and cosine of 2 pi doy / 365.25, not "
                "renormalised; lengths are taken per pixel-year. Averages run over land pixels, then over years. No "
                "quality filter. Coverage is the pixel-years holding the value over all pixel-years in the window."
            ),
            "caution": CAUTION,
        },
    )


def extract(context: RunContext) -> ProductResult:
    import ee

    initialise(context.project)
    collection = ee.ImageCollection(ASSET)
    coordinates = ee.Image.pixelLonLat()
    # WorldCover has one date, so it is read once. Its 10 m mask costs by the area a batch spans, not its sites: 60
    # sites run out of memory and 20 can, while one takes 0.3 s.
    land = sample_cells(
        land_mask().addBands(coordinates),
        collection.first().select("Peak_1").projection(),
        context.sites,
        ["land", *PIXEL_KEY],
        "phenology land",
        batch_size=LAND_BATCH_SIZE,
    )
    frames = []
    for year in range(FIRST_PRODUCT_YEAR, LAST_PRODUCT_YEAR + 1):
        image = collection.filterDate(f"{year}-01-01", f"{year + 1}-01-01").first()
        stack = image.select(BANDS).unmask(FILL).addBands(coordinates)
        table = sample_cells(
            stack, image.select("Peak_1").projection(), context.sites, [*BANDS, *PIXEL_KEY], f"phenology {year}"
        )
        frames.append(with_land(table, land).assign(year=year))
    return summarise(pd.concat(frames, ignore_index=True))

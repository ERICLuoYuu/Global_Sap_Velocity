"""ESA CCI biomass v7: the native 1 km pixel with a 3-pixel neighbour rule, averaged over 2010-2012."""

from __future__ import annotations

import math
from collections.abc import Sequence

import netCDF4
import numpy as np
import pandas as pd
from src.static_features.fetch import Download, fetch, sha256_of
from src.static_features.rasters import PixelHit, pixel_index, pixel_with_reach, require_global_axes
from src.static_features.results import (
    BY_YEAR_COLUMNS,
    NEIGHBOUR,
    NO_DATA,
    OK,
    ExtractionFailure,
    ProductResult,
    RunContext,
)

URL = (
    "https://dap.ceda.ac.uk/neodc/esacci/biomass/data/agb/maps/v7.0/netcdf/ESACCI-BIOMASS-L4-AGB-MERGED-1000m-fv7.0.nc"
)
FILE_NAME = "ESACCI-BIOMASS-L4-AGB-MERGED-1000m-fv7.0.nc"
SIZE = 2983287549
DIRECTORY = "esa_cci_biomass_v7"
VARIABLES = {"agb_mg_ha": "agb", "agb_sd": "agb_sd"}
WINDOW = (2010, 2011, 2012)
REACH = 3


def window_value(hits: Sequence[PixelHit]) -> tuple[float, str]:
    """Mean of the years with a value: ok only if every one is the own pixel, neighbour if any is not."""
    found = [hit for hit in hits if hit.status != NO_DATA]
    if not found:
        return math.nan, NO_DATA
    status = OK if all(hit.status == OK for hit in found) else NEIGHBOUR
    return sum(hit.value for hit in found) / len(found), status


def _year_positions(dataset: netCDF4.Dataset) -> dict[int, int]:
    time = dataset.variables["time"]
    dates = netCDF4.num2date(time[:], time.units, calendar=getattr(time, "calendar", "standard"))
    positions = {date.year: position for position, date in enumerate(dates)}
    absent = [year for year in WINDOW if year not in positions]
    if absent:
        raise ExtractionFailure(f"cci: the file has years {sorted(positions)}, without {absent}")
    return positions


def _is_present(block: np.ndarray) -> np.ndarray:
    return ~np.isnan(block)  # 0 is a valid biomass


def _year_hits(variable, position: int, pixels: list[tuple[int, int]]) -> list[PixelHit]:
    def read_block(row_start: int, row_stop: int, col_start: int, col_stop: int) -> np.ndarray:
        block = variable[position, row_start:row_stop, col_start:col_stop]
        return np.ma.filled(block.astype("f8"), np.nan)

    return [pixel_with_reach(read_block, variable.shape[1:], row, col, REACH, _is_present) for row, col in pixels]


def _site_pixels(context: RunContext, n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    return [
        (pixel_index(site.lat, 90.0, -180.0 / n_rows, n_rows), pixel_index(site.lon, -180.0, 360.0 / n_cols, n_cols))
        for site in context.sites.itertuples(index=False)
    ]


def extract(context: RunContext) -> ProductResult:
    path = context.raw_root / DIRECTORY / FILE_NAME
    names = context.sites["site_name"].tolist()
    values, status, rows = {}, {}, []
    with netCDF4.Dataset(path) as dataset:
        positions = _year_positions(dataset)
        latitude, longitude = dataset.variables["lat"][:], dataset.variables["lon"][:]
        require_global_axes("cci", latitude, longitude)
        pixels = _site_pixels(context, latitude.size, longitude.size)
        for column, variable_name in VARIABLES.items():
            variable = dataset.variables[variable_name]
            hits = {year: _year_hits(variable, position, pixels) for year, position in positions.items()}
            window = [window_value([hits[year][index] for year in WINDOW]) for index in range(len(names))]
            values[column] = [value for value, _ in window]
            status[column] = [state for _, state in window]
            rows.extend(
                (name, year, column, year_hits[index].value, year_hits[index].status)
                for year, year_hits in hits.items()
                for index, name in enumerate(names)
            )

    index = pd.Index(names, name="site_name")
    return ProductResult(
        product="cci",
        values=pd.DataFrame(values, index=index),
        status=pd.DataFrame(status, index=index),
        coverage=pd.DataFrame(index=index),
        by_year=pd.DataFrame(rows, columns=list(BY_YEAR_COLUMNS)),
        extras={},
        provenance={
            "source": URL,
            "version": "ESA CCI Biomass v7.0, merged 1 km",
            "checksum": {FILE_NAME: sha256_of(path)},
            "time_window": "2010-2012",
            "method": (
                f"Native pixel, valid when not fill, 0 being a value; else the nearest valid pixel within {REACH}, "
                "longitude wrapping. Taken per year, then the mean of the window's years that have a value."
            ),
        },
    )


def fetch_inputs(context: RunContext) -> None:
    fetch(Download(URL, FILE_NAME, size=SIZE), context.raw_root / DIRECTORY)

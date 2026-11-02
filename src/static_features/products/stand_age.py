"""BGI forest age: the nearest node with no neighbour search, because the merge falls back to ground st_age on empty."""

from __future__ import annotations

import math

import h5py
import numpy as np
import pandas as pd
from src.static_features.results import NO_DATA, OK, ExtractionFailure, ProductResult, RunContext

THRESHOLDS = ("TC000", "TC010", "TC020", "TC030")
MAX_AGE = 500.0
PROVENANCE = {
    "source": "https://doi.org/10.17871/ForestAgeBGI.2021",
    "version": "BGIForestAge MPI-BGC 1.0.0 (Besnard et al. 2021)",
    "checksum": None,
    "time_window": "circa 2010",
    "method": (
        "Nearest node with no neighbour search, valid when 0 < age <= 500. A filled value would displace the ground "
        "st_age in the merge, which falls back to it only when the mean of these columns is empty."
    ),
}


def age_value(raw: float) -> float:
    return float(raw) if 0.0 < raw <= MAX_AGE else math.nan  # NaN fails the comparison


def _read_node(dataset: h5py.Dataset, lat_index: int, lon_index: int, n_lats: int) -> float:
    if dataset.shape[0] == n_lats:
        return np.asarray(dataset[lat_index, lon_index]).item()
    return np.asarray(dataset[lon_index, lat_index]).item()


def extract(context: RunContext) -> ProductResult:
    directory = context.raw_root / "stand_age"
    paths = sorted(directory.glob("*BGIForestAge*.nc"))
    if len(paths) != 1:
        raise ExtractionFailure(f"stand_age: expected one *BGIForestAge*.nc in {directory}, found {len(paths)}")

    sites = context.sites
    values = {}
    with h5py.File(paths[0], "r") as handle:
        lats = np.asarray(handle["latitude"][:]).flatten()
        lons = np.asarray(handle["longitude"][:]).flatten()
        for threshold in THRESHOLDS:
            dataset = handle[f"ForestAge_{threshold}"]
            column = []
            for site in sites.itertuples(index=False):
                lat_index = int(np.abs(lats - site.lat).argmin())
                lon_index = int(np.abs(lons - site.lon).argmin())
                column.append(age_value(_read_node(dataset, lat_index, lon_index, len(lats))))
            values[f"stand_age_{threshold}"] = column

    index = pd.Index(sites["site_name"], name="site_name")
    frame = pd.DataFrame(values, index=index)
    return ProductResult(
        product="stand_age",
        values=frame,
        status=frame.notna().replace({True: OK, False: NO_DATA}),
        coverage=pd.DataFrame(index=index),
        by_year=None,
        extras={},
        provenance=PROVENANCE,
    )

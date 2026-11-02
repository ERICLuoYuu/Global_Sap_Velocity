"""WorldClim bioclimate and Beck Koppen classes: the native pixel, with today's search, so values reproduce."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from src.static_features.rasters import PixelHit, pixel_with_reach, tiff_reader
from src.static_features.results import ExtractionFailure, ProductResult, RunContext

BIO_FILE = "spatial_features/bio_30s/wc2.1_30s_bio_{n}.tif"
KOPPEN_FILE = "koppen_geiger/koppen/Beck_KG_V1_present_0p0083.tif"
REACH = 3

# Beck's rasters declare no nodata and encode ocean as 0, so the sentinel has to be named.
KOPPEN_OCEAN_CODE = 0
KOPPEN_CLASSES = {
    1: "Af - Tropical rainforest",
    2: "Am - Tropical monsoon",
    3: "Aw - Tropical savanna",
    4: "BWh - Arid, desert, hot",
    5: "BWk - Arid, desert, cold",
    6: "BSh - Arid, steppe, hot",
    7: "BSk - Arid, steppe, cold",
    8: "Csa - Temperate, dry and hot summer",
    9: "Csb - Temperate, dry and warm summer",
    10: "Csc - Temperate, dry and cold summer",
    11: "Cwa - Temperate, dry winter and hot summer",
    12: "Cwb - Temperate, dry winter and warm summer",
    13: "Cwc - Temperate, dry winter and cold summer",
    14: "Cfa - Temperate, no dry season and hot summer",
    15: "Cfb - Temperate, no dry season and warm summer",
    16: "Cfc - Temperate, no dry season and cold summer",
    17: "Dsa - Cold, dry and hot summer",
    18: "Dsb - Cold, dry and warm summer",
    19: "Dsc - Cold, dry and cold summer",
    20: "Dsd - Cold, dry and very cold winter",
    21: "Dwa - Cold, dry winter and hot summer",
    22: "Dwb - Cold, dry winter and warm summer",
    23: "Dwc - Cold, dry winter and cold summer",
    24: "Dwd - Cold, dry winter and very cold winter",
    25: "Dfa - Cold, no dry season and hot summer",
    26: "Dfb - Cold, no dry season and warm summer",
    27: "Dfc - Cold, no dry season and cold summer",
    28: "Dfd - Cold, no dry season and very cold winter",
    29: "ET - Polar, tundra",
    30: "EF - Polar, frost",
}
PROVENANCE = {
    "source": "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_30s_bio.zip; Beck et al. 2018, Sci. Data 5:180214",
    "version": "WorldClim 2.1 bio 30 arc-seconds; Beck_KG_V1 present, 0.0083 degrees",
    "checksum": None,
    "time_window": "1970-2000 (WorldClim), 1980-2016 (Beck)",
    "method": "Native pixel holding the site; if invalid, the nearest valid pixel within 3, clipped at the raster's edge.",
}


def _read_hits(path: Path, sites: pd.DataFrame, sentinel: float | None = None) -> list[PixelHit]:
    with rasterio.open(path) as src:
        nodata = src.nodata

        def is_valid(block: np.ndarray) -> np.ndarray:
            valid = ~np.isnan(block)
            if nodata is not None:
                valid &= block != nodata
            if sentinel is not None:
                valid &= block != sentinel
            return valid

        reader = tiff_reader(src, 1)
        hits = []
        for site in sites.itertuples(index=False):
            row, col = rowcol(src.transform, site.lon, site.lat)
            hits.append(pixel_with_reach(reader, src.shape, int(row), int(col), REACH, is_valid, wrap_cols=False))
    return hits


def _koppen_name(site_name: str, hit: PixelHit) -> str | None:
    if np.isnan(hit.value):
        return None
    code = int(hit.value)
    if code not in KOPPEN_CLASSES:
        raise ExtractionFailure(f"worldclim {site_name}: Koppen code {code} is outside the legend")
    return KOPPEN_CLASSES[code]


def extract(context: RunContext) -> ProductResult:
    sites = context.sites
    index = pd.Index(sites["site_name"], name="site_name")
    values, status = {}, {}
    for n in range(1, 20):  # WorldClim 2.1 publishes all 19 in final units, so none needs scaling
        hits = _read_hits(context.raw_root / BIO_FILE.format(n=n), sites)
        values[f"bio{n}"] = [hit.value for hit in hits]
        status[f"bio{n}"] = [hit.status for hit in hits]

    hits = _read_hits(context.raw_root / KOPPEN_FILE, sites, sentinel=KOPPEN_OCEAN_CODE)
    values["koppen_class"] = [hit.value for hit in hits]
    values["koppen_name"] = [_koppen_name(index[position], hit) for position, hit in enumerate(hits)]
    status["koppen_class"] = [hit.status for hit in hits]
    status["koppen_name"] = [hit.status for hit in hits]

    return ProductResult(
        product="worldclim",
        values=pd.DataFrame(values, index=index),
        status=pd.DataFrame(status, index=index),
        coverage=pd.DataFrame(index=index),
        by_year=None,
        extras={},
        provenance=PROVENANCE,
    )

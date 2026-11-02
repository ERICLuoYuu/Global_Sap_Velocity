"""Liang 2022 tree richness: the native 0.025 degree pixel, with a 1-pixel neighbour rule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from src.static_features.fetch import Download, fetch, sha256_of
from src.static_features.rasters import pixel_with_reach, tiff_reader
from src.static_features.results import ProductResult, RunContext

URL = "https://ndownloader.figshare.com/files/35133814"
FILE_NAME = "S_mean_raster.tif"
MD5 = "bf745222a79918df7e52e4b785ccd993"
REACH = 1


def _is_valid(block: np.ndarray) -> np.ndarray:
    return np.isfinite(block) & (block > 0)


def extract(context: RunContext) -> ProductResult:
    path = context.raw_root / "liang_2022" / FILE_NAME
    with rasterio.open(path) as src:
        reader = tiff_reader(src, 1)
        hits = []
        for site in context.sites.itertuples(index=False):
            row, col = rowcol(src.transform, site.lon, site.lat)
            hits.append(pixel_with_reach(reader, src.shape, int(row), int(col), REACH, _is_valid))

    index = pd.Index(context.sites["site_name"], name="site_name")
    return ProductResult(
        product="liang",
        values=pd.DataFrame({"tree_richness_per_ha": [hit.value for hit in hits]}, index=index),
        status=pd.DataFrame({"tree_richness_per_ha": [hit.status for hit in hits]}, index=index),
        coverage=pd.DataFrame(index=index),
        by_year=None,
        extras={},
        provenance={
            "source": f"https://doi.org/10.6084/m9.figshare.17232491, {URL}",
            "version": "Liang et al. 2022, S_mean_raster",
            "checksum": {FILE_NAME: sha256_of(path)},
            "time_window": None,
            "method": "Native pixel, valid when finite and above 0; else the nearest valid pixel within 1, longitude wrapping.",
        },
    )


def fetch_inputs(context: RunContext) -> None:
    fetch(Download(URL, FILE_NAME, md5=MD5), context.raw_root / "liang_2022")

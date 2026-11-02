"""Lusk 2026 plant traits: all 31, the native 1 km pixel in EPSG:6933, with their uncertainty and applicability bands.

Bands are int16 with a per-band scale and offset, so nodata is tested on the raw value, before any scaling.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol
from src.static_features.fetch import fetch, sha256_of, zenodo_downloads
from src.static_features.rasters import pixel_with_reach, tiff_reader
from src.static_features.results import NO_DATA, ExtractionFailure, ProductResult, RunContext

RECORD_ID = 14646322
FILE_NAME = "X{trait_id}_mean_Shrub_Tree_Grass_1km.tif"
NODATA = -32768
REACH = 3
TRAIT_IDS = (
    282, 237, 47, 3113, 13, 146, 78, 55, 163, 144, 50, 14, 15, 46, 3120, 145,
    3106, 6, 95, 27, 26, 138, 3117, 1080, 614, 4, 169, 281, 21, 289, 297,
)  # fmt: skip
# Starred as experimental by the authors, below their r = 0.5 threshold. 1080 has r 0.46 and no star: as published.
EXPERIMENTAL = frozenset({282, 237, 47, 163, 144, 3120, 3106, 95, 27, 26, 138, 614, 21, 289, 297})
NAMED_COLUMNS = {6: "root_depth_lusk", 4: "wood_density_lusk"}
# As the files spell them. "g cm3" is g cm-3 with its exponent lost, as "cm g" is for SRL: the values run 0.26-0.71.
EXPECTED_UNITS = {6: "m", 4: "g cm3"}
BANDS = (("mean", "", 1), ("cov", "_cov", 2), ("aoa", "_aoa", 3))


def _is_valid_raw(block: np.ndarray) -> np.ndarray:
    return block != NODATA


def _scaled(src: Any, band: int, raw: float) -> float:
    return float(src.scales[band - 1] * raw + src.offsets[band - 1])


def _other_band(src: Any, band: int, row: int, col: int, status: str) -> tuple[float, str]:
    raw = src.read(band, window=((row, row + 1), (col, col + 1)))[0, 0]
    if raw == NODATA:
        return math.nan, NO_DATA
    return _scaled(src, band, float(raw)), status


def read_trait(path: Path, sites: pd.DataFrame) -> pd.DataFrame:
    """One row per site: mean, cov, aoa, and a status for each. The search runs on the mean band only."""
    path = Path(path)
    # The crs lives only in a tag: rasterio reports LOCAL_CS, so an unprojected read would land somewhere plausible.
    transformer = Transformer.from_crs(4326, 6933, always_xy=True)
    rows = []
    with rasterio.open(path) as src:
        tag = src.tags().get("crs", "")
        if "6933" not in tag:
            raise ExtractionFailure(f"lusk {path.name}: the crs tag is {tag!r}, not EPSG:6933")
        reader = tiff_reader(src, 1)
        for site in sites.itertuples(index=False):
            x, y = transformer.transform(site.lon, site.lat)
            row, col = rowcol(src.transform, x, y)
            hit = pixel_with_reach(reader, src.shape, int(row), int(col), REACH, _is_valid_raw)
            if hit.row is None:
                rows.append((math.nan, math.nan, math.nan, NO_DATA, NO_DATA, NO_DATA))
                continue
            cov, cov_status = _other_band(src, 2, hit.row, hit.col, hit.status)
            aoa, aoa_status = _other_band(src, 3, hit.row, hit.col, hit.status)
            rows.append((_scaled(src, 1, hit.value), cov, aoa, hit.status, cov_status, aoa_status))
    return pd.DataFrame(
        rows,
        index=pd.Index(sites["site_name"], name="site_name"),
        columns=["mean", "cov", "aoa", "mean_status", "cov_status", "aoa_status"],
    )


def _trait_metadata(path: Path, trait_id: int) -> tuple[dict[str, Any], dict[str, str]]:
    with rasterio.open(path) as src:
        tags = src.tags()
    expected = EXPECTED_UNITS.get(trait_id)
    if expected is not None and tags.get("trait_unit") != expected:
        raise ExtractionFailure(f"lusk {path.name}: the unit tag is {tags.get('trait_unit')!r}, expected {expected!r}")
    performance = json.loads(html.unescape(tags["model_performance"]))
    metadata = {
        "column": NAMED_COLUMNS.get(trait_id, f"lusk_{trait_id}"),
        "name": tags["trait_long_name"],
        "short_name": tags["trait_short_name"],
        "unit": tags["trait_unit"],
        "pearson_r": performance["Pearson's r"],
        "experimental": trait_id in EXPERIMENTAL,
    }
    return metadata, tags


def extract(context: RunContext) -> ProductResult:
    directory = context.raw_root / "lusk_2026"
    index = pd.Index(context.sites["site_name"], name="site_name")
    values, status, traits, checksums = {}, {}, {}, {}
    tags: dict[str, str] = {}
    for trait_id in TRAIT_IDS:
        path = directory / FILE_NAME.format(trait_id=trait_id)
        metadata, tags = _trait_metadata(path, trait_id)
        table = read_trait(path, context.sites)
        for band, suffix, _number in BANDS:
            values[f"{metadata['column']}{suffix}"] = table[band]
            status[f"{metadata['column']}{suffix}"] = table[f"{band}_status"]
        traits[str(trait_id)] = metadata
        checksums[path.name] = sha256_of(path)
    return ProductResult(
        product="lusk",
        values=pd.DataFrame(values, index=index),
        status=pd.DataFrame(status, index=index),
        coverage=pd.DataFrame(index=index),
        by_year=None,
        extras={},
        provenance={
            "source": f"https://zenodo.org/records/{RECORD_ID}",
            "version": tags.get("version"),
            "checksum": checksums,
            "time_window": None,
            "licence": tags.get("license"),
            "rights": tags.get("rights"),
            "traits": traits,
            "method": (
                "Raw int16 tested against -32768, then scale x raw + offset per band. Native pixel in EPSG:6933, else the "
                "nearest valid pixel within 3 on the mean band; the other two bands are read at the pixel it chose. "
                "The maps cover shrubs, trees and grasses, not trees only."
            ),
        },
    )


def fetch_inputs(context: RunContext) -> None:
    directory = context.raw_root / "lusk_2026"
    wanted = {FILE_NAME.format(trait_id=trait_id) for trait_id in TRAIT_IDS}
    items = [item for item in zenodo_downloads(RECORD_ID, directory) if item.name in wanted]
    absent = sorted(wanted - {item.name for item in items})
    if absent:
        raise ExtractionFailure(f"lusk: record {RECORD_ID} does not hold {absent}")
    for item in items:
        fetch(item, directory)

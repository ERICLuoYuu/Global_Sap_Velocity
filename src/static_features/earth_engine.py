"""Earth Engine: retries, reply checks, reducers, sampling, and the one land mask.

A request that fails or comes back incomplete is retried, then stops the run naming its sites, so a
NaN from this module can only mean Earth Engine counted no pixels. ee is imported only inside functions.
"""

from __future__ import annotations

import functools
import logging
import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from src.static_features.grid import cell_bounds
from src.static_features.results import NO_DATA, OK, ExtractionFailure

LOGGER = logging.getLogger(__name__)

BACKOFF_S = 15.0
WORLDCOVER_ASSET = "ESA/WorldCover/v200"
WORLDCOVER_WATER = 80
REDUCER_SUFFIXES = ("_mean", "_count", "_p95")


class ShortReply(Exception):  # noqa: N818
    """Earth Engine answered but left something out. Retried like a transient error."""


def _default_transient() -> tuple[type[BaseException], ...]:
    # earthengine-api translates only HTTP errors to EEException, so a dropped connection arrives raw.
    import http.client

    import ee
    import httplib2

    return (ee.EEException, ShortReply, OSError, http.client.HTTPException, httplib2.HttpLib2Error)


def with_retries(
    call: Callable[[], Any],
    label: str,
    attempts: int = 6,
    backoff_s: float = 15.0,
    transient: tuple[type[BaseException], ...] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Run a request, waiting longer after each transient failure; a programming error goes straight through."""
    caught = _default_transient() if transient is None else transient
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except caught as error:
            last_error = error
            LOGGER.warning("%s: attempt %d of %d failed: %s", label, attempt, attempts, str(error)[:300])
            if attempt < attempts:
                sleep(backoff_s * attempt * secrets.SystemRandom().uniform(0.75, 1.25))
    raise ExtractionFailure(f"{label}: failed after {attempts} attempts: {last_error}") from last_error


def feature_rows(reply: dict) -> list[dict]:
    features = reply.get("features")
    if features is None:
        raise ShortReply("reply holds no features")
    return [feature["properties"] for feature in features]


def _check_sites(returned: set[str], asked: Sequence[str]) -> None:
    missing = sorted(set(asked) - returned)
    unasked = sorted(returned - set(asked))
    if missing or unasked:
        raise ShortReply(f"reply is missing sites {missing} and holds unasked sites {unasked}")


def _count_of(value: str) -> str:
    return value.rsplit("_", 1)[0] + "_count"


def _cell_record(name: str, properties: dict, counts: Sequence[str], values: Sequence[str]) -> dict[str, Any]:
    record: dict[str, Any] = {"site_name": name}
    for count in counts:
        if properties.get(count) is None:
            raise ShortReply(f"{name}: reply holds no {count}")
        record[count] = float(properties[count])
    for value in values:
        answer = properties.get(value)
        if answer is None and record[_count_of(value)] > 0:
            raise ShortReply(
                f"{name}: reply holds no {value} although {_count_of(value)} is {record[_count_of(value)]}"
            )
        record[value] = np.nan if answer is None else float(answer)
    return record


def cells_from_reply(
    reply: dict, site_names: Sequence[str], counts: Sequence[str], values: Sequence[str]
) -> pd.DataFrame:
    """One row per site, in the order asked. A value may be empty only where its band counted no pixels."""
    unpaired = [_count_of(value) for value in values if _count_of(value) not in counts]
    if unpaired:
        raise ValueError(f"values cannot be checked without their counts: {unpaired}")
    by_site: dict[str, dict] = {}
    for properties in feature_rows(reply):
        name = str(properties.get("site_name"))
        if name in by_site:
            raise ShortReply(f"reply repeats site {name}")
        by_site[name] = properties
    _check_sites(set(by_site), site_names)
    records = [_cell_record(name, by_site[name], counts, values) for name in site_names]
    return pd.DataFrame(records, columns=["site_name", *counts, *values]).set_index("site_name")


def pixels_from_reply(reply: dict, site_names: Sequence[str], columns: Sequence[str]) -> pd.DataFrame:
    """One row per pixel. No site may come back without pixels, and every band must hold a value."""
    rows = feature_rows(reply)
    _check_sites({str(properties.get("site_name")) for properties in rows}, site_names)
    for properties in rows:
        empty = [column for column in columns if properties.get(column) is None]
        if empty:
            raise ShortReply(f"{properties['site_name']}: a sampled pixel holds no {empty}")
    return pd.DataFrame(rows, columns=["site_name", *columns])


def check_bands(requested: Sequence[str], available: Sequence[str], label: str) -> None:
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ExtractionFailure(f"{label}: the image has no bands {unknown}; it has {sorted(available)}")
    if len(available) < 2:
        raise ExtractionFailure(f"{label}: a one-band image names its outputs after the reducer, not the band")


def initialise(project: str) -> None:
    import ee

    with_retries(
        functools.partial(ee.Initialize, project=project), f"Earth Engine initialise {project}", backoff_s=BACKOFF_S
    )


def cell_collection(sites: pd.DataFrame) -> Any:
    """One feature per site, the rectangle of its 30 arc-second cell. Sites sharing a cell keep their own feature."""
    import ee

    features = []
    for site in sites.itertuples(index=False):
        west, south, east, north = cell_bounds(site.row, site.col)
        geometry = ee.Geometry.Rectangle([west, south, east, north], "EPSG:4326", False)
        features.append(ee.Feature(geometry, {"site_name": site.site_name}))
    return ee.FeatureCollection(features)


def mean_and_count() -> Any:
    """Unweighted, so each pixel centred in the cell counts once and a pixel merely touching it not at all."""
    import ee

    return ee.Reducer.mean().unweighted().combine(ee.Reducer.count().unweighted(), sharedInputs=True)


def with_all_band(image: Any) -> Any:
    """Add a band that is never masked, so names stay <band>_mean and coverage is <band>_count / all_count."""
    import ee

    return image.addBands(ee.Image.constant(1).rename("all"))


def land_mask() -> Any:
    """Band land: 1 land, 0 water, where a masked WorldCover pixel is water, read on WorldCover's own grid."""
    import ee

    worldcover = ee.ImageCollection(WORLDCOVER_ASSET).first().select("Map")
    return worldcover.neq(WORLDCOVER_WATER).unmask(0).rename("land").reproject(worldcover.projection())


def _bands_of(names: Sequence[str]) -> list[str]:
    bands = []
    for name in names:
        suffix = next((suffix for suffix in REDUCER_SUFFIXES if name.endswith(suffix)), "")
        bands.append(name[: -len(suffix)] if suffix else name)
    return bands


def _batches(sites: pd.DataFrame, batch_size: int) -> list[pd.DataFrame]:
    return [sites.iloc[start : start + batch_size] for start in range(0, len(sites), batch_size)]


def _request_cells(
    image: Any, reducer: Any, projection: Any, part: pd.DataFrame, counts: Sequence[str], values: Sequence[str]
) -> pd.DataFrame:
    # The request and the check of its reply are one attempt, so a short reply is retried like a dropped one.
    reply = image.reduceRegions(collection=cell_collection(part), reducer=reducer, crs=projection).getInfo()
    return cells_from_reply(reply, list(part["site_name"]), counts, values)


def reduce_cells(
    image: Any,
    reducer: Any,
    projection: Any,
    sites: pd.DataFrame,
    counts: Sequence[str],
    values: Sequence[str],
    label: str,
    batch_size: int = 60,
) -> pd.DataFrame:
    """Reduce each site's cell on the given grid, one row per site. Band names are checked before any request."""
    available = with_retries(image.bandNames().getInfo, f"{label} band names", backoff_s=BACKOFF_S)
    check_bands(_bands_of([*counts, *values]), available, label)
    frames = []
    for part in _batches(sites, batch_size):
        request = functools.partial(_request_cells, image, reducer, projection, part, counts, values)
        frames.append(with_retries(request, f"{label} sites {list(part['site_name'])}", backoff_s=BACKOFF_S))
    return pd.concat(frames)


def _request_pixels(image: Any, projection: Any, part: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    sampled = image.sampleRegions(
        collection=cell_collection(part), properties=["site_name"], projection=projection, geometries=False
    )
    return pixels_from_reply(sampled.getInfo(), list(part["site_name"]), columns)


def sample_cells(
    image: Any, projection: Any, sites: pd.DataFrame, columns: Sequence[str], label: str, batch_size: int = 60
) -> pd.DataFrame:
    """Every pixel centred in each site's cell, one row per pixel. Unmask the image first, or masked pixels vanish."""
    available = with_retries(image.bandNames().getInfo, f"{label} band names", backoff_s=BACKOFF_S)
    check_bands(columns, available, label)
    frames = []
    for part in _batches(sites, batch_size):
        request = functools.partial(_request_pixels, image, projection, part, columns)
        frames.append(with_retries(request, f"{label} sites {list(part['site_name'])}", backoff_s=BACKOFF_S))
    return pd.concat(frames, ignore_index=True)


def cell_means(
    cells: pd.DataFrame, columns: Mapping[str, tuple[str, float]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """values, status and coverage from reduce_cells output. columns maps each output column to (band, factor)."""
    values, status, coverage = {}, {}, {}
    for column, (band, factor) in columns.items():
        counted = cells[f"{band}_count"]
        values[column] = cells[f"{band}_mean"] * factor
        status[column] = np.where(counted > 0, OK, NO_DATA)
        coverage[column] = counted / cells["all_count"]
    index = cells.index
    return (
        pd.DataFrame(values, index=index),
        pd.DataFrame(status, index=index),
        pd.DataFrame(coverage, index=index),
    )

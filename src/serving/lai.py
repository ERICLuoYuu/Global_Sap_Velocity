"""GlobMap LAI for served cells and days, attached the way the merge attaches it to a training site.

Training: ``src/Extractors/extract_globmap_lai.py`` samples every composite ``GlobMapLAIV3.A<YYYY><DOY>``
(dated 00:00 UTC of that day) at the pixel nearest the site, times 0.01, without masking. The merge then
joins composites to every merged HOUR by nearest time within 8 days, inclusive (``merge_asof``,
``direction="nearest"``, ``tolerance=8 d``), and the daily LAI is the plain mean over the day's merged
hours, which skips hours without a match.

A served cell reads the pixel nearest its centre, and each day takes the same blend of composites over
its 24 hours, computed with ``merge_asof`` itself so ties resolve exactly as they did in training. (A
site day missing observed hours weighted its composites by the hours it had; a cell has them all.)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.serving.grid import nearest_index

FILENAME = re.compile(r"^GlobMapLAIV3\.A(\d{4})(\d{3})\..*\.tif$")
LAI_SCALE = 0.01
MATCH_TOLERANCE = pd.Timedelta(days=8)
HOURS_PER_DAY = 24


def composites(directory) -> pd.DataFrame:
    """Every GlobMap composite under ``directory``: ``date`` (UTC midnight) and ``path``, in date order."""
    rows = []
    for path in sorted(Path(directory).glob("GlobMapLAIV3.A*.tif")):
        match = FILENAME.match(path.name)
        if match is None:
            continue
        start = pd.Timestamp(year=int(match.group(1)), month=1, day=1, tz="UTC")
        rows.append({"date": start + pd.Timedelta(days=int(match.group(2)) - 1), "path": str(path)})
    if not rows:
        raise FileNotFoundError(f"no GlobMap composites under {directory}")
    table = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if table["date"].duplicated().any():
        raise ValueError(f"{directory}: more than one composite for a date")
    return table


def day_weights(days, dates) -> np.ndarray:
    """(days, composites): the share of each day's hours matched to each composite; a row of zeros if none."""
    index = pd.DatetimeIndex(days)
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    hours = index.tz_convert(None).values[:, None] + np.arange(HOURS_PER_DAY).astype("timedelta64[h]")[None, :]
    left = pd.DataFrame({"TIMESTAMP": pd.DatetimeIndex(hours.ravel()).tz_localize("UTC")})
    right = pd.DataFrame({"TIMESTAMP": pd.DatetimeIndex(dates), "composite": np.arange(len(dates))})
    matched = pd.merge_asof(left, right, on="TIMESTAMP", direction="nearest", tolerance=MATCH_TOLERANCE)
    composite = matched["composite"].to_numpy(dtype=float)
    valid = np.isfinite(composite)
    weights = np.zeros((len(index), len(dates)))
    day_of_hour = np.repeat(np.arange(len(index)), HOURS_PER_DAY)
    np.add.at(weights, (day_of_hour[valid], composite[valid].astype(int)), 1.0)
    counts = weights.sum(axis=1, keepdims=True)
    return np.divide(weights, counts, out=np.zeros_like(weights), where=counts > 0)


def _pixel_centres(transform, height: int, width: int):
    rows = transform.f + (np.arange(height) + 0.5) * transform.e
    cols = transform.c + (np.arange(width) + 0.5) * transform.a
    return rows, cols


def lai_for_cells(days, directory, latitudes, longitudes) -> np.ndarray:
    """float32 (days, cells) LAI; NaN on a day no composite lies within the match tolerance of."""
    import rasterio  # noqa: PLC0415

    table = composites(directory)
    weights = day_weights(days, table["date"])
    lat = np.asarray(latitudes, dtype=float)
    lon = np.asarray(longitudes, dtype=float)
    out = np.zeros((weights.shape[0], len(lat)), dtype=np.float32)
    pixels, grid_key = None, None
    for k in np.flatnonzero(weights.sum(axis=0) > 0):
        with rasterio.open(table.loc[k, "path"]) as src:
            if src.nodata is not None:
                raise NotImplementedError(f"{src.name} declares nodata; the training extract read GlobMap without it")
            key = (tuple(src.transform), src.height, src.width)
            if key != grid_key:
                centre_y, centre_x = _pixel_centres(src.transform, src.height, src.width)
                pixels, grid_key = (nearest_index(centre_y, lat), nearest_index(centre_x, lon)), key
            values = src.read(1)[pixels].astype(np.float32) * np.float32(LAI_SCALE)
        out += weights[:, k].astype(np.float32)[:, None] * values[None, :]
    out[weights.sum(axis=1) == 0] = np.nan
    return out

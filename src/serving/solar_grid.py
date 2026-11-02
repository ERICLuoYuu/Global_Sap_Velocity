"""``ext_rad`` and ``day_length`` for grid cells, by the per-site functions, per latitude row and longitude band.

Both daily quantities integrate a 24-hour UTC window over the sun's (periodic) daily course, so longitude
moves them only through the declination's drift across the hours by which that course is offset, and
through the phase of the sampling instants. Evaluating at the centres of 30-degree longitude bands (no
cell more than 1 h away) keeps the error under a tenth of the largest day-to-day change (tested), at 12x
the cost of one longitude rather than 3600x. The exception is ``day_length`` at high latitude, where the
sun crosses the horizon at a shallow angle and a minute count moves more: at 67 N it is off by 5-6 minutes
for weeks before midnight sun (tested bound: 0.15 h at any latitude).

``day_length`` counts minutes of pvlib's true (unrefracted) solar elevation above 0, in which the site
altitude the merge passes enters only through topocentric parallax, far below a minute (tested); it is
computed at sea level.
"""

from __future__ import annotations

from multiprocessing import Pool
from typing import Mapping

import numpy as np

from src.serving.era5_daily import day_length_daily, ext_rad_daily

BAND_WIDTH_DEG = 30.0
N_BANDS = int(round(360.0 / BAND_WIDTH_DEG))


def band_of(longitudes) -> np.ndarray:
    bands = np.floor((np.asarray(longitudes, dtype=float) + 180.0) / BAND_WIDTH_DEG).astype(np.int64)
    return np.clip(bands, 0, N_BANDS - 1)


def band_centres() -> np.ndarray:
    return -180.0 + (np.arange(N_BANDS) + 0.5) * BAND_WIDTH_DEG


def _one(task):
    days, latitude, longitude = task
    return (
        ext_rad_daily(days, latitude, longitude).to_numpy(dtype=np.float32),
        day_length_daily(days, latitude, longitude).to_numpy(dtype=np.float32),
    )


def solar_year(days, latitudes, processes: int = 1) -> dict:
    """``ext_rad`` and ``day_length``, float32 (days, latitudes, bands), with the axes they were computed on."""
    lats = np.asarray(latitudes, dtype=float)
    centres = band_centres()
    tasks = [(days, lat, lon) for lat in lats for lon in centres]
    if processes > 1:
        with Pool(processes) as pool:
            results = pool.map(_one, tasks, chunksize=4)
    else:
        results = [_one(task) for task in tasks]
    shape = (len(days), len(lats), len(centres))
    return {
        "ext_rad": np.stack([r[0] for r in results], axis=1).reshape(shape),
        "day_length": np.stack([r[1] for r in results], axis=1).reshape(shape),
        "latitudes": lats,
        "band_centres": centres,
    }


def cell_solar(solar: Mapping, latitude_index, longitudes) -> tuple:
    """``ext_rad`` and ``day_length``, float64 (days, cells), for cells at ``solar["latitudes"][latitude_index]``."""
    index = np.asarray(latitude_index, dtype=np.int64)
    band = band_of(longitudes)
    return solar["ext_rad"][:, index, band].astype(float), solar["day_length"][:, index, band].astype(float)

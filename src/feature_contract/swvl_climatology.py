"""Per-cell ERA5-Land soil-moisture climatology: the servable normalisation.

The cube ``swvl_sigma_climatology_v2_1995_2019.nc`` holds sigma and mean of ERA5-Land
DAILY-mean volumetric soil water per 0.1 deg cell over 1995-01-01..2019-01-01, layers 1-4
(coords ``layer``, ``lat`` descending, ``lon`` -180..180). Training sites and prediction
cells are both normalised against it, so both sides share one definition::

    bare   = raw / sigma_clim
    zscore = (raw - mu_clim) / sigma_clim

No clip. On the 212 training sites (probe job 46722265) raw/sigma_clim exceeds 10 on 9.8% of
layer-1 site-days and on 76% of layer-4 site-days (median 14.8), so the +-10 clip used by the
rooted-v2 prediction path would flatten most of layer 4.

A cell is valid only if ``sigma >= MIN_VALID_SIGMA``: about 1% of cube cells sit at the
1e-10 floor (ice, open water), where mean/sigma reaches ~4e9.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

SIGMA_VAR = "swvl_sigma"
MEAN_VAR = "swvl_mean"
MIN_VALID_SIGMA = 1e-6
DEFAULT_MAX_RADIUS_CELLS = 3
EARTH_RADIUS_KM = 6371.0
FULL_CIRCLE_DEG = 359.9


@dataclass(frozen=True)
class SwvlClimatology:
    lats: np.ndarray
    lons: np.ndarray
    sigma: dict[int, np.ndarray]
    mean: dict[int, np.ndarray]
    path: str
    sha256: str
    attrs: dict[str, str]

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(sorted(self.sigma))


@dataclass(frozen=True)
class CellStats:
    layer: int
    sigma: float
    mean: float
    cell_lat: float
    cell_lon: float
    distance_km: float
    fallback: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_climatology(path) -> SwvlClimatology:
    """Read the stacked ``(layer, lat, lon)`` cube into memory."""
    path = Path(path)
    with xr.open_dataset(path) as ds:
        for var in (SIGMA_VAR, MEAN_VAR):
            if var not in ds.data_vars:
                raise KeyError(f"{path} has no {var!r} variable (found {list(ds.data_vars)})")
        lons = ds["lon"].values.astype(float)
        if lons.max() > 180.0:
            raise ValueError(f"{path}: longitude axis runs to {lons.max():.1f}; expected -180..180")
        lats = ds["lat"].values.astype(float)
        layers = [int(v) for v in ds["layer"].values]
        sigma = {layer: ds[SIGMA_VAR].sel(layer=layer).values.astype(float) for layer in layers}
        mean = {layer: ds[MEAN_VAR].sel(layer=layer).values.astype(float) for layer in layers}
        attrs = {str(k): str(v) for k, v in ds.attrs.items()}
    return SwvlClimatology(
        lats=lats, lons=lons, sigma=sigma, mean=mean, path=str(path), sha256=_sha256(path), attrs=attrs
    )


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return float(2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


def _step(axis: np.ndarray) -> float:
    return float(abs(axis[1] - axis[0])) if len(axis) > 1 else 0.0


def lookup_cell(
    clim: SwvlClimatology, lat: float, lon: float, layer: int, max_radius_cells: int = DEFAULT_MAX_RADIUS_CELLS
) -> CellStats:
    """The valid cell nearest to (lat, lon) for one layer.

    Searches the whole ``(2r+1)^2`` window and takes the minimum great-circle distance.
    Stopping at the first ring that holds a valid cell is wrong away from the equator:
    at 70 deg N two longitude cells (7.6 km) are nearer than one latitude cell (11.1 km).
    """
    if layer not in clim.sigma:
        raise KeyError(f"layer {layer} is not in the climatology (layers {clim.layers})")
    lats, lons = clim.lats, clim.lons
    lat_step, lon_step = _step(lats), _step(lons)
    inside_lat = lats.min() - lat_step <= lat <= lats.max() + lat_step
    inside_lon = lons.min() - lon_step <= lon <= lons.max() + lon_step
    if not (inside_lat and inside_lon):
        raise LookupError(f"site ({lat:.4f}, {lon:.4f}) is outside the climatology grid")

    sigma = clim.sigma[layer]
    n_lat, n_lon = len(lats), len(lons)
    wraps = (lons.max() - lons.min() + lon_step) >= FULL_CIRCLE_DEG
    i0 = int(np.abs(lats - lat).argmin())
    j0 = int(np.abs(lons - lon).argmin())

    best = None
    for i in range(max(i0 - max_radius_cells, 0), min(i0 + max_radius_cells, n_lat - 1) + 1):
        for dj in range(-max_radius_cells, max_radius_cells + 1):
            j = j0 + dj
            if wraps:
                j %= n_lon
            elif not 0 <= j < n_lon:
                continue
            s = sigma[i, j]
            if not (np.isfinite(s) and s >= MIN_VALID_SIGMA):
                continue
            distance = _haversine_km(lat, lon, lats[i], lons[j])
            if best is None or distance < best[0]:
                best = (distance, i, j)

    if best is None:
        raise LookupError(
            f"no cell with sigma >= {MIN_VALID_SIGMA} within {max_radius_cells} cells of "
            f"({lat:.4f}, {lon:.4f}) for layer {layer}"
        )
    distance, i, j = best
    return CellStats(
        layer=layer,
        sigma=float(sigma[i, j]),
        mean=float(clim.mean[layer][i, j]),
        cell_lat=float(lats[i]),
        cell_lon=float(lons[j]),
        distance_km=distance,
        fallback=(i, j) != (i0, j0),
    )


def normalise_swvl(raw, sigma: float, mean: float):
    """Return ``(raw / sigma, (raw - mean) / sigma)`` for an array or Series; the index is kept."""
    if not (np.isfinite(sigma) and sigma >= MIN_VALID_SIGMA):
        raise ValueError(f"sigma must be finite and >= {MIN_VALID_SIGMA}, got {sigma!r}")
    return raw / sigma, (raw - mean) / sigma

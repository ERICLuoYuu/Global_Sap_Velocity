from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

"""Put every product on one time axis and one spatial grid -- loudly.

The pipeline used to do both at once with
``ds.reindex_like(ref, method="nearest", tolerance=0.06)``. On lat/lon that
tolerance is degrees, but on a datetime64 axis pandas reads it as
``pd.to_timedelta(0.06)`` == ``Timedelta(0)``, so any period mismatch -- or an
hour of stamp drift -- returned an all-NaN product with no error at all.
Time equality is therefore asserted here, and only lat/lon are interpolated.
"""

logger = logging.getLogger(__name__)

ALIGNMENT_METHODS = ("nearest", "linear")


def _already_on(ds: xr.Dataset, dim: str, want) -> bool:
    """True when `ds` is already on exactly the reference coordinates for `dim`."""
    have = ds[dim].values
    return len(have) == len(want) and bool(np.allclose(have, want))


def _reference_key(datasets: dict[str, xr.Dataset], reference_key: str | None) -> str:
    if reference_key is not None:
        return reference_key
    return "sap_velocity" if "sap_velocity" in datasets else next(iter(datasets))


def assert_common_time_axis(
    datasets: dict[str, xr.Dataset],
    reference_key: str | None = None,
) -> None:
    """Raise unless every product carries the identical time index.

    Raises
    ------
    ValueError
        Naming each product whose axis differs, with its own extent, so the
        message says which period was actually loaded.
    """
    if len(datasets) < 2:
        return

    ref_key = _reference_key(datasets, reference_key)
    ref_time = pd.DatetimeIndex(datasets[ref_key]["time"].values)

    problems = []
    for key, ds in datasets.items():
        if key == ref_key:
            continue
        other = pd.DatetimeIndex(ds["time"].values)
        if not other.equals(ref_time):
            problems.append(f"{key}: {len(other)} steps {other[0]} .. {other[-1]}")

    if problems:
        raise ValueError(
            f"products do not share one time axis; reference '{ref_key}' has "
            f"{len(ref_time)} steps {ref_time[0]} .. {ref_time[-1]}, but "
            + "; ".join(problems)
            + ". Align the periods (or re-preprocess) rather than letting the "
            "comparison reindex them to NaN."
        )


def align_to_reference(
    datasets: dict[str, xr.Dataset],
    reference_key: str | None = None,
    method: str = "nearest",
    tolerance: float = 0.06,
) -> dict[str, xr.Dataset]:
    """Put every product's lat/lon on the reference grid. Time is left alone.

    Parameters
    ----------
    method : {'nearest', 'linear'}
        'nearest' reproduces the published run: a product on x.05 centres is
        snapped to the x.0 prediction grid, a consistent half-cell (~5.5 km)
        displacement. 'linear' averages the straddling cells instead, which is
        the area-weighted value for a half-cell offset, at the cost of
        propagating NaN one cell inward from every data edge.
    """
    if method not in ALIGNMENT_METHODS:
        raise ValueError(f"method must be one of {ALIGNMENT_METHODS}, got {method!r}")
    if len(datasets) < 2:
        return dict(datasets)

    ref_key = _reference_key(datasets, reference_key)
    ref = datasets[ref_key]
    lat = ref["lat"].values
    lon = ref["lon"].values

    aligned = {ref_key: ref}
    for key, ds in datasets.items():
        if key == ref_key:
            continue
        if method == "nearest":
            aligned[key] = ds.reindex(lat=lat, lon=lon, method="nearest", tolerance=tolerance)
        else:
            # Only interpolate an axis that actually differs. scipy evaluates
            # the bracketing interval even at an exact node, so interpolating a
            # matching axis spreads every NaN one cell and costs a row at each
            # coastline -- for a product already on the grid, that is pure loss.
            targets = {
                dim: want for dim, want in (("lat", lat), ("lon", lon)) if not _already_on(ds, dim, want)
            }
            aligned[key] = ds.interp(method="linear", **targets) if targets else ds
        logger.info(f"  Aligned {key} to {ref_key} grid by {method}: {dict(aligned[key].sizes)}")

    return aligned

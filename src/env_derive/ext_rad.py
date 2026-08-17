"""Derive ``ext_rad`` -- extraterrestrial radiation on a horizontal surface.

SAPFLUXNET ships ``ext_rad`` as a column of ``*_env_data.csv``; this repository
computes it nowhere and consumes it as a training feature in twenty-odd places.
That is fine for the 202 external sites, which carry the column, and fatal for
the 24 internal ones, which do not: both trainers treat a missing feature column
as a reason to skip the **whole site**
(``test_hyperparameter_tuning_ML_spatial_stratified.py:299-302``,
``forward_selection/data_loader.py:185-192``), so the internal corpus would be
dropped from training with one ``logging.warning`` per site and nothing raised.

**Which quantity this is was measured, not assumed.** Two places in the tree
encoded contradictory definitions -- ``gap_filling/tests/test_e2e_real_site.py``
treats ``ext_rad == 0`` as night, while ``scripts/audit_data_quality.py`` exempts
``ext_rad`` from its "night mean should be low" check, which only makes sense for
a day-of-year-only quantity that never reaches zero. Scoring candidates against
the 202 sites that ship the column (Palma job 46236407, six sites spanning
latitude -51.6 to +69.5) settled it:

===================================================  =============  ==============
candidate                                            rmse (W m-2)   r
===================================================  =============  ==============
``E0(doy)`` only, REddyProc ``fCalcExtRadiation``    995 - 1144     ~0 / negative
``clip(E0(doy) * cos(zenith), 0)``                   **6.8 - 26.6** **0.998-1.000**
same with REddyProc's ``E0`` series                  6.8 - 26.6     indistinguishable
===================================================  =============  ==============

So ``ext_rad`` is top-of-atmosphere irradiance on a *horizontal* surface, zero at
night, and the solar geometry -- not the choice of solar-constant series -- is
what carries the accuracy. The residual few W m-2 is instantaneous-versus-interval
sampling and a different solar-position algorithm upstream; it is under 2% of the
~1300 W m-2 range and far below the between-site spread the model sees.

**Never overwrite a shipped value.** :func:`fill_ext_rad` fills only where the
column is absent or NaN, so the 202 external sites stay byte-identical and the
synthesised values cannot silently displace the archive's own. Writing a
synthesised column over a measured one would put a systematic offset exactly
along the external/internal split -- a site-dependent bias the model could learn,
which is strictly worse than the column being missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["extraterrestrial_radiation", "fill_ext_rad"]

# No `TimeLike = pd.DatetimeIndex | pd.Series` alias here, deliberately. That is
# a RUNTIME expression, not an annotation, so `from __future__ import
# annotations` does not defer it and Palma's Python 3.9.25 raises TypeError on
# `type | type` at import. A formatter hook introduced exactly that once. Spelled
# inline in the signatures below the unions stay lazy strings and 3.9 is happy.


def _as_utc_index(times: pd.DatetimeIndex | pd.Series) -> pd.DatetimeIndex:
    """Return *times* as a UTC-aware DatetimeIndex.

    Both corpus dialects land here: external files carry ISO-8601 with an
    explicit ``Z`` and parse tz-aware, internal files carry a naive
    ``YYYY-MM-DD HH:MM:SS``. The naive form is UTC -- verified against the
    independent ``solar_TIMESTAMP`` column at DEU_HAI, CZE_LNZ, FIN_HYY and
    ESP_LM1, where the observed offset matches ``longitude/15 + equation of
    time`` to under a minute. Localising a naive stamp to anything else would
    rotate the diurnal cycle and stay entirely plausible on inspection.
    """
    index = pd.DatetimeIndex(pd.to_datetime(times, errors="coerce", utc=False))
    if index.tz is None:
        return index.tz_localize("UTC")
    return index.tz_convert("UTC")


def extraterrestrial_radiation(
    times: pd.DatetimeIndex | pd.Series,
    latitude: float,
    longitude: float,
    elevation: float = 0.0,
) -> pd.Series:
    """Extraterrestrial radiation on a horizontal surface, W m-2.

    Parameters
    ----------
    times
        Timestamps. Naive input is treated as UTC; tz-aware input is converted.
        ``NaT`` is permitted and yields ``NaN`` at that position.
    latitude, longitude
        Site coordinates in decimal degrees, north/east positive.
    elevation
        Site elevation in metres. Affects only the refraction correction, so a
        wrong value costs a few W m-2 near sunrise and nothing at midday; a
        non-finite value is treated as sea level rather than propagating NaN
        across the whole series.

    Returns
    -------
    pandas.Series
        Indexed exactly as *times* was, so the result can be assigned straight
        back onto the source frame. Zero at night, never negative.
    """
    import pvlib  # noqa: PLC0415 -- keeps module import cheap for callers that only need fill_ext_rad's guards

    if not np.isfinite(latitude) or not np.isfinite(longitude):
        raise ValueError(f"latitude/longitude must be finite, got {latitude!r}/{longitude!r}")
    altitude = float(elevation) if np.isfinite(elevation) else 0.0

    original_index = times.index if isinstance(times, pd.Series) else pd.RangeIndex(len(times))
    utc = _as_utc_index(times)

    values = np.full(len(utc), np.nan, dtype=float)
    usable = ~utc.isna()
    if usable.any():
        # pvlib raises on NaT, so only finite stamps are handed to it and the
        # rest keep their NaN. Dropping the NaT rows instead would silently
        # shorten the series and misalign every value after the first gap.
        valid_times = utc[usable]
        solar_position = pvlib.solarposition.get_solarposition(valid_times, latitude, longitude, altitude=altitude)
        cos_zenith = np.cos(np.radians(solar_position["apparent_zenith"].to_numpy(dtype=float)))
        solar_constant = pvlib.irradiance.get_extra_radiation(valid_times).to_numpy(dtype=float)
        values[np.asarray(usable)] = np.clip(solar_constant * cos_zenith, 0.0, None)

    return pd.Series(values, index=original_index, name="ext_rad")


def fill_ext_rad(
    existing: pd.Series | None,
    times: pd.DatetimeIndex | pd.Series,
    latitude: float,
    longitude: float,
    elevation: float = 0.0,
) -> pd.Series:
    """Fill ``ext_rad`` where it is absent, preserving every shipped value.

    Mirrors the ``fill_*`` contract used elsewhere for derived environmental
    variables: a real value is never replaced by a computed one. Passing
    ``None`` for *existing* -- the internal-site case, where the column does not
    exist at all -- computes the whole series.
    """
    computed = extraterrestrial_radiation(times, latitude, longitude, elevation)
    if existing is None:
        return computed
    measured = pd.to_numeric(existing, errors="coerce")
    measured = measured.reset_index(drop=True).set_axis(computed.index)
    return measured.where(measured.notna(), computed).rename("ext_rad")

"""Tests for the derived ``ext_rad`` column.

The load-bearing test here is :class:`TestPositiveControlAgainstShippedData`,
which checks the function against the 202 external sites that already carry
``ext_rad``. Everything else guards a specific way the derivation could be
wrong while still looking plausible; the positive control is the only test that
can tell us we are computing *the same quantity SAPFLUXNET computed*, which is
the whole reason it is safe to synthesise the column for the internal sites.

A unit test that only checked "non-negative, peaks at midday" would pass just as
happily on a formula that is off by a factor, an hour, or a hemisphere.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.env_derive.ext_rad import extraterrestrial_radiation, fill_ext_rad

CORPUS = Path(
    os.environ.get(
        "SAPWOOD_CORPUS",
        "/scratch/tmp/yluo2/Global_Sap_Velocity/data/raw/0.1.5/0.1.5/csv/sapwood_combined",
    )
)

# Hainich: mid-latitude, long record, unremarkable -- a site whose agreement is
# not doing anything clever.
REFERENCE_SITE = "DEU_MER_MIX_THI"


def _day(lat: float = 50.0, lon: float = 10.0, freq: str = "30min") -> pd.DatetimeIndex:
    return pd.date_range("2020-06-21 00:00", "2020-06-21 23:30", freq=freq, tz="UTC")


class TestPhysicalShape:
    def test_never_negative(self):
        values = extraterrestrial_radiation(_day(), 50.0, 10.0)
        assert (values.dropna() >= 0).all()

    def test_zero_at_night(self):
        """Night is genuinely zero, not merely small.

        This is the property `gap_filling` relies on when it derives a day mask
        as `ext_rad > 0`, and the property that distinguishes the horizontal-
        surface quantity from the day-of-year-only one.
        """
        values = extraterrestrial_radiation(_day(), 50.0, 10.0)
        midnight = values.iloc[0:4]
        assert (midnight == 0).all()

    def test_peaks_near_solar_noon(self):
        # 10 deg E -> solar noon near 11:20 UTC. Half-hourly index, so the peak
        # sits at position ~23.
        values = extraterrestrial_radiation(_day(), 50.0, 10.0)
        peak_hour = _day()[int(values.idxmax())].hour
        assert 10 <= peak_hour <= 12

    def test_varies_with_latitude(self):
        """Guards against silently implementing the day-of-year-only quantity.

        `E0(doy)` alone is identical at every site on Earth at a given instant
        and scored rmse ~1000 W/m2 against the real column. If a refactor ever
        drops the `cos(zenith)` factor, this is the test that notices.
        """
        times = _day()
        tropical = extraterrestrial_radiation(times, 0.0, 10.0)
        polar = extraterrestrial_radiation(times, 78.0, 10.0)
        assert not np.allclose(tropical.to_numpy(), polar.to_numpy())
        assert tropical.max() > polar.max()

    def test_polar_night_is_all_zero(self):
        times = pd.date_range("2020-12-21 00:00", "2020-12-21 23:30", freq="30min", tz="UTC")
        values = extraterrestrial_radiation(times, 80.0, 10.0)
        assert (values == 0).all()


class TestTimestampHandling:
    def test_naive_is_treated_as_utc(self):
        """The internal corpus ships naive stamps; they are UTC.

        If naive input were localised to anything else the diurnal cycle would
        rotate and every downstream day/night split would quietly shift.
        """
        aware = pd.date_range("2020-06-21 00:00", periods=48, freq="30min", tz="UTC")
        naive = aware.tz_localize(None)
        assert np.allclose(
            extraterrestrial_radiation(aware, 50.0, 10.0).to_numpy(),
            extraterrestrial_radiation(naive, 50.0, 10.0).to_numpy(),
        )

    def test_non_utc_tz_is_converted_not_stripped(self):
        aware = pd.date_range("2020-06-21 00:00", periods=48, freq="30min", tz="UTC")
        shifted = aware.tz_convert("America/New_York")
        assert np.allclose(
            extraterrestrial_radiation(aware, 50.0, 10.0).to_numpy(),
            extraterrestrial_radiation(shifted, 50.0, 10.0).to_numpy(),
        )

    def test_nat_yields_nan_without_misaligning(self):
        """A NaT must not shorten the series.

        pvlib raises on NaT, so the obvious fix is to drop those rows -- which
        silently shifts every value after the first gap onto the wrong
        timestamp. The result must stay the same length and keep its alignment.
        """
        times = pd.Series(pd.date_range("2020-06-21 00:00", periods=48, freq="30min", tz="UTC"))
        clean = extraterrestrial_radiation(times, 50.0, 10.0)
        holed = times.copy()
        holed.iloc[5] = pd.NaT
        result = extraterrestrial_radiation(holed, 50.0, 10.0)

        assert len(result) == len(times)
        assert np.isnan(result.iloc[5])
        after = [i for i in range(len(times)) if i != 5]
        assert np.allclose(result.iloc[after].to_numpy(), clean.iloc[after].to_numpy())

    def test_empty_input(self):
        result = extraterrestrial_radiation(pd.DatetimeIndex([], tz="UTC"), 50.0, 10.0)
        assert len(result) == 0

    def test_all_nat(self):
        times = pd.Series([pd.NaT, pd.NaT])
        result = extraterrestrial_radiation(times, 50.0, 10.0)
        assert result.isna().all()

    def test_preserves_caller_index(self):
        """The result is assigned straight back onto a frame, so the index must
        match what the caller passed, not a fresh RangeIndex."""
        times = pd.Series(
            pd.date_range("2020-06-21 00:00", periods=4, freq="30min", tz="UTC"),
            index=[10, 11, 12, 13],
        )
        assert list(extraterrestrial_radiation(times, 50.0, 10.0).index) == [10, 11, 12, 13]


class TestArgumentValidation:
    @pytest.mark.parametrize("lat,lon", [(np.nan, 10.0), (50.0, np.nan), (np.inf, 10.0)])
    def test_rejects_non_finite_coordinates(self, lat, lon):
        """Coordinates are the one input a wrong value silently corrupts.

        A NaN latitude would otherwise propagate to an all-NaN column, and an
        all-NaN feature column is exactly the state that makes the trainer skip
        the site -- the failure this module exists to prevent.
        """
        with pytest.raises(ValueError):
            extraterrestrial_radiation(_day(), lat, lon)

    def test_non_finite_elevation_falls_back_to_sea_level(self):
        values = extraterrestrial_radiation(_day(), 50.0, 10.0, elevation=np.nan)
        assert values.notna().all()
        assert values.max() > 0


class TestFillSemantics:
    def test_none_computes_everything(self):
        times = _day()
        filled = fill_ext_rad(None, times, 50.0, 10.0)
        assert filled.notna().all()
        assert filled.name == "ext_rad"

    def test_existing_values_are_never_overwritten(self):
        """The 202 external sites must come out byte-identical.

        A synthesised value displacing a shipped one would put a systematic
        offset exactly along the external/internal split.
        """
        times = _day()
        sentinel = pd.Series(np.full(len(times), -999.0))
        filled = fill_ext_rad(sentinel, times, 50.0, 10.0)
        assert (filled == -999.0).all()

    def test_only_gaps_are_filled(self):
        times = _day()
        partial = pd.Series(np.full(len(times), -999.0))
        partial.iloc[20:24] = np.nan
        filled = fill_ext_rad(partial, times, 50.0, 10.0)

        assert (filled.iloc[:20] == -999.0).all()
        assert (filled.iloc[24:] == -999.0).all()
        assert filled.iloc[20:24].notna().all()
        assert (filled.iloc[20:24] != -999.0).all()

    def test_non_numeric_existing_is_coerced_then_filled(self):
        times = _day()
        existing = pd.Series(["NA"] * len(times))
        filled = fill_ext_rad(existing, times, 50.0, 10.0)
        assert filled.notna().all()


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not reachable from this host")
class TestPositiveControlAgainstShippedData:
    """Reproduce `ext_rad` where SAPFLUXNET already provides it.

    Without this the module is an untested hypothesis about someone else's
    column. With it, agreement on a site that ships the real thing is direct
    evidence that the values synthesised for the internal sites are the same
    quantity on the same clock.
    """

    @staticmethod
    def _load(site: str):
        env = pd.read_csv(CORPUS / f"{site}_env_data.csv", usecols=["TIMESTAMP", "ext_rad"])
        md = pd.read_csv(CORPUS / f"{site}_site_md.csv").iloc[0]
        lat = float(pd.to_numeric(md["si_lat"]))
        lon = float(pd.to_numeric(md["si_long"]))
        elev = float(pd.to_numeric(md.get("si_elev", 0.0)) or 0.0)
        env["TIMESTAMP"] = pd.to_datetime(env["TIMESTAMP"], errors="coerce", utc=True)
        env["ext_rad"] = pd.to_numeric(env["ext_rad"], errors="coerce")
        env = env.dropna().reset_index(drop=True)
        return env, lat, lon, elev

    def test_reproduces_the_shipped_column(self):
        env, lat, lon, elev = self._load(REFERENCE_SITE)
        if len(env) < 1000:
            pytest.skip(f"{REFERENCE_SITE} has too few rows to be decisive")
        sample = env.iloc[:: max(1, len(env) // 20000)]

        computed = extraterrestrial_radiation(sample["TIMESTAMP"], lat, lon, elev).to_numpy()
        observed = sample["ext_rad"].to_numpy()

        rmse = float(np.sqrt(np.mean((observed - computed) ** 2)))
        r = float(np.corrcoef(observed, computed)[0, 1])
        bias = float(np.mean(computed - observed))

        # Thresholds sit well outside the measured spread across six sites
        # (rmse 6.8-26.6, r 0.9984-0.9998) but far inside what any real error
        # would produce: a dropped cos(zenith) scored rmse ~1000, a one-hour
        # clock error scores in the hundreds.
        assert r > 0.99, f"r={r:.4f}"
        assert rmse < 60.0, f"rmse={rmse:.2f}"
        assert abs(bias) < 15.0, f"bias={bias:.2f}"

    def test_night_agrees_on_being_night(self):
        env, lat, lon, elev = self._load(REFERENCE_SITE)
        sample = env.iloc[:: max(1, len(env) // 20000)]
        computed = extraterrestrial_radiation(sample["TIMESTAMP"], lat, lon, elev).to_numpy()
        observed = sample["ext_rad"].to_numpy()

        # Where the archive says zero, we must be at or near zero. Disagreement
        # here is the signature of a clock offset, which a correlation on the
        # full series can otherwise hide.
        archive_night = observed == 0
        if archive_night.sum() > 100:
            assert np.percentile(computed[archive_night], 99) < 50.0

    def test_an_hour_of_clock_error_would_be_caught(self):
        """Negative control for the test above.

        Proves the thresholds can actually fail. A gate that cannot fail is not
        a gate -- and a silently-passing check is how the sampled-population
        error in this project's QC work survived for so long.
        """
        env, lat, lon, elev = self._load(REFERENCE_SITE)
        sample = env.iloc[:: max(1, len(env) // 20000)]
        wrong = extraterrestrial_radiation(sample["TIMESTAMP"] + pd.Timedelta(hours=1), lat, lon, elev).to_numpy()
        observed = sample["ext_rad"].to_numpy()

        rmse = float(np.sqrt(np.mean((observed - wrong) ** 2)))
        assert rmse > 60.0, f"a 1 h offset scored rmse={rmse:.2f}, so the real test proves nothing"

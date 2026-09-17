"""The three fixes working as one (plan R3 integration).

A clock corrected where the loaders correct it, env daily statistics taken from env's OWN hours under the
support rule, and a gate that refuses the tree when either did not happen. The fixtures are the three shapes
the corpus has: a site that ships night ``sw_in`` as NA, a site whose logger runs five hours late and is in the
registry, and a site that is simply fine and must come back untouched.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from notebooks.merge_gap_filled_hourly_orginal import env_daily_overwrite, radiation_provenance
from src.clock_offsets.apply import correct_loaded_frames
from src.clock_offsets.registry import ClockOffset
from src.env_daily.gate import main as env_daily_gate
from src.env_daily.invariants import check_support, kt_exceedance_days
from src.env_daily.support import hourly_toa
from src.feature_contract.definitions import PROVENANCE_FILE, PROVENANCE_KEY

LAT, LON = 45.0, 0.0
DAYS = pd.DatetimeIndex(["2015-06-20", "2015-06-21"], tz="UTC")
KT = 0.7
LATE_HOURS = 5.0
SENTINEL = -999.0

NIGHT_NA = "FIX_NIGHTNA_SITE"
LATE = "FIX_LATE_SITE"
CLEAN = "FIX_CLEAN_SITE"

REGISTRY = (
    ClockOffset(site=LATE, component="env", offset_hours=LATE_HOURS, status="applied", evidence="fixture"),
)


def hourly(night_na=False):
    """One row per hour over two days in TRUE time: sw_in at a constant clearness, ta a diurnal wave."""
    index = pd.date_range(DAYS[0], periods=48, freq="h", name="TIMESTAMP")
    toa, dark = hourly_toa(DAYS, LAT, LON)
    sw_in = KT * toa.ravel()
    if night_na:
        sw_in = np.where(dark.ravel(), np.nan, sw_in)
    ta = 15.0 + 10.0 * np.sin(np.arange(48) / 24.0 * 2.0 * np.pi)
    return pd.DataFrame({"TIMESTAMP": index, "ta": ta, "sw_in": sw_in})


def joined_daily():
    """The merge's daily frame before the overwrite: sentinels where env's own statistics will land."""
    frame = pd.DataFrame({"TIMESTAMP": DAYS})
    for col in ("ta", "sw_in", "ext_rad"):
        for suffix in ("", "_min", "_max", "_sum"):
            frame[col + suffix] = SENTINEL
    frame["sap_velocity"] = np.arange(1.0, len(DAYS) + 1.0)
    return frame


def block(complete=True, kt_by_site=None):
    counted = {NIGHT_NA: 0, LATE: 0, CLEAN: 0} if kt_by_site is None else kt_by_site
    return {"name": "servable", "complete": complete, **radiation_provenance(counted)}


class TestTheThreeShapes:
    def test_a_late_logger_corrected_at_load_reproduces_the_truth_exactly(self, tmp_path):
        # The whole point of correcting at the load boundary: everything downstream, the daily statistics
        # included, must be indistinguishable from a site that was never late.
        truth = hourly()
        late = truth.assign(TIMESTAMP=truth["TIMESTAMP"] + pd.Timedelta(hours=LATE_HOURS))
        corrected, _flags = correct_loaded_frames(
            tmp_path / f"{LATE}_env_data.csv", "env", late, None, REGISTRY
        )
        got = env_daily_overwrite(joined_daily(), corrected.set_index("TIMESTAMP"), LAT, LON)
        want = env_daily_overwrite(joined_daily(), truth.set_index("TIMESTAMP"), LAT, LON)
        pd.testing.assert_frame_equal(got, want)

    def test_a_night_na_site_gives_the_24_hour_mean_and_never_exceeds_ext_rad(self):
        # The defect this task exists for: a daylight mean against a 24-hour ext_rad, which reads as a
        # clearness above 1. The scaled mean recovers the day's true mean to well inside a percent.
        daily = env_daily_overwrite(joined_daily(), hourly(night_na=True).set_index("TIMESTAMP"), LAT, LON)
        toa, _dark = hourly_toa(DAYS, LAT, LON)
        np.testing.assert_allclose(daily["sw_in"].to_numpy(), KT * toa.mean(axis=1), rtol=5e-3)
        assert (daily["sw_in"] <= daily["ext_rad"]).all()
        assert kt_exceedance_days(daily) == 0
        check_support(daily)  # raises if any written value lacks its support

    def test_a_site_outside_the_registry_comes_back_untouched(self, tmp_path):
        frame = hourly(night_na=True)
        same, flags = correct_loaded_frames(
            tmp_path / f"{CLEAN}_env_data.csv", "env", frame, None, REGISTRY
        )
        pd.testing.assert_frame_equal(same, frame)
        assert flags is None

    def test_the_uncorrected_late_site_is_visibly_wrong(self):
        # The negative control for the first test: without the correction the daily rows are NOT the truth,
        # so that test cannot be passing for a trivial reason.
        truth = hourly()
        late = truth.assign(TIMESTAMP=truth["TIMESTAMP"] + pd.Timedelta(hours=LATE_HOURS))
        got = env_daily_overwrite(joined_daily(), late.set_index("TIMESTAMP"), LAT, LON)
        want = env_daily_overwrite(joined_daily(), truth.set_index("TIMESTAMP"), LAT, LON)
        assert not np.allclose(got["sw_in"].to_numpy(), want["sw_in"].to_numpy(), equal_nan=True)


class TestTheGateOverWhatTheMergeWrote:
    def test_a_tree_the_merge_finished_under_the_rule_passes(self, tmp_path, capsys):
        (tmp_path / PROVENANCE_FILE).write_text(json.dumps({PROVENANCE_KEY: block()}))
        assert env_daily_gate([str(tmp_path)]) == 0
        assert "ENV DAILY GATE PASS" in capsys.readouterr().out

    def test_a_merge_that_did_not_finish_is_refused(self, tmp_path):
        (tmp_path / PROVENANCE_FILE).write_text(json.dumps({PROVENANCE_KEY: block(complete=False)}))
        assert env_daily_gate([str(tmp_path)]) == 1

    def test_a_tree_whose_radiation_regressed_is_refused(self, tmp_path, capsys):
        regressed = {NIGHT_NA: 0, LATE: 0, CLEAN: 12}  # a site that never had a kt > 1 day before
        (tmp_path / PROVENANCE_FILE).write_text(json.dumps({PROVENANCE_KEY: block(kt_by_site=regressed)}))
        assert env_daily_gate([str(tmp_path)]) == 1
        assert CLEAN in capsys.readouterr().out

    def test_the_block_carries_the_rule_and_the_registry_the_tree_was_built_with(self):
        carried = block()
        assert carried["env_daily_rule"]["version"] >= 1
        assert len(carried["clock_registry_sha256"]) == 64
        assert carried["kt_exceedance"]["n_sites_counted"] == 3

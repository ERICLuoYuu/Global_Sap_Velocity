"""Tests for the throttling and resumability in extract_climatedata_gee.

On 2026-08-18 a full 226-site re-extraction lost 160 of 226 sites, every one of
them to ``Too Many Requests: Exceeded Earth Engine concurrency limit. Your
project is in Restricted Mode`` and none of them to a data problem.  Three
properties are pinned here, because the absence of each is what turned a
transient refusal into a dead run:

* a quota refusal is classified apart from a genuine fault, and draws on its own
  much larger budget -- sharing one 4-attempt allowance meant a "come back
  later" consumed the same credit as a permanent defect;
* the backoff is JITTERED, so workers do not wake in lockstep and re-trigger the
  very limit they are waiting out;
* a cached site is keyed by a digest of its own inputs, so changing any of them
  orphans the cache rather than silently reusing it.

Nothing here contacts Earth Engine: the fake collection's ``map`` ignores the
mapper, so the server-side closure in ``_fetch_chunk`` never runs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.Extractors import extract_climatedata_gee as mod

pytestmark = pytest.mark.unit

# The verbatim message from job 46264509.
QUOTA_MSG = (
    "Too Many Requests: Exceeded Earth Engine concurrency limit. Your project "
    "is in Restricted Mode. Learn more at https://developers.google.com/"
    "earth-engine/guides/noncommercial_tiers#restricted_mode"
)

# The defect this module was originally hardened against.  It must NOT be
# treated as retryable -- retrying a misspelt band forever just hangs the run.
BAND_FAULT_MSG = "Image.select: Pattern 'dewpoint_2m' did not match any bands."


class _FakeCollection:
    """Stands in for an ``ee.ImageCollection`` with a scripted ``getInfo``.

    ``filterDate``/``map``/``flatten`` all return ``self`` and ``map`` discards
    the mapper, which is what keeps this hermetic.
    """

    def __init__(self, outcomes=None, repeat=None):
        self._outcomes = list(outcomes or [])
        self._repeat = repeat
        self.calls = 0

    def filterDate(self, start, end):  # noqa: N802 - mirrors the ee API
        return self

    def map(self, fn):
        return self

    def flatten(self):
        return self

    def getInfo(self):  # noqa: N802 - mirrors the ee API
        self.calls += 1
        outcome = self._outcomes.pop(0) if self._outcomes else self._repeat
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _fetch(collection):
    """Call ``_fetch_chunk`` with arguments the fake never inspects."""
    return mod._fetch_chunk(
        collection,
        point_fc=None,
        chunk_start=pd.Timestamp("2020-03-01 00:00:00"),
        chunk_end=pd.Timestamp("2020-04-01 00:00:00"),
        scale=mod.ERA5_NATIVE_SCALE_M,
    )


@pytest.fixture()
def no_sleep(monkeypatch):
    """Skip the waiting, but record what would have been waited."""
    slept = []
    monkeypatch.setattr(mod, "_sleep_backoff", lambda attempt, quota: slept.append((attempt, quota)))
    return slept


# --------------------------------------------------------------------------
# classification: positive and negative control
# --------------------------------------------------------------------------


def test_the_real_restricted_mode_message_is_a_quota_error():
    assert mod._is_quota_error(RuntimeError(QUOTA_MSG)) is True


def test_a_misspelt_band_is_not_a_quota_error():
    """Negative control: the band defect must fail fast, not retry forever."""
    assert mod._is_quota_error(RuntimeError(BAND_FAULT_MSG)) is False


@pytest.mark.parametrize(
    "message",
    [
        "Too Many Requests",
        "Exceeded Earth Engine concurrency limit",
        "Your project is in Restricted Mode",
        "User quota exceeded",
        "HTTP 429",
    ],
)
def test_quota_markers_all_match(message):
    assert mod._is_quota_error(RuntimeError(message)) is True


@pytest.mark.parametrize(
    "message",
    [
        BAND_FAULT_MSG,
        "Collection.first: Collection has no elements.",
        "Geometry constructor: Invalid coordinates.",
    ],
)
def test_genuine_faults_are_not_quota(message):
    assert mod._is_quota_error(RuntimeError(message)) is False


# --------------------------------------------------------------------------
# separate budgets
# --------------------------------------------------------------------------


def test_quota_refusal_draws_on_the_quota_budget(no_sleep):
    """A persistent 429 gets MAX_QUOTA_ATTEMPTS tries, not MAX_ATTEMPTS."""
    collection = _FakeCollection(repeat=RuntimeError(QUOTA_MSG))

    with pytest.raises(mod.ExtractionError) as excinfo:
        _fetch(collection)

    assert collection.calls == mod.MAX_QUOTA_ATTEMPTS
    assert collection.calls > mod.MAX_ATTEMPTS, "quota must outlast the fault budget"
    assert f"quota={mod.MAX_QUOTA_ATTEMPTS}" in str(excinfo.value)


def test_genuine_fault_fails_fast(no_sleep):
    collection = _FakeCollection(repeat=RuntimeError(BAND_FAULT_MSG))

    with pytest.raises(mod.ExtractionError) as excinfo:
        _fetch(collection)

    assert collection.calls == mod.MAX_ATTEMPTS
    assert f"faults={mod.MAX_ATTEMPTS}" in str(excinfo.value)
    assert BAND_FAULT_MSG in str(excinfo.value)


def test_a_transient_refusal_recovers(no_sleep):
    """Two refusals then a payload: the chunk must succeed, not fail."""
    payload = {"features": [{"properties": {"timestamp": {"value": 0}}}]}
    collection = _FakeCollection([RuntimeError(QUOTA_MSG), RuntimeError(QUOTA_MSG), payload])

    assert _fetch(collection) == payload
    assert collection.calls == 3
    assert no_sleep == [(1, True), (2, True)]


def test_mixed_failures_keep_separate_tallies(no_sleep):
    """Quota tries must not be spent out of the fault allowance."""
    outcomes = [RuntimeError(QUOTA_MSG)] * 3 + [RuntimeError(BAND_FAULT_MSG)] * mod.MAX_ATTEMPTS
    collection = _FakeCollection(outcomes, repeat=RuntimeError(BAND_FAULT_MSG))

    with pytest.raises(mod.ExtractionError) as excinfo:
        _fetch(collection)

    message = str(excinfo.value)
    assert "quota=3" in message
    assert f"faults={mod.MAX_ATTEMPTS}" in message


# --------------------------------------------------------------------------
# the jitter, which is the load-bearing part
# --------------------------------------------------------------------------


@pytest.fixture()
def slept(monkeypatch):
    durations = []
    monkeypatch.setattr(mod.time, "sleep", durations.append)
    return durations


def test_backoff_is_jittered(slept):
    """Identical inputs must NOT produce identical waits.

    Regression for the thundering herd: with a fixed schedule all 32 workers
    woke together and re-triggered the limit, degrading the run from 3 s/site
    to 70 s/site instead of recovering.
    """
    for _ in range(40):
        mod._sleep_backoff(3, quota=True)

    assert len(set(slept)) > 1, "backoff is deterministic -- the herd is back"


def test_jittered_backoff_stays_in_the_upper_half(slept):
    """Jitter must decorrelate without collapsing to ~0 and wasting an attempt."""
    base = min(mod.BACKOFF_SECONDS * (2**2), mod.BACKOFF_CAP_SECONDS)
    for _ in range(40):
        mod._sleep_backoff(3, quota=True)

    assert all(base * 0.5 <= d <= base for d in slept)


def test_quota_backoff_grows_but_is_capped(slept):
    mod._sleep_backoff(1, quota=True)
    mod._sleep_backoff(6, quota=True)
    mod._sleep_backoff(30, quota=True)

    first, later, absurd = slept
    assert later > first, "quota backoff must grow"
    assert absurd <= mod.BACKOFF_CAP_SECONDS


def test_fault_backoff_is_linear_and_capped(slept):
    mod._sleep_backoff(1, quota=False)
    mod._sleep_backoff(1000, quota=False)

    assert slept[1] <= mod.BACKOFF_CAP_SECONDS


# --------------------------------------------------------------------------
# resumability
# --------------------------------------------------------------------------

_SITE = {
    "site_name": "AUT_ZOE",
    "lat": 47.8412,
    "lon": 16.2233,
    "start_date": "2020-01-01 00:00:00",
    "end_date": "2020-01-01 02:00:00",
}
_VARS = ["temperature_2m", "total_precipitation"]
_ROWS = [
    {
        "site_id": 1,
        "site_name": "AUT_ZOE",
        "latitude": 47.8412,
        "longitude": 16.2233,
        "datetime": pd.Timestamp("2020-01-01 00:00:00"),
        "temperature_2m": 271.5,
        "total_precipitation": 0.0,
    },
    {
        "site_id": 1,
        "site_name": "AUT_ZOE",
        "latitude": 47.8412,
        "longitude": 16.2233,
        "datetime": pd.Timestamp("2020-01-01 01:00:00"),
        "temperature_2m": 272.25,
        "total_precipitation": 0.001,
    },
]


def test_cache_is_disabled_unless_a_directory_is_given(monkeypatch):
    """The default must reproduce the previous all-or-nothing behaviour."""
    monkeypatch.setattr(mod, "CACHE_DIR", None)
    assert mod._cache_load(_SITE, mod.ERA5_NATIVE_SCALE_M, _VARS, 0) is None
    mod._cache_store(_SITE, mod.ERA5_NATIVE_SCALE_M, _VARS, _ROWS)  # must not raise


def test_cache_round_trips_and_restamps_site_id(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "CACHE_DIR", str(tmp_path))

    mod._cache_store(_SITE, mod.ERA5_NATIVE_SCALE_M, _VARS, _ROWS)
    loaded = mod._cache_load(_SITE, mod.ERA5_NATIVE_SCALE_M, _VARS, 6)

    assert loaded is not None
    assert len(loaded) == len(_ROWS)
    # site_id is a row position, not part of the signature, so it is re-stamped.
    assert {row["site_id"] for row in loaded} == {7}
    assert [row["temperature_2m"] for row in loaded] == [271.5, 272.25]
    assert loaded[0]["datetime"] == pd.Timestamp("2020-01-01 00:00:00")


@pytest.mark.parametrize(
    "field,value",
    [
        ("lat", 47.9),
        ("lon", 16.3),
        ("start_date", "2019-01-01 00:00:00"),
        ("end_date", "2020-06-01 00:00:00"),
    ],
)
def test_cache_misses_when_the_site_signature_changes(monkeypatch, tmp_path, field, value):
    """A changed input must orphan the cache, never reuse it."""
    monkeypatch.setattr(mod, "CACHE_DIR", str(tmp_path))
    mod._cache_store(_SITE, mod.ERA5_NATIVE_SCALE_M, _VARS, _ROWS)

    moved = dict(_SITE)
    moved[field] = value
    assert mod._cache_load(moved, mod.ERA5_NATIVE_SCALE_M, _VARS, 0) is None


def test_cache_misses_when_the_band_list_changes(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "CACHE_DIR", str(tmp_path))
    mod._cache_store(_SITE, mod.ERA5_NATIVE_SCALE_M, _VARS, _ROWS)

    assert mod._cache_load(_SITE, mod.ERA5_NATIVE_SCALE_M, _VARS + ["dewpoint_temperature_2m"], 0) is None


def test_cache_misses_when_the_sampling_scale_changes(monkeypatch, tmp_path):
    """The 10000-vs-11132 defect must not be able to hide in a cache hit."""
    monkeypatch.setattr(mod, "CACHE_DIR", str(tmp_path))
    mod._cache_store(_SITE, mod.ERA5_NATIVE_SCALE_M, _VARS, _ROWS)

    assert mod._cache_load(_SITE, 10000.0, _VARS, 0) is None


def test_cache_band_order_does_not_matter(monkeypatch, tmp_path):
    """The digest sorts the bands, so a reordered request still hits."""
    monkeypatch.setattr(mod, "CACHE_DIR", str(tmp_path))
    mod._cache_store(_SITE, mod.ERA5_NATIVE_SCALE_M, _VARS, _ROWS)

    assert mod._cache_load(_SITE, mod.ERA5_NATIVE_SCALE_M, list(reversed(_VARS)), 0) is not None


def test_a_cache_file_short_of_a_band_raises(monkeypatch, tmp_path):
    """Belt and braces: the digest makes this unreachable, so prove the guard."""
    monkeypatch.setattr(mod, "CACHE_DIR", str(tmp_path))
    path = mod._cache_path(_SITE, mod.ERA5_NATIVE_SCALE_M, _VARS)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_ROWS).drop(columns=["total_precipitation"]).to_csv(path, index=False)

    with pytest.raises(mod.ExtractionError, match="missing 1 requested band"):
        mod._cache_load(_SITE, mod.ERA5_NATIVE_SCALE_M, _VARS, 0)


def test_cache_store_leaves_no_partial_behind(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "CACHE_DIR", str(tmp_path))
    mod._cache_store(_SITE, mod.ERA5_NATIVE_SCALE_M, _VARS, _ROWS)

    assert not list(tmp_path.glob("*.partial")), "atomic promote left a .partial"
    assert len(list(tmp_path.glob("*.csv"))) == 1


# --------------------------------------------------------------------------
# closed-window tiling
#
# The chunk list decides which hours are asked of Earth Engine at all, and it
# had NO tests -- the original closed-window fix was verified ad hoc.  It was
# wrong: it ended one step past ``end_date`` rather than one step past the last
# image inside the window, so every site whose record stops mid-hour got one
# extra hour.  Measured on the 226-site corpus (job 46319656): exactly one
# surplus row at each of 139 sites, none at the 87 that end on the hour.
# --------------------------------------------------------------------------


def _covered_hours(start, end):
    """Every whole-hour image the chunk list would actually request.

    Models ``filterDate``'s half-open semantics: an image is fetched when its
    instant lies in ``[chunk_start, chunk_end)``.  Going through the chunks
    rather than asserting on their bounds means the test pins what the run
    RECEIVES, which is the thing that was wrong.
    """
    hours = []
    for chunk_start, chunk_end in mod.get_monthly_chunks(start, end):
        cursor = pd.Timestamp(chunk_start).ceil("h")
        while cursor < pd.Timestamp(chunk_end):
            hours.append(cursor)
            cursor += mod.STEP
    return hours


@pytest.mark.parametrize(
    "site,start,end,last",
    [
        # The three verbatim from the residuals probe.
        ("AUS_CAN_ST2_MIX", "2007-05-01 00:30", "2007-06-26 00:30", "2007-06-26 00:00"),
        ("ESP_CAN", "2012-01-01 00:15", "2012-12-31 23:45", "2012-12-31 23:00"),
        ("CAN_TUR_P39_POS", "2016-12-01 00:30", "2017-01-01 00:30", "2017-01-01 00:00"),
    ],
)
def test_a_window_ending_mid_hour_stops_at_the_last_image_inside_it(site, start, end, last):
    """The off-by-one, pinned per site: 00:30 must not reach 01:00."""
    hours = _covered_hours(start, end)
    assert hours[-1] == pd.Timestamp(last), site
    assert pd.Timestamp(last) + mod.STEP not in hours, site


@pytest.mark.parametrize(
    "start,end",
    [
        ("2020-01-01 00:00", "2020-01-01 23:00"),
        ("2015-03-05 06:00", "2015-09-30 23:00"),
        ("2010-01-01 00:00", "2019-12-31 23:00"),
    ],
)
def test_an_hour_aligned_window_is_unaffected(start, end):
    """The 87 sites that were already right must stay right."""
    hours = _covered_hours(start, end)
    assert hours[0] == pd.Timestamp(start)
    assert hours[-1] == pd.Timestamp(end)
    expected = int((pd.Timestamp(end) - pd.Timestamp(start)) / mod.STEP) + 1
    assert len(hours) == expected


@pytest.mark.parametrize(
    "start,end",
    [
        ("2007-05-01 00:30", "2007-06-26 00:30"),
        ("2012-01-01 00:15", "2012-12-31 23:45"),
        ("2005-07-20 17:30", "2005-08-20 17:30"),
        ("2020-02-01 00:00", "2020-03-01 00:00"),
        ("2020-01-01 00:00", "2020-01-01 00:00"),
    ],
)
def test_no_requested_hour_falls_outside_the_closed_window(start, end):
    """The property the corpus violated 139 times."""
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    for hour in _covered_hours(start, end):
        assert lo <= hour <= hi, "%s outside [%s, %s]" % (hour, lo, hi)


def test_usa_che_map_gets_the_744_hours_its_window_holds():
    """A whole real window, counted independently of the tiling.

    The live probe returned 745 rows for this site under the old bound; 744 is
    what the closed window actually contains.
    """
    hours = _covered_hours("2005-07-20 17:30", "2005-08-20 17:30")
    assert len(hours) == 744
    assert hours[0] == pd.Timestamp("2005-07-20 18:00")
    assert hours[-1] == pd.Timestamp("2005-08-20 17:00")


def test_chunks_still_tile_with_no_gap_and_no_overlap():
    chunks = mod.get_monthly_chunks("2014-01-01 00:00", "2017-12-31 23:00")
    assert chunks[0][0] == pd.Timestamp("2014-01-01 00:00")
    for (_, earlier_end), (later_start, _) in zip(chunks, chunks[1:]):
        assert earlier_end == later_start
    assert all(start < end for start, end in chunks)


def test_a_window_inside_a_single_hour_requests_nothing():
    """No whole hour lies in [00:10, 00:50], so no image does either."""
    assert _covered_hours("2020-01-01 00:10", "2020-01-01 00:50") == []


def test_a_backwards_window_raises():
    with pytest.raises(ValueError, match="ends before it starts"):
        mod.get_monthly_chunks("2020-06-01 00:00", "2020-05-01 00:00")


# --------------------------------------------------------------------------
# the cache's behaviour token
#
# The digest covers the inputs, so a changed input orphans the entry.  It does
# NOT cover this module: after the tiling fix above, all 226 cached sites had
# byte-identical inputs and stale contents.
# --------------------------------------------------------------------------


def test_a_bumped_behaviour_version_orphans_the_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "CACHE_DIR", str(tmp_path))
    mod._cache_store(_SITE, mod.ERA5_NATIVE_SCALE_M, _VARS, _ROWS)

    # Positive control: without the bump this is a hit, so the miss below is
    # attributable to the token and to nothing else.
    assert mod._cache_load(_SITE, mod.ERA5_NATIVE_SCALE_M, _VARS, 0) is not None

    monkeypatch.setattr(mod, "CACHE_BEHAVIOUR_VERSION", "999")
    assert mod._cache_load(_SITE, mod.ERA5_NATIVE_SCALE_M, _VARS, 0) is None


def test_the_behaviour_version_is_in_the_cache_key(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "CACHE_DIR", str(tmp_path))
    before = mod._cache_path(_SITE, mod.ERA5_NATIVE_SCALE_M, _VARS)
    monkeypatch.setattr(mod, "CACHE_BEHAVIOUR_VERSION", "999")
    after = mod._cache_path(_SITE, mod.ERA5_NATIVE_SCALE_M, _VARS)

    assert before.name != after.name
    # The site name still leads, so a human can still find a site's entry.
    assert before.name.startswith("AUT_ZOE.") and after.name.startswith("AUT_ZOE.")

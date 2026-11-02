from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.static_features import earth_engine
from src.static_features.earth_engine import (
    ShortReply,
    cells_from_reply,
    check_bands,
    pixels_from_reply,
    reduce_cells,
    sample_cells,
    with_retries,
)
from src.static_features.results import ExtractionFailure


def _feature(**properties):
    return {"type": "Feature", "geometry": None, "id": "0", "properties": properties}


def _reply(*features):
    return {"type": "FeatureCollection", "columns": {}, "features": list(features)}


def _flaky(failures, error):
    calls = []

    def call():
        calls.append(1)
        if len(calls) <= failures:
            raise error
        return "answer"

    return call, calls


def test_with_retries_recovers():
    import ee

    call, calls = _flaky(2, ee.EEException("Too many concurrent aggregations"))
    sleeps = []
    assert with_retries(call, "label", sleep=sleeps.append) == "answer"
    assert len(calls) == 3
    assert len(sleeps) == 2
    assert 15.0 * 0.5 <= sleeps[0] <= 15.0 * 1.5 and 30.0 * 0.5 <= sleeps[1] <= 30.0 * 1.5


def test_with_retries_stops_after_six_with_label():
    import ee

    call, calls = _flaky(99, ee.EEException("still broken"))
    sleeps = []
    with pytest.raises(ExtractionFailure, match="terrain sites A..B.*still broken"):
        with_retries(call, "terrain sites A..B", sleep=sleeps.append)
    assert len(calls) == 6
    assert len(sleeps) == 5  # no pointless wait after the last attempt


def test_with_retries_lets_programming_errors_through():
    call, calls = _flaky(99, TypeError("a bug, not a network"))
    with pytest.raises(TypeError):
        with_retries(call, "label", sleep=lambda seconds: None)
    assert len(calls) == 1


def test_with_retries_retries_a_short_reply():
    call, calls = _flaky(1, ShortReply("missing site A"))
    assert with_retries(call, "label", sleep=lambda seconds: None) == "answer"
    assert len(calls) == 2


@pytest.mark.parametrize("error", [ConnectionResetError("reset"), TimeoutError("slow"), OSError("network down")])
def test_with_retries_retries_a_dropped_connection(error):
    # earthengine-api translates only HTTP errors, so a dropped connection arrives raw.
    call, calls = _flaky(1, error)
    assert with_retries(call, "label", sleep=lambda seconds: None) == "answer"
    assert len(calls) == 2


def test_check_bands_refuses_unknown_band():
    check_bands(["b1", "b2"], ["b1", "b2", "b3"], "label")
    with pytest.raises(ExtractionFailure, match="label.*b9"):
        check_bands(["b1", "b9"], ["b1", "b2"], "label")


def test_check_bands_refuses_a_single_band_image():
    # A one-band image names its outputs "mean" and "count", not after the band.
    with pytest.raises(ExtractionFailure, match="label"):
        check_bands(["b1"], ["b1"], "label")


COUNTS = ["elevation_m_count", "all_count"]
VALUES = ["elevation_m_mean"]


def _row(site, mean=10.0, count=13, all_count=16):
    return _feature(site_name=site, elevation_m_mean=mean, elevation_m_count=count, all_count=all_count)


def test_reply_becomes_one_row_per_site_in_the_order_asked():
    frame = cells_from_reply(_reply(_row("B", 2.0), _row("A", 1.5)), ["A", "B"], COUNTS, VALUES)
    assert list(frame.index) == ["A", "B"] and frame.index.name == "site_name"
    assert frame.loc["A", "elevation_m_mean"] == 1.5
    assert frame.loc["B", "all_count"] == 16.0


# Review Focus 1
def test_reply_missing_a_site_is_short():
    with pytest.raises(ShortReply, match="B"):
        cells_from_reply(_reply(_row("A")), ["A", "B"], COUNTS, VALUES)


def test_reply_with_an_unasked_site_is_short():
    with pytest.raises(ShortReply, match="Z"):
        cells_from_reply(_reply(_row("A"), _row("Z")), ["A"], COUNTS, VALUES)


def test_reply_repeating_a_site_is_short():
    with pytest.raises(ShortReply, match="A"):
        cells_from_reply(_reply(_row("A"), _row("A")), ["A"], COUNTS, VALUES)


def test_reply_without_features_is_short():
    with pytest.raises(ShortReply):
        cells_from_reply({"type": "FeatureCollection"}, ["A"], COUNTS, VALUES)


def test_reply_without_count_is_short():
    reply = _reply(_feature(site_name="A", elevation_m_mean=1.0, all_count=16))
    with pytest.raises(ShortReply, match="A.*elevation_m_count"):
        cells_from_reply(reply, ["A"], COUNTS, VALUES)


def test_reply_null_mean_with_pixels_is_short():
    reply = _reply(_row("A", mean=None, count=13))
    with pytest.raises(ShortReply, match="A.*elevation_m_mean"):
        cells_from_reply(reply, ["A"], COUNTS, VALUES)


def test_reply_absent_mean_with_pixels_is_short():
    reply = _reply(_feature(site_name="A", elevation_m_count=13, all_count=16))
    with pytest.raises(ShortReply, match="A.*elevation_m_mean"):
        cells_from_reply(reply, ["A"], COUNTS, VALUES)


def test_reply_null_mean_with_no_pixels_is_nan():
    frame = cells_from_reply(_reply(_row("A", mean=None, count=0)), ["A"], COUNTS, VALUES)
    assert np.isnan(frame.loc["A", "elevation_m_mean"])
    assert frame.loc["A", "elevation_m_count"] == 0.0


def test_reply_percentile_pairs_with_its_band_count():
    reply = _reply(_feature(site_name="A", h_p95=None, h_count=5))
    with pytest.raises(ShortReply, match="h_p95"):
        cells_from_reply(reply, ["A"], ["h_count"], ["h_p95"])


def test_value_without_a_requested_count_is_a_programming_error():
    with pytest.raises(ValueError, match="elevation_m_count"):
        cells_from_reply(_reply(_row("A")), ["A"], ["all_count"], VALUES)


# Review Focus 4
def test_sites_sharing_a_cell_each_get_a_row():
    frame = cells_from_reply(
        _reply(_row("GBR_DEV_CON", 7.0), _row("GBR_DEV_DRO", 7.0)), ["GBR_DEV_CON", "GBR_DEV_DRO"], COUNTS, VALUES
    )
    assert list(frame.index) == ["GBR_DEV_CON", "GBR_DEV_DRO"]
    assert frame.loc["GBR_DEV_CON", "elevation_m_mean"] == frame.loc["GBR_DEV_DRO", "elevation_m_mean"] == 7.0


def test_pixels_reply_is_one_row_per_pixel():
    reply = _reply(
        _feature(site_name="A", b1=1, b2=2), _feature(site_name="A", b1=3, b2=4), _feature(site_name="B", b1=5, b2=6)
    )
    frame = pixels_from_reply(reply, ["A", "B"], ["b1", "b2"])
    assert list(frame.columns) == ["site_name", "b1", "b2"]
    assert list(frame["site_name"]) == ["A", "A", "B"]
    assert list(frame["b1"]) == [1, 3, 5]


def test_pixels_reply_refuses_site_with_no_pixels():
    reply = _reply(_feature(site_name="A", b1=1, b2=2))
    with pytest.raises(ShortReply, match="B"):
        pixels_from_reply(reply, ["A", "B"], ["b1", "b2"])


def test_pixels_reply_refuses_an_unasked_site():
    reply = _reply(_feature(site_name="A", b1=1), _feature(site_name="Z", b1=2))
    with pytest.raises(ShortReply, match="Z"):
        pixels_from_reply(reply, ["A"], ["b1"])


def test_pixels_reply_refuses_a_null_band():
    # Every band is unmasked before sampling, so a null means the reply is broken.
    reply = _reply(_feature(site_name="A", b1=None))
    with pytest.raises(ShortReply, match="b1"):
        pixels_from_reply(reply, ["A"], ["b1"])


class _Call:
    def __init__(self, answer):
        self.answer = answer

    def getInfo(self):  # noqa: N802
        return self.answer


class _FakeImage:
    """Stands in for ee.Image: hands out scripted replies and records what it was asked."""

    def __init__(self, replies, bands=("elevation_m", "all")):
        self.replies = list(replies)
        self.bands = list(bands)
        self.requests = []

    def bandNames(self):  # noqa: N802
        return _Call(self.bands)

    def reduceRegions(self, collection, reducer, crs):  # noqa: N802
        self.requests.append(collection)
        return _Call(self.replies.pop(0))

    def sampleRegions(self, collection, properties, projection, geometries):  # noqa: N802
        self.requests.append(collection)
        return _Call(self.replies.pop(0))


@pytest.fixture
def no_ee(monkeypatch):
    monkeypatch.setattr(earth_engine, "cell_collection", lambda part: list(part["site_name"]))
    monkeypatch.setattr(earth_engine, "BACKOFF_S", 0.0)


SITES = pd.DataFrame({"site_name": ["A", "B", "C"]})


def test_reduce_cells_batches_and_joins_in_site_order(no_ee):
    image = _FakeImage([_reply(_row("A", 1.0), _row("B", 2.0)), _reply(_row("C", 3.0))])
    frame = reduce_cells(image, "reducer", "projection", SITES, COUNTS, VALUES, "terrain", batch_size=2)

    assert image.requests == [["A", "B"], ["C"]]
    assert list(frame.index) == ["A", "B", "C"]
    assert list(frame["elevation_m_mean"]) == [1.0, 2.0, 3.0]


# Review Focus 1
def test_reduce_cells_retries_a_short_reply_then_succeeds(no_ee):
    image = _FakeImage([_reply(_row("A")), _reply(_row("A"), _row("B"), _row("C"))])
    frame = reduce_cells(image, "reducer", "projection", SITES, COUNTS, VALUES, "terrain")
    assert list(frame.index) == ["A", "B", "C"]
    assert len(image.requests) == 2


def test_reduce_cells_stops_naming_the_sites_when_the_reply_stays_short(no_ee):
    image = _FakeImage([_reply(_row("A"), _row("B"))] * 6)
    with pytest.raises(ExtractionFailure, match="terrain.*C"):
        reduce_cells(image, "reducer", "projection", SITES, COUNTS, VALUES, "terrain")
    assert len(image.requests) == 6


def test_reduce_cells_checks_band_names_before_any_request(no_ee):
    image = _FakeImage([_reply(_row("A"))], bands=("elevation", "all"))
    with pytest.raises(ExtractionFailure, match="elevation_m"):
        reduce_cells(image, "reducer", "projection", SITES, COUNTS, VALUES, "terrain")
    assert image.requests == []


def test_sample_cells_joins_pixels_across_batches(no_ee):
    image = _FakeImage(
        [
            _reply(_feature(site_name="A", b1=1), _feature(site_name="A", b1=2), _feature(site_name="B", b1=3)),
            _reply(_feature(site_name="C", b1=4)),
        ],
        bands=("b1", "land"),
    )
    frame = sample_cells(image, "projection", SITES, ["b1"], "phenology", batch_size=2)
    assert list(frame["site_name"]) == ["A", "A", "B", "C"]
    assert list(frame["b1"]) == [1, 2, 3, 4]


def test_sample_cells_stops_naming_a_site_that_never_returns_pixels(no_ee):
    image = _FakeImage([_reply(_feature(site_name="A", b1=1), _feature(site_name="B", b1=2))] * 6, bands=("b1", "land"))
    with pytest.raises(ExtractionFailure, match="phenology.*C"):
        sample_cells(image, "projection", SITES, ["b1"], "phenology")

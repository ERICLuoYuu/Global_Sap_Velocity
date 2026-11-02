"""The Earth Engine retry wrapper: which failures are transient, and which must not be swallowed."""

import pytest

from src.serving import era5_gee_daily

TRUNCATED = "EOF: reading array data, expected 262100 bytes got 64016"


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    """The jittered backoff would make these tests take minutes."""
    monkeypatch.setattr(era5_gee_daily.time, "sleep", lambda seconds: None)


def test_a_truncated_pixel_payload_is_retried():
    # Earth Engine sometimes closes the pixel stream early; numpy reports the short read as a
    # ValueError, not an EEException. The request was accepted and the transport failed, so it is
    # transient -- unretried it killed a whole year mid-run (jobs 46733052 at 180/366, 46733618 at 88/92).
    attempts = []

    def request():
        attempts.append(1)
        if len(attempts) < 3:
            raise ValueError(TRUNCATED)
        return "pixels"

    assert era5_gee_daily._with_retries(request) == "pixels"
    assert len(attempts) == 3


def test_a_value_error_that_is_not_a_short_read_is_raised_at_once():
    # Guards the lazy fix of catching ValueError wholesale: a real defect must surface on the first
    # call, not be retried MAX_ATTEMPTS times and re-raised minutes later.
    attempts = []

    def request():
        attempts.append(1)
        raise ValueError("no cells to acquire")

    with pytest.raises(ValueError, match="no cells"):
        era5_gee_daily._with_retries(request)
    assert len(attempts) == 1


def test_a_refusal_is_retried_to_the_attempt_limit_and_then_raised():
    # The behaviour that already existed must survive the change.
    attempts = []

    def request():
        attempts.append(1)
        raise era5_gee_daily.ee.EEException("Too Many Requests")

    with pytest.raises(era5_gee_daily.ee.EEException):
        era5_gee_daily._with_retries(request)
    assert len(attempts) == era5_gee_daily.MAX_ATTEMPTS

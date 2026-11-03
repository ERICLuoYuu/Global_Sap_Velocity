from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from src.static_features import results
from src.static_features.results import (
    NEIGHBOUR,
    NO_DATA,
    OK,
    ExtractionFailure,
    ProductResult,
    read_part,
    validate,
    write_part,
)

SITES = ["A", "B"]


def _frame(rows, sites=SITES):
    frame = pd.DataFrame(rows, index=pd.Index(sites, name="site_name"))
    return frame


def _result(values=None, status=None, coverage=None, by_year=None, extras=None, provenance=None, product="demo"):
    values = _frame({"x": [1.0, np.nan]}) if values is None else values
    status = _frame({"x": [OK, NO_DATA]}) if status is None else status
    coverage = _frame({"x": [1.0, 0.0]}) if coverage is None else coverage
    return ProductResult(
        product=product,
        values=values,
        status=status,
        coverage=coverage,
        by_year=by_year,
        extras=extras or {},
        provenance=provenance or {"source": "test"},
    )


def test_a_consistent_result_passes():
    validate(_result(), SITES)


def test_neighbour_with_a_value_passes():
    validate(_result(values=_frame({"x": [1.0, 2.0]}), status=_frame({"x": [OK, NEIGHBOUR]})), SITES)


def test_validate_refuses_nan_with_ok_status():
    bad = _result(status=_frame({"x": [OK, OK]}))
    with pytest.raises(ExtractionFailure, match="demo.*x.*B"):
        validate(bad, SITES)


def test_validate_refuses_nan_with_neighbour_status():
    bad = _result(status=_frame({"x": [OK, NEIGHBOUR]}))
    with pytest.raises(ExtractionFailure, match="B"):
        validate(bad, SITES)


def test_validate_refuses_value_with_no_data_status():
    bad = _result(values=_frame({"x": [1.0, 2.0]}))
    with pytest.raises(ExtractionFailure, match="demo.*x.*B"):
        validate(bad, SITES)


def test_validate_refuses_unknown_status():
    bad = _result(status=_frame({"x": [OK, "failed"]}))
    with pytest.raises(ExtractionFailure, match="failed"):
        validate(bad, SITES)


def test_validate_checks_each_site_against_its_own_row_not_by_position():
    status = _frame({"x": [NO_DATA, OK]}, sites=["B", "A"])
    validate(_result(status=status), SITES)


def test_validate_refuses_a_missing_site():
    bad = _result(values=_frame({"x": [1.0]}, sites=["A"]), status=_frame({"x": [OK]}, sites=["A"]))
    with pytest.raises(ExtractionFailure, match="B"):
        validate(bad, SITES)


def test_validate_refuses_an_extra_site():
    with pytest.raises(ExtractionFailure, match="B"):
        validate(_result(), ["A"])


def test_validate_refuses_a_repeated_site():
    repeated = ["A", "A"]
    bad = _result(values=_frame({"x": [1.0, 2.0]}, repeated), status=_frame({"x": [OK, OK]}, repeated))
    with pytest.raises(ExtractionFailure, match="A"):
        validate(bad, SITES)


def test_validate_refuses_values_and_status_with_different_columns():
    bad = _result(status=_frame({"y": [OK, NO_DATA]}))
    with pytest.raises(ExtractionFailure, match="x"):
        validate(bad, SITES)


def test_validate_refuses_coverage_outside_unit_interval():
    with pytest.raises(ExtractionFailure, match="coverage"):
        validate(_result(coverage=_frame({"x": [1.5, 0.0]})), SITES)
    with pytest.raises(ExtractionFailure, match="coverage"):
        validate(_result(coverage=_frame({"x": [-0.1, 0.0]})), SITES)


def test_validate_refuses_nan_coverage():
    with pytest.raises(ExtractionFailure, match="coverage"):
        validate(_result(coverage=_frame({"x": [1.0, np.nan]})), SITES)


def test_validate_refuses_coverage_of_an_unknown_column():
    with pytest.raises(ExtractionFailure, match="coverage"):
        validate(_result(coverage=_frame({"z": [1.0, 0.0]})), SITES)


def test_a_product_may_measure_no_coverage():
    validate(_result(coverage=pd.DataFrame(index=pd.Index(SITES, name="site_name"))), SITES)


def _by_year(value=1.0, status=OK, site="A"):
    return pd.DataFrame({"site_name": [site], "year": [2010], "column": ["x"], "value": [value], "status": [status]})


def test_by_year_is_checked_like_the_values():
    validate(_result(by_year=_by_year()), SITES)
    with pytest.raises(ExtractionFailure, match="by_year"):
        validate(_result(by_year=_by_year(np.nan, OK)), SITES)
    with pytest.raises(ExtractionFailure, match="by_year"):
        validate(_result(by_year=_by_year(1.0, NO_DATA)), SITES)
    with pytest.raises(ExtractionFailure, match="by_year"):
        validate(_result(by_year=_by_year(1.0, "failed")), SITES)
    with pytest.raises(ExtractionFailure, match="by_year"):
        validate(_result(by_year=_by_year(site="Z")), SITES)
    with pytest.raises(ExtractionFailure, match="by_year"):
        validate(_result(by_year=_by_year().drop(columns="year")), SITES)


def test_validate_refuses_an_extra_named_like_a_part_file():
    with pytest.raises(ExtractionFailure, match="values"):
        validate(_result(extras={"values": pd.DataFrame({"a": [1]})}), SITES)


def test_part_round_trips_floats_exactly(tmp_path):
    # to_csv with read_csv's default parser moves floats by one ulp.
    tricky = 0.1 + 0.2
    values = _frame({"x": [tricky, np.nan], "label": ["NA", None]})
    status = _frame({"x": [OK, NO_DATA], "label": [OK, NO_DATA]})
    original = _result(
        values=values,
        status=status,
        coverage=_frame({"x": [tricky / 2, 0.0]}),
        by_year=_by_year(tricky),
        extras={"pixels": pd.DataFrame({"site_name": ["A"], "evi": [tricky]})},
        provenance={"source": "test", "r": 0.25},
    )
    write_part(original, tmp_path, "hash123")
    back, site_hash = read_part(tmp_path, "demo")

    assert site_hash == "hash123"
    assert back.product == "demo"
    assert back.values.loc["A", "x"] == tricky
    assert np.isnan(back.values.loc["B", "x"])
    assert back.values.loc["A", "label"] == "NA"  # a literal "NA" is a value, only a blank is missing
    assert pd.isna(back.values.loc["B", "label"])
    assert back.status.loc["B", "x"] == NO_DATA
    assert back.coverage.loc["A", "x"] == tricky / 2
    assert back.by_year.loc[0, "value"] == tricky
    assert back.extras["pixels"].loc[0, "evi"] == tricky
    assert back.provenance == {"source": "test", "r": 0.25}
    validate(back, SITES)


def test_part_without_by_year_reads_back_as_none(tmp_path):
    write_part(_result(), tmp_path, "h")
    back, _ = read_part(tmp_path, "demo")
    assert back.by_year is None and back.extras == {}


def test_provenance_accepts_numpy_numbers(tmp_path):
    write_part(_result(provenance={"r": np.float64(0.5), "n": np.int64(3)}), tmp_path, "h")
    back, _ = read_part(tmp_path, "demo")
    assert back.provenance == {"r": 0.5, "n": 3}


def test_read_part_names_a_missing_product(tmp_path):
    with pytest.raises(ExtractionFailure, match="demo"):
        read_part(tmp_path, "demo")


# Review Focus 2
def test_write_part_failure_keeps_previous_part(tmp_path, monkeypatch):
    write_part(_result(values=_frame({"x": [1.0, np.nan]})), tmp_path, "old")

    def full_disk(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(results.json, "dump", full_disk)
    with pytest.raises(OSError):
        write_part(_result(values=_frame({"x": [9.0, np.nan]})), tmp_path, "new")
    monkeypatch.undo()

    back, site_hash = read_part(tmp_path, "demo")
    assert site_hash == "old" and back.values.loc["A", "x"] == 1.0
    assert sorted(path.name for path in tmp_path.iterdir()) == ["demo"]  # no temp directory left


def test_write_part_replaces_the_previous_part(tmp_path):
    write_part(_result(values=_frame({"x": [1.0, np.nan]})), tmp_path, "old")
    write_part(_result(values=_frame({"x": [9.0, np.nan]})), tmp_path, "new")

    back, site_hash = read_part(tmp_path, "demo")
    assert site_hash == "new" and back.values.loc["A", "x"] == 9.0
    assert sorted(path.name for path in tmp_path.iterdir()) == ["demo"]


def test_write_part_removes_a_temp_directory_left_by_a_killed_job(tmp_path):
    stale = tmp_path / ".tmp_demo_999999"
    stale.mkdir()
    (stale / "values.csv").write_text("half a file", encoding="utf-8")
    other = tmp_path / ".tmp_otherproduct_999999"
    other.mkdir()

    write_part(_result(), tmp_path, "h")

    assert not stale.exists()
    assert other.exists()  # another product's job may still be running


def test_read_part_never_reads_a_temp_directory(tmp_path):
    stale = tmp_path / ".tmp_demo_999999"
    stale.mkdir()
    (stale / "site_hash.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(ExtractionFailure):
        read_part(tmp_path, "demo")


def test_provenance_file_is_plain_json(tmp_path):
    path = write_part(_result(), tmp_path, "h")
    assert json.loads((path / "provenance.json").read_text(encoding="utf-8")) == {"source": "test"}


def test_validate_refuses_an_infinite_value():
    with pytest.raises(ExtractionFailure, match="infinite"):
        validate(_result(values=_frame({"x": [np.inf, np.nan]})), SITES)


def test_code_state_names_the_commit_and_whether_the_package_differs_from_it():
    state = results.code_state()

    assert len(state["commit"]) == 40
    assert isinstance(state["package_clean"], bool)


def test_code_state_reads_a_package_change_as_not_clean(monkeypatch):
    class Answer:
        returncode = 0

        def __init__(self, stdout):
            self.stdout = stdout

    answers = iter([Answer("abc\n"), Answer(" M src/static_features/grid.py\n")])
    monkeypatch.setattr(results.subprocess, "run", lambda *_, **__: next(answers))

    assert results.code_state() == {"commit": "abc", "package_clean": False}


def test_code_state_outside_a_repository_claims_nothing(monkeypatch):
    class Answer:
        returncode = 128
        stdout = ""

    monkeypatch.setattr(results.subprocess, "run", lambda *_, **__: Answer())

    assert results.code_state() == {"commit": None, "package_clean": None}

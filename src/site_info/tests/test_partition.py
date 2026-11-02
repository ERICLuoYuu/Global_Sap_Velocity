"""Tests for src/site_info/partition.py."""

from __future__ import annotations

import pandas as pd
import pytest
from src.site_info.partition import PartitionError, partition, site_names


def write_site_info(path, names, *, line_terminator="\n"):
    frame = pd.DataFrame(
        {
            "start_date": ["2015-01-01 00:00:00+00:00"] * len(names),
            "end_date": ["2015-12-31 00:00:00+00:00"] * len(names),
            "lat": [48.0] * len(names),
            "lon": [7.0] * len(names),
            "site_name": names,
        }
    )
    frame.to_csv(path, index=False, lineterminator=line_terminator)
    return path


# --------------------------------------------------------------------------
# site_names
# --------------------------------------------------------------------------


def test_reads_the_site_name_column(tmp_path):
    path = write_site_info(tmp_path / "s.csv", ["ARG_MAZ", "DEU_HOH"])
    assert site_names(path) == ["ARG_MAZ", "DEU_HOH"]


def test_windows_line_endings_do_not_poison_the_comparison(tmp_path):
    """A trailing \\r makes every set intersection silently empty.

    The delivered sapwood/site_info.csv is CRLF, so this is the real file, not
    a hypothetical one.
    """
    crlf = write_site_info(tmp_path / "crlf.csv", ["ARG_MAZ"], line_terminator="\r\n")
    lf = write_site_info(tmp_path / "lf.csv", ["ARG_MAZ"])

    assert site_names(crlf) == site_names(lf) == ["ARG_MAZ"]
    assert partition(site_names(crlf), site_names(lf)).new == ()


def test_duplicate_site_name_raises(tmp_path):
    path = write_site_info(tmp_path / "dup.csv", ["ARG_MAZ", "ARG_MAZ"])
    with pytest.raises(PartitionError, match="duplicate"):
        site_names(path)


def test_missing_column_raises(tmp_path):
    pd.DataFrame({"name": ["ARG_MAZ"]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(PartitionError, match="site_name"):
        site_names(tmp_path / "bad.csv")


def test_absent_file_raises(tmp_path):
    with pytest.raises(PartitionError, match="no such file"):
        site_names(tmp_path / "nope.csv")


# --------------------------------------------------------------------------
# partition
# --------------------------------------------------------------------------


def test_splits_into_shared_new_and_reference_only():
    result = partition(
        target=["A", "B", "C"],
        reference=["B", "C", "D"],
    )
    assert result.shared == ("B", "C")
    assert result.new == ("A",)
    assert result.reference_only == ("D",)


def test_every_target_site_lands_in_exactly_one_bucket():
    target = ["A", "B", "C", "D"]
    result = partition(target, reference=["C", "D", "E"])
    assert set(result.shared) | set(result.new) == set(target)
    assert set(result.shared) & set(result.new) == set()


def test_a_reference_that_covers_everything_leaves_nothing_to_extract():
    result = partition(["A", "B"], reference=["A", "B", "C"])
    assert result.new == ()
    assert result.reference_only == ("C",)


def test_an_empty_reference_means_extract_everything():
    result = partition(["A", "B"], reference=[])
    assert result.new == ("A", "B")
    assert result.shared == ()


def test_prefix_colliding_codes_are_distinct_members():
    """Set membership is exact, unlike the prefix globs elsewhere in the tree."""
    result = partition(
        target=["CHE_DAV", "CHE_DAV_SEE", "SWE_NOR", "SWE_NOR_ST1_AF1"],
        reference=["CHE_DAV_SEE", "SWE_NOR_ST1_AF1"],
    )
    assert result.new == ("CHE_DAV", "SWE_NOR")
    assert result.shared == ("CHE_DAV_SEE", "SWE_NOR_ST1_AF1")


def test_summary_counts_add_up():
    result = partition(["A", "B", "C"], reference=["C", "D"])
    assert "target = 3 (1 reusable + 2 to extract)" in result.summary()
    assert "reference-only = 1" in result.summary()

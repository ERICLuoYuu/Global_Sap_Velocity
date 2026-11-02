"""Tests for the land-cover extractor.

Only the naming is covered here: the Earth Engine calls need a network round
trip, but the mapping from LC_Type1 to the short IGBP code does not, and it was
previously unreachable from any test because it lived inline in the loop.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.site_info.pft_resolution import (  # noqa: E402
    IGBP_BY_LC_TYPE1,
    LandcoverVocabularyError,
)

_SPEC = importlib.util.spec_from_file_location("extract_pft", PROJECT_ROOT / "src" / "Extractors" / "extract_pft.py")
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


EXPECTED_COLUMNS = [
    "site_name",
    "year",
    "landcover_source_year",
    "lon",
    "lat",
    "landcover_type",
    "pft",
]


def test_the_row_carries_both_the_raw_code_and_the_short_name():
    row = mod.landcover_row("SITE_A", 2007, 2007, 7.0, 51.0, 4)
    assert list(row.columns) == EXPECTED_COLUMNS  # nosec B101
    assert row["landcover_type"].iloc[0] == 4  # nosec B101
    assert row["pft"].iloc[0] == "DBF"  # nosec B101


def test_the_raw_integer_survives_so_the_naming_stays_reversible():
    """Replacing it would cost a getInfo round trip per site-year to undo."""
    row = mod.landcover_row("S", 2007, 2007, 0.0, 0.0, 8)
    assert row["landcover_type"].iloc[0] == 8  # nosec B101
    assert row["landcover_type"].iloc[0] != "WSA"  # nosec B101
    assert row["pft"].iloc[0] == "WSA"  # nosec B101


def test_the_borrowed_source_year_is_still_recorded_separately():
    row = mod.landcover_row("S", 2025, 2024, 1.0, 2.0, 1)
    assert row["year"].iloc[0] == 2025  # nosec B101
    assert row["landcover_source_year"].iloc[0] == 2024  # nosec B101


def test_every_published_class_is_nameable():
    for value in range(1, 18):
        assert (
            mod.landcover_row("S", 2001, 2001, 0.0, 0.0, value)["pft"].iloc[0]
            == (  # nosec B101
                IGBP_BY_LC_TYPE1[value]
            )
        )


def test_the_name_written_is_never_a_long_name():
    """A long name in this column would one-hot beside si_igbp's short codes."""
    for value in range(1, 18):
        name = mod.landcover_row("S", 2001, 2001, 0.0, 0.0, value)["pft"].iloc[0]
        assert " " not in name  # nosec B101
        assert name.isupper()  # nosec B101


def test_an_unmapped_code_raises_instead_of_writing_a_blank_name():
    with pytest.raises(LandcoverVocabularyError):
        mod.landcover_row("S", 2007, 2007, 0.0, 0.0, 18)


def test_a_missing_class_raises_instead_of_writing_a_blank_name():
    with pytest.raises(LandcoverVocabularyError):
        mod.landcover_row("S", 2007, 2007, 0.0, 0.0, None)


def test_the_extractor_imports_the_shared_mapping_rather_than_copying_it():
    """A second private copy of a mapping is how the VPD unit error survived."""
    source = (PROJECT_ROOT / "src" / "Extractors" / "extract_pft.py").read_text(encoding="utf-8")
    assert "from src.site_info.pft_resolution import igbp_code" in source  # nosec B101
    for long_name in ("Evergreen Needleleaf", "Deciduous Broadleaf", "Mixed Forest"):
        assert long_name not in source  # nosec B101

"""Tests for src/site_info/assemble_extracted.py."""

from __future__ import annotations

import pandas as pd
import pytest
from src.site_info.assemble_extracted import (
    AssemblyError,
    Product,
    assemble_product,
    build_site_id_map,
)

PLAIN = Product("plain", "some_dir", "some_file.csv")
NUMBERED = Product("numbered", "era5land_site_data", "era5.csv", renumber_site_id=True)


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def lines(path):
    return path.read_text(encoding="utf-8").splitlines()


# --------------------------------------------------------------------------
# the concatenation itself
# --------------------------------------------------------------------------


def test_output_is_the_reference_rows_then_the_delta_rows(tmp_path):
    reference = write(
        tmp_path / "ref.csv",
        "site_name,value\nARG_MAZ,1\nAUS_KAR,2\n",
    )
    delta = write(tmp_path / "delta.csv", "site_name,value\nDEU_HOH,3\n")
    out = tmp_path / "out.csv"

    report = assemble_product(PLAIN, reference, delta, out)

    assert lines(out) == [
        "site_name,value",
        "ARG_MAZ,1",
        "AUS_KAR,2",
        "DEU_HOH,3",
    ]
    assert report.reference_rows == 2
    assert report.delta_rows == 1
    assert report.sites == 3


def test_reference_values_are_reproduced_character_for_character(tmp_path):
    """The reuse claim in full: 165 sites are carried over, not recomputed.

    Values that a numeric round trip would quietly rewrite -- trailing zeros,
    exponent notation, a bare NA token, an empty field, a very long decimal.
    """
    body = "site_name,a,b,c,d\nARG_MAZ,0.0100,1e-05,NA,\nAUS_KAR,-1.9545000000000001,0,,0000123\n"
    reference = write(tmp_path / "ref.csv", body)
    delta = write(tmp_path / "delta.csv", "site_name,a,b,c,d\nDEU_HOH,1,2,3,4\n")
    out = tmp_path / "out.csv"

    assemble_product(PLAIN, reference, delta, out)

    assert lines(out)[:3] == body.splitlines()


def test_chunk_boundaries_do_not_change_the_output(tmp_path):
    """Streaming is an implementation detail; the file must not depend on it."""
    reference = write(
        tmp_path / "ref.csv",
        "site_name,value\n" + "".join(f"S{i:03d},{i}\n" for i in range(50)),
    )
    delta = write(tmp_path / "delta.csv", "site_name,value\nNEW,999\n")

    one_chunk = tmp_path / "one.csv"
    many_chunks = tmp_path / "many.csv"
    assemble_product(PLAIN, reference, delta, one_chunk, chunk_rows=10_000)
    assemble_product(PLAIN, reference, delta, many_chunks, chunk_rows=7)

    assert one_chunk.read_text() == many_chunks.read_text()


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------


def test_a_site_in_both_halves_is_refused(tmp_path):
    """Double-counting a site is the failure this whole split can produce."""
    reference = write(tmp_path / "ref.csv", "site_name,value\nARG_MAZ,1\n")
    delta = write(tmp_path / "delta.csv", "site_name,value\nARG_MAZ,2\n")
    out = tmp_path / "out.csv"

    with pytest.raises(AssemblyError, match="BOTH halves"):
        assemble_product(PLAIN, reference, delta, out)


def test_a_refused_assembly_leaves_no_output_behind(tmp_path):
    reference = write(tmp_path / "ref.csv", "site_name,value\nARG_MAZ,1\n")
    delta = write(tmp_path / "delta.csv", "site_name,value\nARG_MAZ,2\n")
    out = tmp_path / "out.csv"

    with pytest.raises(AssemblyError):
        assemble_product(PLAIN, reference, delta, out)

    assert not out.exists()
    assert not list(tmp_path.glob("*.partial"))


def test_a_missing_input_raises(tmp_path):
    reference = write(tmp_path / "ref.csv", "site_name,value\nARG_MAZ,1\n")
    with pytest.raises(AssemblyError, match="missing input"):
        assemble_product(PLAIN, reference, tmp_path / "absent.csv", tmp_path / "out.csv")


def test_an_input_without_site_name_raises(tmp_path):
    reference = write(tmp_path / "ref.csv", "name,value\nARG_MAZ,1\n")
    delta = write(tmp_path / "delta.csv", "site_name,value\nDEU_HOH,2\n")
    with pytest.raises(AssemblyError, match="site_name"):
        assemble_product(PLAIN, reference, delta, tmp_path / "out.csv")


def test_an_existing_output_is_replaced_not_appended_to(tmp_path):
    reference = write(tmp_path / "ref.csv", "site_name,value\nARG_MAZ,1\n")
    delta = write(tmp_path / "delta.csv", "site_name,value\nDEU_HOH,2\n")
    out = write(tmp_path / "out.csv", "site_name,value\nSTALE,999\n")

    assemble_product(PLAIN, reference, delta, out)

    assert "STALE" not in out.read_text()
    assert len(lines(out)) == 3


# --------------------------------------------------------------------------
# columns
# --------------------------------------------------------------------------


def test_a_column_only_the_delta_has_is_kept_and_reported(tmp_path):
    """Koppen: the reference predates it, the delta now extracts it."""
    reference = write(tmp_path / "ref.csv", "site_name,bio1\nARG_MAZ,10\n")
    delta = write(tmp_path / "delta.csv", "site_name,bio1,koppen_class\nDEU_HOH,11,25\n")
    out = tmp_path / "out.csv"

    report = assemble_product(PLAIN, reference, delta, out)

    assert report.delta_only_columns == ("koppen_class",)
    assert report.reference_only_columns == ()
    frame = pd.read_csv(out, dtype=str, na_filter=False)
    assert list(frame.columns) == ["site_name", "bio1", "koppen_class"]
    assert frame.loc[frame["site_name"] == "ARG_MAZ", "koppen_class"].item() == ""
    assert frame.loc[frame["site_name"] == "DEU_HOH", "koppen_class"].item() == "25"


def test_a_column_only_the_reference_has_is_kept_and_reported(tmp_path):
    reference = write(tmp_path / "ref.csv", "site_name,bio1,legacy\nARG_MAZ,10,x\n")
    delta = write(tmp_path / "delta.csv", "site_name,bio1\nDEU_HOH,11\n")

    report = assemble_product(PLAIN, reference, delta, tmp_path / "out.csv")

    assert report.reference_only_columns == ("legacy",)


def test_reference_column_order_survives(tmp_path):
    reference = write(tmp_path / "ref.csv", "site_name,z,a\nARG_MAZ,1,2\n")
    delta = write(tmp_path / "delta.csv", "a,z,site_name,extra\n2,1,DEU_HOH,9\n")

    assemble_product(PLAIN, reference, delta, tmp_path / "out.csv")

    assert lines(tmp_path / "out.csv")[0] == "site_name,z,a,extra"


# --------------------------------------------------------------------------
# site_id renumbering
# --------------------------------------------------------------------------


def test_site_id_is_renumbered_against_site_info(tmp_path):
    """Extracted alone, the delta's site_id restarts at 1 and collides."""
    site_info = write(
        tmp_path / "site_info.csv",
        "start_date,end_date,lat,lon,site_name\n"
        "2015-01-01,2015-12-31,1,1,ARG_MAZ\n"
        "2015-01-01,2015-12-31,2,2,AUS_KAR\n"
        "2015-01-01,2015-12-31,3,3,DEU_HOH\n",
    )
    reference = write(
        tmp_path / "ref.csv",
        "site_id,site_name,t\n1,ARG_MAZ,x\n2,AUS_KAR,y\n",
    )
    delta = write(tmp_path / "delta.csv", "site_id,site_name,t\n1,DEU_HOH,z\n")
    out = tmp_path / "out.csv"

    assemble_product(NUMBERED, reference, delta, out, site_id_by_name=build_site_id_map(site_info))

    frame = pd.read_csv(out, dtype=str)
    assert dict(zip(frame["site_name"], frame["site_id"])) == {
        "ARG_MAZ": "1",
        "AUS_KAR": "2",
        "DEU_HOH": "3",
    }
    assert frame["site_id"].is_unique


def test_a_row_whose_site_is_not_in_site_info_raises(tmp_path):
    site_info = write(
        tmp_path / "site_info.csv",
        "start_date,end_date,lat,lon,site_name\n2015-01-01,2015-12-31,1,1,ARG_MAZ\n",
    )
    reference = write(tmp_path / "ref.csv", "site_id,site_name,t\n1,ARG_MAZ,x\n")
    delta = write(tmp_path / "delta.csv", "site_id,site_name,t\n1,GHOST,z\n")

    with pytest.raises(AssemblyError, match="absent from site_info"):
        assemble_product(
            NUMBERED,
            reference,
            delta,
            tmp_path / "out.csv",
            site_id_by_name=build_site_id_map(site_info),
        )


def test_products_without_the_flag_keep_their_columns_untouched(tmp_path):
    site_info = write(
        tmp_path / "site_info.csv",
        "start_date,end_date,lat,lon,site_name\n2015-01-01,2015-12-31,1,1,ARG_MAZ\n2015-01-01,2015-12-31,2,2,DEU_HOH\n",
    )
    reference = write(tmp_path / "ref.csv", "site_id,site_name\n7,ARG_MAZ\n")
    delta = write(tmp_path / "delta.csv", "site_id,site_name\n7,DEU_HOH\n")
    out = tmp_path / "out.csv"

    assemble_product(PLAIN, reference, delta, out, site_id_by_name=build_site_id_map(site_info))

    # PLAIN has renumber_site_id=False: an odd site_id column is left alone.
    assert lines(out) == ["site_id,site_name", "7,ARG_MAZ", "7,DEU_HOH"]


def test_build_site_id_map_is_one_based_in_file_order(tmp_path):
    site_info = write(
        tmp_path / "site_info.csv",
        "start_date,end_date,lat,lon,site_name\n2015-01-01,2015-12-31,1,1,ZZZ\n2015-01-01,2015-12-31,2,2,AAA\n",
    )
    assert build_site_id_map(site_info) == {"ZZZ": 1, "AAA": 2}

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from src.static_features.sites import load_sites, site_list_hash

HEADER = "start_date,end_date,lat,lon,site_name\n"
ROW = "2009-11-19 03:00:00+00:00,2009-12-01 02:00:00+00:00,{lat},{lon},{name}\n"


def _write(tmp_path: Path, rows, igbp=None):
    """rows: (name, lat, lon). igbp: name -> si_igbp, default DBF; None as a value skips the file."""
    igbp = igbp or {}
    info = tmp_path / "site_info.csv"
    info.write_text(HEADER + "".join(ROW.format(name=n, lat=la, lon=lo) for n, la, lo in rows), encoding="utf-8")
    for name, _lat, _lon in rows:
        value = igbp.get(name, "DBF")
        if value is not None:
            (tmp_path / f"{name}_site_md.csv").write_text(f"si_code,si_igbp\n{name},{value}\n", encoding="utf-8")
    return info


def test_load_sites_adds_cell_and_igbp(tmp_path):
    info = _write(tmp_path, [("AAA", 51.5, 10.25), ("BBB", 90.0, -180.0)], igbp={"BBB": "ENF"})
    sites = load_sites(info, tmp_path)

    assert list(sites.columns) == ["site_name", "lat", "lon", "start_date", "end_date", "si_igbp", "row", "col"]
    first = sites.set_index("site_name").loc["AAA"]
    assert (first["row"], first["col"], first["si_igbp"]) == (4620, 22830, "DBF")
    assert sites.set_index("site_name").loc["BBB", "si_igbp"] == "ENF"
    assert first["start_date"] == "2009-11-19 03:00:00+00:00"


def test_load_sites_keeps_coordinates_exact(tmp_path):
    info = _write(tmp_path, [("AAA", -37.4222, 144.0944)])
    sites = load_sites(info, tmp_path)
    assert sites.loc[0, "lat"] == -37.4222 and sites.loc[0, "lon"] == 144.0944


def test_blank_igbp_is_kept_as_missing_not_refused(tmp_path):
    # Two corpus sites have a blank si_igbp, and only the agreement report's grouping reads it.
    info = _write(tmp_path, [("AAA", 51.5, 10.25)], igbp={"AAA": ""})
    assert pd.isna(load_sites(info, tmp_path).loc[0, "si_igbp"])


# Review Focus 4
def test_load_sites_refuses_duplicate_names(tmp_path):
    info = _write(tmp_path, [("GBR_DEV_CON", 56.03333, -3.716667), ("GBR_DEV_CON", 56.03333, -3.716667)])
    with pytest.raises(ValueError, match="GBR_DEV_CON"):
        load_sites(info, tmp_path)


def test_sites_sharing_a_cell_keep_their_own_rows(tmp_path):
    info = _write(tmp_path, [("GBR_DEV_CON", 56.03333, -3.716667), ("GBR_DEV_DRO", 56.03333, -3.716667)])
    sites = load_sites(info, tmp_path)
    assert list(sites["site_name"]) == ["GBR_DEV_CON", "GBR_DEV_DRO"]
    assert list(sites["row"]) == [4076, 4076] and list(sites["col"]) == [21153, 21153]


def test_load_sites_refuses_missing_metadata(tmp_path):
    info = _write(tmp_path, [("AAA", 51.5, 10.25), ("NOFILE", 50.0, 10.0)], igbp={"NOFILE": None})
    with pytest.raises(ValueError, match="NOFILE"):
        load_sites(info, tmp_path)


def test_load_sites_refuses_metadata_without_igbp_column(tmp_path):
    info = _write(tmp_path, [("AAA", 51.5, 10.25)])
    (tmp_path / "AAA_site_md.csv").write_text("si_code\nAAA\n", encoding="utf-8")
    with pytest.raises(ValueError, match="AAA"):
        load_sites(info, tmp_path)


def test_load_sites_refuses_a_coordinate_it_cannot_place(tmp_path):
    info = _write(tmp_path, [("AAA", 51.5, 10.25), ("BAD", 95.0, 10.0)])
    with pytest.raises(ValueError, match="BAD"):
        load_sites(info, tmp_path)


def test_load_sites_refuses_a_missing_column(tmp_path):
    info = tmp_path / "site_info.csv"
    info.write_text("lat,lon,site_name\n51.5,10.25,AAA\n", encoding="utf-8")
    with pytest.raises(ValueError, match="start_date"):
        load_sites(info, tmp_path)


def test_load_sites_refuses_an_empty_list(tmp_path):
    info = tmp_path / "site_info.csv"
    info.write_text(HEADER, encoding="utf-8")
    with pytest.raises(ValueError, match="no sites"):
        load_sites(info, tmp_path)


def test_site_list_hash_ignores_row_order_and_notices_a_moved_site():
    sites = pd.DataFrame({"site_name": ["A", "B"], "lat": [1.5, 2.5], "lon": [3.5, 4.5]})
    reordered = sites.iloc[::-1].reset_index(drop=True)
    moved = sites.assign(lat=[1.5, 2.6])
    renamed = sites.assign(site_name=["A", "C"])

    assert site_list_hash(sites) == site_list_hash(reordered)
    assert site_list_hash(sites) != site_list_hash(moved)
    assert site_list_hash(sites) != site_list_hash(renamed)
    assert len(site_list_hash(sites)) == 64

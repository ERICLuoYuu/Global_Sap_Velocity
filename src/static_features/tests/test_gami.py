from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest
from src.static_features.grid import cell_of
from src.static_features.products import gami
from src.static_features.products.gami import cell_mean, extract, fetch_inputs, read_order
from src.static_features.results import NO_DATA, OK, ExtractionFailure, RunContext, validate

FILL = -9999
N_LAT, N_LON = 43200, 86400  # 240 pixels per degree, so 2 x 2 pixels per cell; sparse, so cheap on disk
LAT, LON = 50.0525, 10.0525  # cell row 4793, col 22806, so pixel rows 9586-9587, cols 45612-45613
SITE_ROWS = (9586, 9588)
SITE_COLS = (45612, 45614)


def _write(tmp_path, cells, members=2, days=(0, 3652), latitude=None):
    """cells: (member, time position) -> 2 x 2 array for the site's cell."""
    directory = tmp_path / "gami_v2_1"
    directory.mkdir()
    with h5py.File(directory / "GAMIv2-1_2010-2020_100m.nc", "w") as handle:
        handle.attrs["_FillValue"] = FILL
        handle["time"] = np.array(days, dtype="f8")
        handle["latitude"] = latitude if latitude is not None else 90.0 - (np.arange(N_LAT) + 0.5) / 240
        handle["longitude"] = -180.0 + (np.arange(N_LON) + 0.5) / 240
        dataset = handle.create_dataset(
            "forest_age",
            shape=(members, N_LAT, N_LON, len(days)),
            dtype="i2",
            chunks=(1, 1000, 1000, 1),
            fillvalue=FILL,
            compression="gzip",
        )
        for (member, time), block in cells.items():
            dataset[member, SITE_ROWS[0] : SITE_ROWS[1], SITE_COLS[0] : SITE_COLS[1], time] = block


def _sites(**extra):
    row, col = cell_of(LAT, LON)
    return pd.DataFrame({"site_name": ["A"], "lat": [LAT], "lon": [LON], "row": [row], "col": [col], **extra})


def _context(tmp_path):
    return RunContext(sites=_sites(), raw_root=tmp_path, output_root=tmp_path / "out", project="p")


def test_gami_cell_mean_hand_computed():
    block = np.array([[10, -9999], [30, -9999]])
    assert cell_mean(block, -9999) == (20.0, 0.5)


def test_gami_cell_mean_of_no_forest():
    mean, share = cell_mean(np.array([[-9999, -9999]]), -9999)
    assert np.isnan(mean) and share == 0.0


def test_gami_cell_mean_of_an_empty_block_is_refused():
    with pytest.raises(ValueError):
        cell_mean(np.empty((0, 0), dtype="i2"), -9999)


def test_gami_averages_within_each_member_then_across_members(tmp_path):
    # Member 0 mean 20 of 4 forest pixels; member 1 mean 50 of 1. Pooling pixels would give 26.
    _write(
        tmp_path,
        {
            (0, 0): [[10, 20], [20, 30]],
            (1, 0): [[50, FILL], [FILL, FILL]],
        },
    )
    result = extract(_context(tmp_path))

    assert result.values.loc["A", "forest_age_gami"] == 35.0
    assert result.status.loc["A", "forest_age_gami"] == OK
    assert result.coverage.loc["A", "forest_age_gami"] == pytest.approx((1.0 + 0.25) / 2)
    validate(result, ["A"])


def test_gami_member_without_forest_is_left_out_of_the_mean_but_lowers_coverage(tmp_path):
    _write(tmp_path, {(0, 0): [[40, 40], [40, 40]]})
    result = extract(_context(tmp_path))

    assert result.values.loc["A", "forest_age_gami"] == 40.0
    assert result.coverage.loc["A", "forest_age_gami"] == 0.5


def test_gami_no_forest_is_no_data_with_zero_coverage(tmp_path):
    _write(tmp_path, {})
    result = extract(_context(tmp_path))

    assert np.isnan(result.values.loc["A", "forest_age_gami"])
    assert result.status.loc["A", "forest_age_gami"] == NO_DATA
    assert result.coverage.loc["A", "forest_age_gami"] == 0.0
    validate(result, ["A"])


def test_gami_uses_2010_and_keeps_2020_in_by_year(tmp_path):
    _write(
        tmp_path,
        {
            (0, 0): [[10, 10], [10, 10]],
            (1, 0): [[10, 10], [10, 10]],
            (0, 1): [[70, 70], [70, 70]],
            (1, 1): [[70, 70], [70, 70]],
        },
    )
    result = extract(_context(tmp_path))

    assert result.values.loc["A", "forest_age_gami"] == 10.0
    table = result.by_year.set_index("year")
    assert table.loc[2010, "value"] == 10.0 and table.loc[2020, "value"] == 70.0
    assert set(result.by_year["column"]) == {"forest_age_gami"}
    validate(result, ["A"])


def test_gami_refuses_a_file_without_both_years(tmp_path):
    _write(tmp_path, {}, days=(0,))
    with pytest.raises(ExtractionFailure, match="2020"):
        extract(_context(tmp_path))


def test_gami_provenance_records_the_authors_advice(tmp_path):
    _write(tmp_path, {})
    provenance = extract(_context(tmp_path)).provenance
    assert "two decades" in provenance["caution"]


def test_gami_read_order_groups_sites_in_one_chunk():
    # At 1125 pixels per degree with chunks of 1125 x 1125, a chunk is one degree square.
    sites = pd.DataFrame({"lat": [10.5, 50.5, 10.6, 50.6], "lon": [20.5, 20.5, 20.6, 20.6]})
    assert read_order(sites, 1125, 1125) == [1, 3, 0, 2]  # north first, and the two sites of a chunk adjacent


def test_gami_read_order_splits_chunks_by_column_too():
    sites = pd.DataFrame({"lat": [10.5, 10.6, 10.7], "lon": [30.5, 20.5, 30.6]})
    assert read_order(sites, 1125, 1125) == [1, 0, 2]


def test_fetch_inputs_pins_url_and_size(tmp_path, monkeypatch):
    requested = []
    monkeypatch.setattr(gami, "fetch", lambda item, directory: requested.append((item, directory)))
    fetch_inputs(_context(tmp_path))

    item, directory = requested[0]
    assert item.url == "https://datapub.gfz.de/download/10.5880.GFZ.1.4.2023.006-VEnuo/GAMIv2-1_2010-2020_100m.nc"
    assert item.size == 234035960874
    assert directory == tmp_path / "gami_v2_1"


def test_gami_refuses_a_latitude_axis_it_does_not_expect(tmp_path):
    _write(tmp_path, {}, latitude=-90.0 + (np.arange(N_LAT) + 0.5) / 240)
    with pytest.raises(ExtractionFailure, match="latitude"):
        extract(_context(tmp_path))

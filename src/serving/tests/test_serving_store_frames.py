import numpy as np
import pandas as pd
import pytest

from src.feature_contract.definitions import SWC_LAYERS, swc_anomaly_col, swc_level_col
from src.feature_contract.swvl_climatology import normalise_swvl
from src.serving.base_columns import DAY_COLUMN, TIME_COLUMN
from src.serving.era5_daily import solar_timestamp
from src.serving.frames import cell_frame, swvl_zscore
from src.serving.grid import Grid
from src.serving.store import (
    Store,
    column_file_stem,
    load_grid,
    open_year_column,
    read_year_column,
    save_grid,
    write_json,
    year_days,
)


def test_column_file_stem_and_year_days():
    assert column_file_stem("prcip/PET") == "prcip_over_PET"
    assert len(year_days(2016)) == 366
    assert str(year_days(2015).tz) == "UTC"


def test_year_column_is_created_nan_and_its_shape_is_checked(tmp_path):
    path = tmp_path / "era5_daily" / "2015" / "ta.npy"
    array = open_year_column(path, 3, 2)
    assert np.isnan(array).all()
    array[1, 1] = 5.0
    array.flush()
    del array
    assert open_year_column(path, 3, 2)[1, 1] == 5.0
    assert read_year_column(path, 3, 2)[1, 1] == 5.0
    with pytest.raises(ValueError):
        open_year_column(path, 4, 2)
    with pytest.raises(ValueError):
        read_year_column(path, 3, 3)
    with pytest.raises(FileNotFoundError):
        read_year_column(tmp_path / "missing.npy", 3, 2)


def test_grid_and_json_round_trip(tmp_path):
    store = Store(tmp_path)
    grid = Grid(west=5.95, north=50.05, step=0.1, n_rows=3, n_cols=4)
    save_grid(store, grid)
    assert load_grid(store) == grid
    write_json(tmp_path / "x" / "a.json", {"k": 1})
    assert [p.name for p in (tmp_path / "x").iterdir()] == ["a.json"]


def test_swvl_zscore_is_normalise_swvl_and_nan_without_climatology():
    raw = np.array([0.2, 0.35, np.nan])
    _, expected = normalise_swvl(raw, 0.06, 0.3)
    np.testing.assert_array_equal(swvl_zscore(raw, 0.3, 0.06), expected)
    assert np.isnan(swvl_zscore(raw, 0.3, np.nan)).all()
    assert np.isnan(swvl_zscore(raw, np.nan, 0.06)).all()
    assert np.isnan(swvl_zscore(raw, 0.3, 0.0)).all()


def _cell():
    cell = {"row": 0, "col": 0, "latitude": 50.0, "longitude": 7.25, "pft": "ENF", "elevation": 420.0}
    for layer in SWC_LAYERS:
        cell[f"swvl_mean_{layer}"] = 0.3
        cell[f"swvl_sigma_{layer}"] = 0.06
    return cell


def test_cell_frame_builds_anomaly_statics_and_timestamps():
    days = pd.date_range("2015-01-01", periods=5, freq="D", tz="UTC")
    series = {"ta": np.arange(5.0), swc_level_col(1): np.full(5, 0.36, dtype=np.float32)}
    frame = cell_frame(days, series, _cell())
    np.testing.assert_allclose(frame[swc_anomaly_col(1)], (float(np.float32(0.36)) - 0.3) / 0.06, rtol=1e-12)
    assert swc_anomaly_col(2) not in frame.columns
    assert (frame["elevation"] == 420.0).all()
    assert "slope" not in frame.columns
    assert (frame["pft"] == "ENF").all()
    assert (frame["latitude"] == 50.0).all() and (frame["longitude"] == 7.25).all()
    assert pd.DatetimeIndex(frame[DAY_COLUMN]).equals(days)
    assert pd.DatetimeIndex(frame[TIME_COLUMN]).equals(solar_timestamp(days, 7.25))
    np.testing.assert_array_equal(frame["ta"], np.arange(5.0))


def test_cell_frame_refuses_a_series_of_the_wrong_length():
    days = pd.date_range("2015-01-01", periods=5, freq="D", tz="UTC")
    with pytest.raises(ValueError):
        cell_frame(days, {"ta": np.arange(4.0)}, _cell())

import numpy as np
import pandas as pd
import pytest

from src.serving.acquire_era5 import STORE_COLUMNS, acquire, cell_windows
from src.serving.grid import Grid
from src.serving.store import Store, save_grid, year_days

GRID = Grid(west=-0.05, north=10.05, step=0.1, n_rows=7, n_cols=11)
YEAR = 2015
WINDOW = (3, 4)


def test_store_columns_are_the_daily_image_columns():
    from src.serving.era5_gee_daily import BAND_TO_COLUMN, DAILY_BANDS

    assert sorted(STORE_COLUMNS) == sorted(BAND_TO_COLUMN.get(band, band) for band in DAILY_BANDS)
    assert len(STORE_COLUMNS) == len(set(STORE_COLUMNS))


def test_windows_hold_every_cell_once_and_contain_it():
    rng = np.random.default_rng(0)
    rows, cols = rng.integers(0, GRID.n_rows, 60), rng.integers(0, GRID.n_cols, 60)
    windows = cell_windows(rows, cols, GRID, *WINDOW)
    assert sorted(np.concatenate([w.cells for w in windows]).tolist()) == list(range(60))
    for w in windows:
        assert np.all((rows[w.cells] >= w.row0) & (rows[w.cells] < w.row0 + w.height))
        assert np.all((cols[w.cells] >= w.col0) & (cols[w.cells] < w.col0 + w.width))
        assert w.row0 + w.height <= GRID.n_rows and w.col0 + w.width <= GRID.n_cols
    edge = cell_windows([6], [10], GRID, *WINDOW)[0]
    assert (edge.row0, edge.col0, edge.height, edge.width) == (6, 8, 1, 3)


def test_windows_refuse_cells_outside_the_grid():
    with pytest.raises(ValueError):
        cell_windows([7], [0], GRID)
    with pytest.raises(ValueError):
        cell_windows([0], [-1], GRID)


def _expected(day, k, row, col):
    return np.float32(day.dayofyear + 0.01 * row + 0.0001 * col + 1000 * k)


def _fetch(calls, empty_days=()):
    def fetch(day, grid, row0, col0, height, width):
        calls.append((day, row0, col0))
        r, c = np.meshgrid(np.arange(row0, row0 + height), np.arange(col0, col0 + width), indexing="ij")
        out = {
            column: (day.dayofyear + 0.01 * r + 0.0001 * c + 1000 * k).astype(np.float32)
            for k, column in enumerate(STORE_COLUMNS)
        }
        if day in empty_days:
            out["ta"] = np.full((height, width), np.nan, dtype=np.float32)
        return out

    return fetch


@pytest.fixture
def store(tmp_path):
    store = Store(tmp_path)
    save_grid(store, GRID)
    cells = pd.DataFrame({"row": [0, 0, 4, 6, 3], "col": [0, 9, 5, 10, 3]})
    cells.to_parquet(store.cells, index=False)
    return store, cells


def test_each_cell_gets_its_own_pixel_and_only_complete_days_are_marked(store):
    store, cells = store
    days = year_days(YEAR)
    calls = []
    summary = acquire(store, YEAR, days[:3], threads=3, fetch=_fetch(calls), window_shape=WINDOW)
    assert summary["acquired"] == 3 and summary["empty_days"] == []
    for k, column in enumerate(STORE_COLUMNS):
        array = np.load(store.era5(YEAR, column))
        assert array.shape == (len(days), len(cells))
        for d, day in enumerate(days[:3]):
            for i, (row, col) in enumerate(zip(cells["row"], cells["col"])):  # noqa: B905
                assert array[d, i] == _expected(day, k, row, col)
        assert np.isnan(array[3:]).all()
    assert all(store.era5_done(YEAR, day).exists() for day in days[:3])
    assert not store.era5_done(YEAR, days[3]).exists()
    assert len(calls) == 3 * len(cell_windows(cells["row"], cells["col"], GRID, *WINDOW))


def test_a_rerun_skips_marked_days(store):
    store, _ = store
    days = year_days(YEAR)[:2]
    acquire(store, YEAR, days, threads=2, fetch=_fetch([]), window_shape=WINDOW)
    calls = []
    summary = acquire(store, YEAR, days, threads=2, fetch=_fetch(calls), window_shape=WINDOW)
    assert calls == [] and summary["already_done"] == 2 and summary["acquired"] == 0


def test_a_day_with_no_valid_cell_is_left_unmarked(store):
    store, _ = store
    days = year_days(YEAR)[:2]
    summary = acquire(store, YEAR, days, threads=2, fetch=_fetch([], empty_days={days[1]}), window_shape=WINDOW)
    assert summary["empty_days"] == [str(days[1].date())]
    assert store.era5_done(YEAR, days[0]).exists() and not store.era5_done(YEAR, days[1]).exists()


def test_a_failed_fetch_propagates_and_marks_nothing(store):
    store, _ = store

    def fetch(*args):
        raise RuntimeError("quota")

    with pytest.raises(RuntimeError):
        acquire(store, YEAR, year_days(YEAR)[:2], threads=2, fetch=fetch, window_shape=WINDOW)
    assert not store.era5_done(YEAR, year_days(YEAR)[0]).exists()


def test_days_outside_the_year_are_refused(store):
    store, _ = store
    with pytest.raises(ValueError):
        acquire(store, YEAR, year_days(YEAR + 1)[:1], fetch=_fetch([]), window_shape=WINDOW)

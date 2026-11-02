import numpy as np
import pandas as pd
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from src.serving.lai import LAI_SCALE, composites, day_weights, lai_for_cells  # noqa: E402


def _write(directory, year, doy, values, nodata=None):
    path = directory / f"GlobMapLAIV3.A{year}{doy:03d}.Global.LAI.tif"
    profile = {
        "driver": "GTiff",
        "height": values.shape[0],
        "width": values.shape[1],
        "count": 1,
        "dtype": "int16",
        "crs": "EPSG:4326",
        "transform": from_origin(0.0, 3.0, 1.0, 1.0),
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values.astype(np.int16), 1)
    return path


def test_composites_are_dated_by_year_and_day_of_year(tmp_path):
    _write(tmp_path, 2015, 9, np.zeros((3, 4)))
    _write(tmp_path, 2014, 361, np.zeros((3, 4)))
    (tmp_path / "notes.txt").write_text("not a composite")
    assert composites(tmp_path)["date"].tolist() == [
        pd.Timestamp("2014-12-27", tz="UTC"),
        pd.Timestamp("2015-01-09", tz="UTC"),
    ]
    with pytest.raises(FileNotFoundError):
        composites(tmp_path / "empty")


def test_each_hour_takes_the_nearest_composite():
    dates = pd.DatetimeIndex(["2015-01-01", "2015-01-09"], tz="UTC")
    weights = day_weights(pd.date_range("2015-01-02", "2015-01-08", freq="D", tz="UTC"), dates)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    assert weights[0].tolist() == [1.0, 0.0]
    assert weights[-1].tolist() == [0.0, 1.0]
    assert weights[3, 1] >= 23 / 24 - 1e-12


def test_the_eight_day_tolerance_is_inclusive_and_later_days_have_no_lai():
    weights = day_weights(pd.date_range("2015-01-09", "2015-01-10", freq="D", tz="UTC"), pd.DatetimeIndex(["2015-01-01"], tz="UTC"))
    assert weights[0].tolist() == [1.0]
    assert weights[1].tolist() == [0.0]


def test_cells_read_their_nearest_pixel_blended_by_hour(tmp_path):
    first = np.arange(12).reshape(3, 4) * 10
    second = first + 100
    _write(tmp_path, 2015, 1, first)
    _write(tmp_path, 2015, 9, second)
    days = pd.date_range("2015-01-02", "2015-01-05", freq="D", tz="UTC")
    got = lai_for_cells(days, tmp_path, [2.5, 0.6, 1.49], [0.5, 3.9, 2.51])
    weights = day_weights(days, composites(tmp_path)["date"])
    assert got.dtype == np.float32
    for cell, (row, col) in enumerate([(0, 0), (2, 3), (1, 2)]):
        expected = (weights[:, 0] * first[row, col] + weights[:, 1] * second[row, col]) * LAI_SCALE
        np.testing.assert_allclose(got[:, cell], expected, rtol=1e-6)
    assert np.isnan(lai_for_cells(pd.date_range("2015-02-01", periods=1, tz="UTC"), tmp_path, [1.0], [1.0])).all()


def test_a_composite_declaring_nodata_is_refused(tmp_path):
    _write(tmp_path, 2015, 1, np.zeros((3, 4)), nodata=-1)
    with pytest.raises(NotImplementedError):
        lai_for_cells(pd.date_range("2015-01-01", periods=1, tz="UTC"), tmp_path, [1.0], [1.0])

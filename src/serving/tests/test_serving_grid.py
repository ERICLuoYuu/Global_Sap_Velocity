import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import Affine

from src.feature_contract.swvl_climatology import SwvlClimatology, lookup_cell
from src.serving.grid import (
    ERA5_LAND_NATIVE,
    MASK_CLASSES_TAG,
    Grid,
    climatology_for_cells,
    forest_cells,
    grid_from_raster,
)


def _mask(path, west=-10.05, north=50.05, step=0.1, tag=MASK_CLASSES_TAG):
    codes = np.array([[0, 1, 2], [4, 7, 11]], dtype=np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff", height=2, width=3, count=1, dtype="uint8", crs="EPSG:4326",
        transform=Affine(step, 0.0, west, 0.0, -step, north), nodata=0,
    ) as dst:
        dst.write(codes, 1)
        dst.update_tags(classes=tag)
    return path


def test_native_grid_centres_sit_on_tenths():
    assert ERA5_LAND_NATIVE.latitudes()[:2].tolist() == [90.0, 89.9]
    assert ERA5_LAND_NATIVE.longitudes()[[0, 1800, 3599]].tolist() == [-180.0, 0.0, 179.9]


def test_affine_and_tiles_cover_the_grid():
    grid = Grid(west=-180.05, north=78.05, step=0.1, n_rows=5, n_cols=7)
    assert grid.affine(2, 3)["translateX"] == pytest.approx(-179.75)
    assert grid.affine(2, 3)["translateY"] == pytest.approx(77.85)
    tiles = grid.tiles(2, 3)
    assert sum(h * w for _, _, h, w in tiles) == 35
    assert tiles[-1] == (4, 6, 1, 1)


def test_forest_cells_decode_the_mask(tmp_path):
    cells = forest_cells(_mask(tmp_path / "mask.tif"))
    assert cells["pft"].tolist() == ["ENF", "EBF", "DBF", "WET"]
    assert cells[["latitude", "longitude"]].iloc[0].tolist() == [50.0, -9.9]


@pytest.mark.parametrize(
    "kwargs",
    [{"west": -10.0}, {"step": 0.25}, {"tag": "1=ENF"}],
    ids=["half-cell-shift", "other-step", "recoded-classes"],
)
def test_masks_off_the_native_grid_or_recoded_are_refused(tmp_path, kwargs):
    with pytest.raises(ValueError):
        forest_cells(_mask(tmp_path / "mask.tif", **kwargs))


def _climatology():
    lats = 50.0 - (np.arange(6) + 0.5) * 0.100000457
    lons = -10.3 + (np.arange(8) + 0.5) * 0.100000457
    rng = np.random.default_rng(0)
    sigma = {layer: rng.uniform(0.01, 0.05, (6, 8)) for layer in (1, 2, 3, 4)}
    mean = {layer: rng.uniform(0.1, 0.4, (6, 8)) for layer in (1, 2, 3, 4)}
    sigma[2][2, 2] = 1e-10  # the nearest cube cell of the third test cell: forces the fallback search
    return SwvlClimatology(lats=lats, lons=lons, sigma=sigma, mean=mean, path="synthetic", sha256="", attrs={})


def test_cell_climatology_is_what_lookup_cell_gives_a_site_at_the_centre():
    clim = _climatology()
    cells = pd.DataFrame({"latitude": np.round(49.95 - 0.1 * np.arange(5), 6), "longitude": np.round(-10.2 + 0.1 * np.arange(5), 6)})
    got = climatology_for_cells(cells, clim)
    for k, (lat, lon) in enumerate(cells.itertuples(index=False)):
        for layer in (1, 2, 3, 4):
            want = lookup_cell(clim, lat, lon, layer)
            assert got.loc[k, f"swvl_sigma_{layer}"] == want.sigma
            assert got.loc[k, f"swvl_mean_{layer}"] == want.mean
            assert got.loc[k, f"swvl_fallback_{layer}"] == want.fallback

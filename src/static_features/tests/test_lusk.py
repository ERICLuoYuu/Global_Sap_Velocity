from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from src.static_features.fetch import Download
from src.static_features.products import lusk
from src.static_features.products.lusk import extract, fetch_inputs, read_trait
from src.static_features.results import NEIGHBOUR, NO_DATA, OK, ExtractionFailure, RunContext, validate

NODATA = -32768
PERFORMANCE = "{&quot;R2&quot;: 0.34, &quot;Pearson's r&quot;: 0.6, &quot;nRMSE&quot;: 0.17}"
LICENCE = "This dataset is provided for research purposes only."

# A site at (0.02, 0.02) projects to about x 1928, y 2571 in EPSG:6933. On this grid of 1000 m pixels from
# (-5000, 5000) that is row 2, col 6; read without projecting, the same numbers would land at row 4, col 5.
SITE_ROW, SITE_COL = 2, 6


def _write(
    directory, trait_id, mean=None, cov=None, aoa=None, unit="m", crs_tag="EPSG:6933", scale=0.01132, offset=370.9
):
    """mean, cov, aoa: {(row, col): raw int16}, -32768 elsewhere."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"X{trait_id}_mean_Shrub_Tree_Grass_1km.tif"
    profile = {
        "driver": "GTiff", "height": 10, "width": 10, "count": 3, "dtype": "int16",
        "transform": from_origin(-5000, 5000, 1000, 1000), "nodata": NODATA,
    }  # fmt: skip
    with rasterio.open(path, "w", **profile) as dst:
        for band, points in ((1, mean), (2, cov), (3, aoa)):
            grid = np.full((10, 10), NODATA, dtype="int16")
            for (row, col), raw in (points or {}).items():
                grid[row, col] = raw
            dst.write(grid, band)
        dst.scales = (scale, 0.001, 1.0)
        dst.offsets = (offset, 0.5, 0.0)
        dst.update_tags(
            crs=crs_tag,
            trait_unit=unit,
            trait_long_name=f"Trait {trait_id}",
            trait_short_name=f"T{trait_id}",
            model_performance=PERFORMANCE,
            license=LICENCE,
            version="1.1.0",
        )
    return path


SITES = pd.DataFrame({"site_name": ["A"], "lat": [0.02], "lon": [0.02]})


def _context(tmp_path):
    return RunContext(sites=SITES, raw_root=tmp_path, output_root=tmp_path / "out", project="p")


def test_lusk_projects_sites_to_equal_area(tmp_path):
    path = _write(tmp_path, 6, mean={(SITE_ROW, SITE_COL): 100})
    assert read_trait(path, SITES).loc["A", "mean_status"] == OK


def test_lusk_applies_scale_and_offset(tmp_path, monkeypatch):
    monkeypatch.setattr(lusk, "TRAIT_IDS", (6,))
    _write(
        tmp_path / "lusk_2026",
        6,
        mean={(SITE_ROW, SITE_COL): 100},
        cov={(SITE_ROW, SITE_COL): 200},
        aoa={(SITE_ROW, SITE_COL): 1},
    )
    result = extract(_context(tmp_path))

    assert result.values.loc["A", "root_depth_lusk"] == pytest.approx(372.032)
    assert result.values.loc["A", "root_depth_lusk_cov"] == pytest.approx(0.7)
    assert result.values.loc["A", "root_depth_lusk_aoa"] == 1.0
    assert result.status.loc["A", "root_depth_lusk"] == OK
    validate(result, ["A"])


def test_lusk_other_bands_read_at_the_chosen_pixel(tmp_path, monkeypatch):
    monkeypatch.setattr(lusk, "TRAIT_IDS", (6,))
    neighbour = (SITE_ROW, SITE_COL + 2)
    _write(
        tmp_path / "lusk_2026",
        6,
        mean={neighbour: 100},
        cov={neighbour: 200, (SITE_ROW, SITE_COL): 999},
        aoa={neighbour: 1, (SITE_ROW, SITE_COL): 0},
    )
    result = extract(_context(tmp_path))

    assert result.status.loc["A", "root_depth_lusk"] == NEIGHBOUR
    assert result.values.loc["A", "root_depth_lusk_cov"] == pytest.approx(0.7)
    assert result.values.loc["A", "root_depth_lusk_aoa"] == 1.0
    assert result.status.loc["A", "root_depth_lusk_cov"] == NEIGHBOUR


def test_lusk_reach_is_three_pixels(tmp_path):
    three_away = _write(tmp_path / "a", 6, mean={(SITE_ROW, SITE_COL + 3): 100})
    assert read_trait(three_away, SITES).loc["A", "mean_status"] == NEIGHBOUR
    four_away = _write(tmp_path / "b", 6, mean={(SITE_ROW + 4, SITE_COL): 100})
    assert read_trait(four_away, SITES).loc["A", "mean_status"] == NO_DATA


def test_lusk_nodata_is_tested_before_scaling(tmp_path, monkeypatch):
    # -32768 x 0.01132 + 370.9 is close to 0, a believable depth, if the test comes after the scaling.
    monkeypatch.setattr(lusk, "TRAIT_IDS", (6,))
    _write(tmp_path / "lusk_2026", 6)
    result = extract(_context(tmp_path))

    assert np.isnan(result.values.loc["A", "root_depth_lusk"])
    assert result.status.loc["A", "root_depth_lusk"] == NO_DATA
    assert result.status.loc["A", "root_depth_lusk_cov"] == NO_DATA
    validate(result, ["A"])


def test_lusk_empty_other_band_is_no_data_for_that_column_only(tmp_path, monkeypatch):
    monkeypatch.setattr(lusk, "TRAIT_IDS", (6,))
    _write(tmp_path / "lusk_2026", 6, mean={(SITE_ROW, SITE_COL): 100}, aoa={(SITE_ROW, SITE_COL): 1})
    result = extract(_context(tmp_path))

    assert result.status.loc["A", "root_depth_lusk"] == OK
    assert result.status.loc["A", "root_depth_lusk_cov"] == NO_DATA
    assert result.status.loc["A", "root_depth_lusk_aoa"] == OK
    validate(result, ["A"])


def test_lusk_refuses_file_whose_crs_tag_is_not_6933(tmp_path):
    path = _write(tmp_path, 6, mean={(SITE_ROW, SITE_COL): 100}, crs_tag="EPSG:4326")
    with pytest.raises(ExtractionFailure, match="X6.*6933"):
        read_trait(path, SITES)


def test_lusk_columns_name_the_two_agreement_traits(tmp_path, monkeypatch):
    monkeypatch.setattr(lusk, "TRAIT_IDS", (6, 4, 3117))
    directory = tmp_path / "lusk_2026"
    _write(directory, 6, unit="m")
    _write(directory, 4, unit="g cm3")
    _write(directory, 3117, unit="mm2 mg-1")
    result = extract(_context(tmp_path))

    assert list(result.values.columns) == [
        "root_depth_lusk", "root_depth_lusk_cov", "root_depth_lusk_aoa",
        "wood_density_lusk", "wood_density_lusk_cov", "wood_density_lusk_aoa",
        "lusk_3117", "lusk_3117_cov", "lusk_3117_aoa",
    ]  # fmt: skip


@pytest.mark.parametrize("trait_id, unit", [(6, "cm"), (4, "kg m-3")])
def test_lusk_refuses_an_unexpected_unit_on_the_agreement_traits(tmp_path, monkeypatch, trait_id, unit):
    monkeypatch.setattr(lusk, "TRAIT_IDS", (trait_id,))
    _write(tmp_path / "lusk_2026", trait_id, unit=unit)
    with pytest.raises(ExtractionFailure, match=f"X{trait_id}.*{unit}"):
        extract(_context(tmp_path))


def test_lusk_provenance_records_each_trait_and_the_licence(tmp_path, monkeypatch):
    monkeypatch.setattr(lusk, "TRAIT_IDS", (6, 282))
    directory = tmp_path / "lusk_2026"
    _write(directory, 6, unit="m")
    _write(directory, 282, unit="um")
    provenance = extract(_context(tmp_path)).provenance

    assert provenance["licence"] == LICENCE
    assert provenance["traits"]["6"]["column"] == "root_depth_lusk"
    assert provenance["traits"]["6"]["name"] == "Trait 6"
    assert provenance["traits"]["6"]["unit"] == "m"
    assert provenance["traits"]["6"]["pearson_r"] == 0.6
    assert provenance["traits"]["6"]["experimental"] is False
    assert provenance["traits"]["282"]["experimental"] is True
    assert provenance["checksum"]["X6_mean_Shrub_Tree_Grass_1km.tif"]


def test_all_31_traits_are_listed_and_15_are_experimental():
    assert len(lusk.TRAIT_IDS) == 31 and len(set(lusk.TRAIT_IDS)) == 31
    assert len(lusk.EXPERIMENTAL) == 15 and set(lusk.TRAIT_IDS) >= lusk.EXPERIMENTAL
    assert 1080 not in lusk.EXPERIMENTAL  # r 0.46 but unstarred; recorded as published


def test_fetch_inputs_takes_only_the_traits_from_the_record(tmp_path, monkeypatch):
    monkeypatch.setattr(lusk, "TRAIT_IDS", (6, 4))
    items = [
        Download("u6", "X6_mean_Shrub_Tree_Grass_1km.tif", size=1),
        Download("u4", "X4_mean_Shrub_Tree_Grass_1km.tif", size=1),
        Download("ux", "README.txt", size=1),
    ]
    fetched = []
    monkeypatch.setattr(lusk, "zenodo_downloads", lambda record_id, directory: items)
    monkeypatch.setattr(lusk, "fetch", lambda item, directory: fetched.append(item.name))

    fetch_inputs(_context(tmp_path))

    assert sorted(fetched) == ["X4_mean_Shrub_Tree_Grass_1km.tif", "X6_mean_Shrub_Tree_Grass_1km.tif"]


def test_fetch_inputs_stops_when_the_record_lacks_a_trait(tmp_path, monkeypatch):
    monkeypatch.setattr(lusk, "TRAIT_IDS", (6, 4))
    monkeypatch.setattr(
        lusk,
        "zenodo_downloads",
        lambda record_id, directory: [Download("u6", "X6_mean_Shrub_Tree_Grass_1km.tif", size=1)],
    )
    with pytest.raises(ExtractionFailure, match="X4"):
        fetch_inputs(_context(tmp_path))

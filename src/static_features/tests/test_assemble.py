from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from src.static_features import assemble as assemble_module
from src.static_features.assemble import aspect_report, assemble, main
from src.static_features.results import (
    BY_YEAR_COLUMNS,
    NO_DATA,
    OK,
    ExtractionFailure,
    ProductResult,
    RunContext,
    write_part,
)
from src.static_features.run import PRODUCTS
from src.static_features.sites import site_list_hash

SOIL_COLUMNS = [
    f"{prop}_{depth}"
    for prop in ("clay", "sand", "silt", "bdod", "cfvo", "soc", "phh2o", "nitrogen")
    for depth in ("0_5", "5_15", "15_30", "30_60", "60_100", "100_200")
]
BIO_COLUMNS = [f"bio{n}" for n in range(1, 20)]
AGE_COLUMNS = ["stand_age_TC000", "stand_age_TC010", "stand_age_TC020", "stand_age_TC030"]
FAN_COLUMNS = [
    "root_depth",
    "infiltration_depth",
    "deep_drainage",
    "drainage_frequency",
    "regolith_flush_rate",
    "gw_residence_time",
    "root_depth_mask_valid",
    "root_depth_fallback_px",
    "root_depth_continent",
    "root_depth_lat_pixel",
    "root_depth_lon_pixel",
]
COLUMNS = {
    "terrain": ["elevation_m", "slope_deg", "aspect_sin", "aspect_cos", "land_share"],
    "canopy": ["canopy_height_m", "canopy_height_p95_m"],
    "soil": [*SOIL_COLUMNS, "texture_class_0_5"],
    "worldclim": [*BIO_COLUMNS, "koppen_class", "koppen_name"],
    "stand_age": AGE_COLUMNS,
    "fan": FAN_COLUMNS,
    "han": ["root_depth_han", "root_depth_han_type"],
    "yang": ["wood_density"],
    "lusk": ["root_depth_lusk", "wood_density_lusk", "wood_density_lusk_cov", "wood_density_lusk_aoa"],
    "liang": ["tree_richness_per_ha"],
    "lesiv": ["fm_share_11", "fm_dominant"],
    "cci": ["agb_mg_ha"],
    "gami": ["forest_age_gami"],
    "tree_cover": ["tree_cover_pct"],
    "evi": ["evi_mean"],
    "phenology": ["phen_peak_sin"],
}
TEXT_COLUMNS = {"texture_class_0_5", "koppen_name", "root_depth_continent", "root_depth_han_type"}
NEW_COLUMNS = [
    "root_depth_han",
    "root_depth_han_type",
    "wood_density",
    "root_depth_lusk",
    "wood_density_lusk",
    "tree_richness_per_ha",
    "fm_share_11",
    "fm_dominant",
    "agb_mg_ha",
    "forest_age_gami",
    "tree_cover_pct",
    "evi_mean",
    "phen_peak_sin",
    "canopy_height_p95_m",
    "land_share",
]
ENV_HEADER = (
    "start_date,end_date,lat,lon,site_name,canopy_height_m,elevation_m,slope_deg,aspect_deg,aspect_standardized,"
    "aspect_category,aspect_sin,aspect_cos,bio1,bio2,bio3,bio4,bio5,bio6,bio7,bio8,bio9,bio10,bio11,bio12,bio13,bio14,"
    "bio15,bio16,bio17,bio18,bio19,koppen_class,koppen_name,slope_deg_centre"
)
SOIL_HEADER = (
    "start_date,end_date,lat,lon,site_name,data_source,offset_distance_m,snapped_lat,snapped_lon,clay_0_5,clay_5_15,"
    "clay_15_30,clay_30_60,clay_60_100,clay_100_200,sand_0_5,sand_5_15,sand_15_30,sand_30_60,sand_60_100,sand_100_200,"
    "silt_0_5,silt_5_15,silt_15_30,silt_30_60,silt_60_100,silt_100_200,bdod_0_5,bdod_5_15,bdod_15_30,bdod_30_60,"
    "bdod_60_100,bdod_100_200,cfvo_0_5,cfvo_5_15,cfvo_15_30,cfvo_30_60,cfvo_60_100,cfvo_100_200,soc_0_5,soc_5_15,"
    "soc_15_30,soc_30_60,soc_60_100,soc_100_200,phh2o_0_5,phh2o_5_15,phh2o_15_30,phh2o_30_60,phh2o_60_100,phh2o_100_200,"
    "nitrogen_0_5,nitrogen_5_15,nitrogen_15_30,nitrogen_30_60,nitrogen_60_100,nitrogen_100_200,texture_class_0_5"
)
AGE_HEADER = "start_date,end_date,lat,lon,site_name,stand_age_TC000,stand_age_TC010,stand_age_TC020,stand_age_TC030"
FAN_HEADER = (
    "site_name,root_depth,infiltration_depth,deep_drainage,drainage_frequency,regolith_flush_rate,gw_residence_time,"
    "root_depth_mask_valid,root_depth_fallback_px,root_depth_continent,root_depth_lat_pixel,root_depth_lon_pixel"
)


def _context(tmp_path):
    sites = pd.DataFrame(
        {
            "site_name": ["A", "B"],
            "lat": [50.5, -30.25],
            "lon": [10.5, 140.75],
            "start_date": ["2010-01-01", "2011-01-01"],
            "end_date": ["2012-01-01", "2013-01-01"],
            "si_igbp": ["ENF", "EBF"],
            "row": [1, 2],
            "col": [3, 4],
        }
    )
    return RunContext(sites=sites, raw_root=tmp_path, output_root=tmp_path / "static_1km" / "scale", project="p")


def _write_part(context, product, columns=None, site_hash=None, order=None, **replacements):
    columns = COLUMNS[product] if columns is None else columns
    index = pd.Index(context.sites["site_name"] if order is None else order, name="site_name")
    values = pd.DataFrame(1.0, index=index, columns=columns)
    for column in TEXT_COLUMNS.intersection(columns):
        values[column] = "text"
    values = values.assign(**replacements)
    status = pd.DataFrame(np.where(values.notna(), OK, NO_DATA), index=index, columns=columns)
    by_year = pd.DataFrame(
        [(index[0], 2010, columns[0], 1.0, OK)],
        columns=list(BY_YEAR_COLUMNS),
    )
    result = ProductResult(
        product=product,
        values=values,
        status=status,
        coverage=pd.DataFrame({columns[0]: 0.5}, index=index),
        by_year=by_year,
        extras={f"{product}_extra": pd.DataFrame({"x": [1]})} if product == "evi" else {},
        provenance={
            "source": f"{product} url",
            "version": "v1",
            "checksum": {"f": "abc"},
            "time_window": "2010",
            "method": "m",
        },
    )
    write_part(result, context.parts_root, site_list_hash(context.sites) if site_hash is None else site_hash)


def _all_parts(context):
    for product in PRODUCTS:
        _write_part(context, product)


def _header(path):
    return path.read_text(encoding="utf-8").splitlines()[0]


@pytest.fixture
def built(tmp_path):
    context = _context(tmp_path)
    _all_parts(context)
    return context, assemble(context, git_commit="abc123")


def test_env_file_has_the_exact_header(built):
    _, output = built
    assert _header(output / "site_info_with_env_data.csv") == ENV_HEADER


def test_soil_file_has_the_exact_header(built):
    _, output = built
    assert _header(output / "soilgrids_data.csv") == SOIL_HEADER


def test_stand_age_file_holds_only_four_stand_age_columns(built):
    _, output = built  # the merge averages any column whose name holds stand_age_TC, so nothing else may
    assert _header(output / "stand_age_data.csv") == AGE_HEADER


def test_fan_file_has_the_exact_header(built):
    _, output = built
    assert _header(output / "site_info_with_root_depth_fan2017.csv") == FAN_HEADER


def test_merge_files_carry_the_site_columns(built):
    context, output = built
    table = pd.read_csv(output / "stand_age_data.csv")

    assert table["site_name"].tolist() == ["A", "B"]
    assert table["lat"].tolist() == [50.5, -30.25]
    assert table["end_date"].tolist() == ["2012-01-01", "2013-01-01"]


def test_soil_file_says_where_it_came_from(built):
    _, output = built
    table = pd.read_csv(output / "soilgrids_data.csv")

    assert set(table["data_source"]) == {"earth_engine_1km"}
    assert set(table["offset_distance_m"]) == {0}
    assert table["snapped_lat"].tolist() == table["lat"].tolist()
    assert table["snapped_lon"].tolist() == table["lon"].tolist()


def test_env_file_reports_the_aspect_vector_as_a_bearing(tmp_path):
    context = _context(tmp_path)
    _all_parts(context)
    _write_part(context, "terrain", aspect_sin=[1.0, 0.0], aspect_cos=[0.0, 0.0])
    table = pd.read_csv(
        assemble(context) / "site_info_with_env_data.csv", keep_default_na=False, na_values=[""]
    ).set_index("site_name")

    assert table.loc["A", "aspect_deg"] == 90.0
    assert table.loc["A", "aspect_standardized"] == 90.0
    assert table.loc["A", "aspect_category"] == "E"
    assert np.isnan(table.loc["B", "aspect_deg"])
    assert np.isnan(table.loc["B", "aspect_standardized"])
    assert np.isnan(table.loc["B", "aspect_category"])
    assert table["slope_deg_centre"].isna().all()


def test_rows_follow_the_site_list_whatever_order_a_part_has(tmp_path):
    context = _context(tmp_path)
    _all_parts(context)
    _write_part(context, "stand_age", order=["B", "A"], stand_age_TC000=[7.0, 3.0])
    table = pd.read_csv(assemble(context) / "stand_age_data.csv")

    assert table["site_name"].tolist() == ["A", "B"]
    assert table["stand_age_TC000"].tolist() == [3.0, 7.0]


def test_new_static_features_has_no_status_columns(built):
    _, output = built
    columns = _header(output / "new_static_features.csv").split(",")

    assert columns == ["site_name", *NEW_COLUMNS]
    assert not [column for column in columns if column.endswith(("_status", "_coverage"))]


def test_quality_file_has_a_status_for_every_value_column(built):
    _, output = built
    table = pd.read_csv(output / "static_quality.csv")
    every_column = [column for columns in COLUMNS.values() for column in columns]

    assert table["site_name"].tolist() == ["A", "B"]
    assert not [column for column in every_column if f"{column}_status" not in table.columns]
    assert set(table["elevation_m_status"]) == {OK}


def test_quality_file_has_coverage_only_where_measured(built):
    _, output = built
    table = pd.read_csv(output / "static_quality.csv")

    assert table["elevation_m_coverage"].tolist() == [0.5, 0.5]
    assert "slope_deg_coverage" not in table.columns


def test_by_year_concatenates_every_part(built):
    _, output = built
    table = pd.read_csv(output / "static_features_by_year.csv")

    assert list(table.columns) == list(BY_YEAR_COLUMNS)
    assert len(table) == len(PRODUCTS)
    assert "forest_age_gami" in set(table["column"])


def test_extras_become_files_at_the_root(built):
    _, output = built
    assert pd.read_csv(output / "evi_extra.csv")["x"].tolist() == [1]


def test_every_table_has_one_row_per_site(built):
    _, output = built
    for name in (
        "site_info_with_env_data.csv",
        "soilgrids_data.csv",
        "stand_age_data.csv",
        "site_info_with_root_depth_fan2017.csv",
        "new_static_features.csv",
        "static_quality.csv",
    ):
        assert len(pd.read_csv(output / name)) == 2, name


def test_assemble_refuses_part_from_another_site_list(tmp_path):
    context = _context(tmp_path)
    _all_parts(context)
    _write_part(context, "yang", site_hash="other")
    with pytest.raises(ExtractionFailure, match="yang"):
        assemble(context)


def test_assemble_refuses_a_missing_product(tmp_path):
    context = _context(tmp_path)
    for product in PRODUCTS:
        if product != "gami":
            _write_part(context, product)
    with pytest.raises(ExtractionFailure, match="gami"):
        assemble(context)


def test_assemble_refuses_a_column_from_two_products(tmp_path):
    context = _context(tmp_path)
    _all_parts(context)
    _write_part(context, "han", columns=["root_depth_han", "wood_density"])
    with pytest.raises(ExtractionFailure, match=r"wood_density.*han.*yang|wood_density.*yang.*han"):
        assemble(context)


def test_assemble_refuses_a_part_without_a_column_the_merge_needs(tmp_path):
    context = _context(tmp_path)
    _all_parts(context)
    _write_part(context, "terrain", columns=["slope_deg", "aspect_sin", "aspect_cos", "land_share"])
    with pytest.raises(ExtractionFailure, match="elevation_m"):
        assemble(context)


def test_assemble_refuses_a_part_whose_numbers_and_codes_disagree(tmp_path):
    context = _context(tmp_path)
    _all_parts(context)
    directory = context.parts_root / "yang"
    status = pd.read_csv(directory / "status.csv")
    status["wood_density"] = NO_DATA
    status.to_csv(directory / "status.csv", index=False)
    with pytest.raises(ExtractionFailure, match="yang"):
        assemble(context)


def test_assemble_failure_leaves_previous_output_intact(tmp_path):
    context = _context(tmp_path)
    _all_parts(context)
    output = assemble(context, git_commit="first")
    before = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}

    _write_part(context, "yang", site_hash="other")
    with pytest.raises(ExtractionFailure):
        assemble(context, git_commit="second")

    after = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    assert after == before
    assert not list(output.glob(".tmp_*"))


def test_assemble_leaves_no_temp_directory_and_keeps_parts(built):
    context, output = built
    assert not list(output.glob(".tmp_*"))
    assert (context.parts_root / "gami" / "values.csv").exists()


def test_assemble_clears_a_temp_directory_left_by_a_killed_job(tmp_path):
    context = _context(tmp_path)
    _all_parts(context)
    stale = context.output_root / ".tmp_assemble_99999"
    stale.mkdir(parents=True)
    (stale / "half.csv").write_text("x", encoding="utf-8")

    assemble(context)
    assert not stale.exists()


def test_aspect_report_hand_computed():
    assert aspect_report(1.0, 0.0) == (90.0, "E")
    assert aspect_report(0.0, 1.0) == (0.0, "N")
    assert aspect_report(0.0, -1.0) == (180.0, "S")
    assert aspect_report(-1.0, 0.0) == (270.0, "W")
    assert aspect_report(0.0, 0.0) == (None, None)
    assert aspect_report(np.nan, 0.5) == (None, None)


def test_aspect_report_keeps_a_short_vector_and_splits_sectors_on_the_half():
    bearing, sector = aspect_report(0.001, 0.001)
    assert bearing == pytest.approx(45.0)
    assert sector == "NE"
    assert aspect_report(np.sin(np.radians(337.6)), np.cos(np.radians(337.6)))[1] == "N"
    assert aspect_report(np.sin(np.radians(22.4)), np.cos(np.radians(22.4)))[1] == "N"
    assert aspect_report(np.sin(np.radians(22.6)), np.cos(np.radians(22.6)))[1] == "NE"


def test_provenance_records_source_checksum_window_and_commit(built):
    context, output = built
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))

    assert provenance["code_commit"] == "abc123"
    assert provenance["site_list_hash"] == site_list_hash(context.sites)
    assert set(provenance["products"]) == set(PRODUCTS)
    for key, expected in (
        ("source", "gami url"),
        ("version", "v1"),
        ("checksum", {"f": "abc"}),
        ("time_window", "2010"),
    ):
        assert provenance["products"]["gami"][key] == expected
    assert "slope_deg_centre" in provenance["notes"]


def test_main_writes_the_files_and_returns_zero(tmp_path, monkeypatch):
    context = _context(tmp_path)
    _all_parts(context)
    monkeypatch.setattr(assemble_module.RunContext, "for_scale", classmethod(lambda cls, scale, project: context))

    assert main(["--scale", "scale"]) == 0
    assert (context.output_root / "provenance.json").exists()


def test_main_returns_nonzero_on_a_failure(tmp_path, monkeypatch):
    context = _context(tmp_path)
    monkeypatch.setattr(assemble_module.RunContext, "for_scale", classmethod(lambda cls, scale, project: context))

    assert main(["--scale", "scale"]) == 1


def test_lusk_quality_bands_stay_out_of_the_merge_in_their_own_file(built):
    _, output = built

    assert _header(output / "lusk_quality.csv").split(",") == [
        "site_name",
        "wood_density_lusk_cov",
        "wood_density_lusk_aoa",
    ]
    assert not [
        column
        for column in _header(output / "new_static_features.csv").split(",")
        if column.endswith(("_cov", "_aoa"))
    ]


def test_provenance_records_the_coordinate_corrections(built):
    _context, output = built
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    corrected = provenance["coordinate_corrections"]["GBR_DEV_CON"]

    assert (corrected["published_lat"], corrected["lat"]) == (56.03333, 56.0792)
    assert corrected["reason"]


def test_a_job_killed_while_swapping_files_leaves_no_provenance(tmp_path, monkeypatch):
    context = _context(tmp_path)
    _all_parts(context)
    output = assemble(context, git_commit="first")
    real_replace = assemble_module.os.replace
    swapped = []

    def dies_on_the_third(source, target):
        if len(swapped) == 2:
            raise OSError("killed")
        swapped.append(target)
        real_replace(source, target)

    monkeypatch.setattr(assemble_module.os, "replace", dies_on_the_third)
    with pytest.raises(OSError, match="killed"):
        assemble(context, git_commit="second")

    assert not (output / "provenance.json").exists()


def test_provenance_is_the_last_file_put_in_place(tmp_path, monkeypatch):
    context = _context(tmp_path)
    _all_parts(context)
    real_replace = assemble_module.os.replace
    order = []

    def recording(source, target):
        order.append(str(target).replace("\\", "/").rsplit("/", 1)[-1])
        real_replace(source, target)

    monkeypatch.setattr(assemble_module.os, "replace", recording)
    assemble(context, git_commit="abc")

    assert order[-1] == "provenance.json"
    assert order.count("provenance.json") == 1

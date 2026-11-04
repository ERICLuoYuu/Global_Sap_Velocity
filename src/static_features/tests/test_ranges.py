from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.static_features.checks import ranges
from src.static_features.checks.ranges import RANGES, coverage_report, low_land_sites, main, out_of_range
from src.static_features.results import RunContext
from src.static_features.run import PRODUCTS


def _values(columns, names=("A", "B")):
    return pd.DataFrame(columns, index=pd.Index(list(names), name="site_name"), dtype=float)


def test_out_of_range_names_site_and_column():
    table = out_of_range(_values({"slope_deg": [10.0, 95.0]}))

    assert table["site_name"].tolist() == ["B"]
    assert table["column"].tolist() == ["slope_deg"]
    assert table["value"].tolist() == [95.0]
    assert table["low"].tolist() == [0.0]
    assert table["high"].tolist() == [90.0]


def test_nan_is_not_out_of_range():
    assert out_of_range(_values({"slope_deg": [np.nan, 10.0]})).empty


def test_range_bounds_are_inside():
    assert out_of_range(_values({"slope_deg": [0.0, 90.0]})).empty


def test_column_without_a_range_is_ignored():
    assert out_of_range(_values({"bio1": [-999.0, 999.0]})).empty


def test_a_table_with_no_problem_still_has_the_columns():
    assert list(out_of_range(_values({"slope_deg": [1.0, 2.0]})).columns) == [
        "site_name",
        "column",
        "value",
        "low",
        "high",
    ]


@pytest.mark.parametrize(
    ("column", "low", "high"),
    [
        ("elevation_m", -450, 9000),
        ("aspect_cos", -1, 1),
        ("canopy_height_m", 0, 60),
        ("canopy_height_p95_m", 0, 100),
        ("land_share", 0, 1),
        ("fm_share_53", 0, 1),
        ("phen_cycle_share", 0, 1),
        ("clay_0_5", 0, 100),
        ("cfvo_100_200", 0, 100),
        ("bdod_30_60", 0, 2.2),
        ("soc_5_15", 0, 1000),
        ("phh2o_60_100", 3, 11),
        ("nitrogen_15_30", 0, 100),
        ("root_depth_han", 0, 100),
        ("wood_density", 0.1, 1.5),
        ("tree_richness_per_ha", 1, 250),
        ("agb_mg_ha", 0, 1500),
        ("agb_sd", 0, 1500),
        ("forest_age_gami", 0, 300),
        ("tree_cover_pct", 0, 100),
        ("evi_mean", -0.2, 1),
        ("evi_amplitude", -0.2, 1),
        ("evi_season_amplitude", 0, 1),
        ("evi_season_r2", 0, 1),
        ("evi_peak_cos_w", -1, 1),
        ("evi_peak_sin_w", -1, 1),
        ("phen_evi2_min", -0.2, 1),
        ("phen_evi2_amplitude", 0, 1),
        ("phen_midgreendown_sin", -1, 1),
        ("phen_greenup_cos", -1, 1),
        ("phen_season_days", 0, 730),
        ("phen_midseason_days", 0, 730),
        ("phen_num_cycles", 0, 2),
        ("phen_years_used", 0, 20),
        ("evi_years_used", 0, 20),
        ("root_depth_lusk", 0, 100),
        ("wood_density_lusk", 0.1, 1.5),
    ],
)
def test_ranges_hold_the_pinned_bounds(column, low, high):
    assert RANGES[column] == (low, high)


def test_low_land_sites_lists_below_half():
    assert low_land_sites(_values({"land_share": [0.3, 0.5]})) == ["A"]


def test_low_land_sites_ignores_a_missing_share():
    assert low_land_sites(_values({"land_share": [np.nan, 0.9]})) == []


def test_shares_summing_above_one_are_reported():
    table = out_of_range(_values({"fm_share_11": [0.7, 0.2], "fm_share_20": [0.5, 0.3]}))

    assert table["site_name"].tolist() == ["A"]
    assert table["column"].tolist() == ["fm_share_sum"]
    assert table["value"].tolist() == [pytest.approx(1.2)]


def test_shares_summing_to_one_through_rounding_pass():
    assert out_of_range(
        _values({"fm_share_11": [0.1, 0.0], "fm_share_20": [0.2, 0.0], "fm_share_31": [0.7, 1.0]})
    ).empty


def test_coverage_report_counts_per_product_and_reason():
    quality = pd.DataFrame(
        {
            "site_name": ["A", "B", "C"],
            "root_depth_han_status": ["ok", "neighbour", "no_data"],
            "root_depth_han_uncertainty_status": ["ok", "no_data", "no_data"],
            "wood_density_status": ["ok", "ok", "ok"],
            "wood_density_coverage": [1.0, 1.0, 1.0],
        }
    )
    products = {"root_depth_han": "han", "root_depth_han_uncertainty": "han", "wood_density": "yang"}
    table = coverage_report(quality, products).set_index(["product", "reason"])

    assert table.loc[("han", "ok"), "columns"] == 2
    assert table.loc[("han", "ok"), "sites_min"] == 1
    assert table.loc[("han", "no_data"), "sites_min"] == 1
    assert table.loc[("han", "no_data"), "sites_max"] == 2
    assert table.loc[("yang", "ok"), "sites_max"] == 3
    assert ("yang", "no_data") not in table.index


def _files(tmp_path, slope=10.0):
    output = tmp_path / "static_1km" / "scale"
    checks_free = output / "parts"
    for product in PRODUCTS:
        directory = checks_free / product
        directory.mkdir(parents=True)
        pd.DataFrame({"site_name": ["A"], f"{product}_x": ["ok"]}).to_csv(directory / "status.csv", index=False)
    index = {"site_name": ["A", "B"]}
    pd.DataFrame({**index, "lat": [1.0, 2.0], "slope_deg": [slope, 5.0], "elevation_m": [10.0, 20.0]}).to_csv(
        output / "site_info_with_env_data.csv", index=False
    )
    pd.DataFrame({**index, "lat": [1.0, 2.0], "clay_0_5": [10.0, 20.0]}).to_csv(
        output / "soilgrids_data.csv", index=False
    )
    pd.DataFrame({**index, "land_share": [0.3, 1.0], "fm_share_11": [0.2, 0.0]}).to_csv(
        output / "new_static_features.csv", index=False
    )
    pd.DataFrame({**index, "terrain_x_status": ["ok", "no_data"]}).to_csv(output / "static_quality.csv", index=False)
    return RunContext(sites=pd.DataFrame(index), raw_root=tmp_path, output_root=output, project="p")


def _run(context, monkeypatch):
    monkeypatch.setattr(ranges.RunContext, "for_scale", classmethod(lambda cls, scale, project: context))
    return main(["--scale", "scale"])


def test_main_exits_zero_and_writes_the_reports_when_clean(tmp_path, monkeypatch):
    context = _files(tmp_path)

    assert _run(context, monkeypatch) == 0
    checks = context.output_root / "checks"
    assert pd.read_csv(checks / "out_of_range.csv").empty
    assert pd.read_csv(checks / "low_land_sites.csv")["site_name"].tolist() == ["A"]
    assert set(pd.read_csv(checks / "coverage_report.csv")["product"]) == {"terrain"}


def test_main_exits_nonzero_and_lists_each_exception(tmp_path, monkeypatch, caplog):
    context = _files(tmp_path, slope=95.0)

    assert _run(context, monkeypatch) == 1
    assert pd.read_csv(context.output_root / "checks" / "out_of_range.csv")["site_name"].tolist() == ["A"]
    assert "slope_deg" in caplog.text and "A" in caplog.text


def test_main_exits_nonzero_without_an_assembled_output(tmp_path, monkeypatch):
    context = _files(tmp_path)
    (context.output_root / "static_quality.csv").unlink()

    assert _run(context, monkeypatch) == 1

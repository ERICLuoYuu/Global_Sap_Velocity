from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from src.static_features.products.phenology import main_cycle, summarise, with_land
from src.static_features.results import NO_DATA, OK, ExtractionFailure, validate

FILL = -9999
CYCLE_BANDS = (
    "Greenup",
    "MidGreenup",
    "Maturity",
    "Peak",
    "Senescence",
    "MidGreendown",
    "Dormancy",
    "EVI_Minimum",
    "EVI_Amplitude",
    "EVI_Area",
    "QA_Overall",
)
A_CYCLE = {
    "Greenup_1": 14640,
    "MidGreenup_1": 14660,
    "Maturity_1": 14680,
    "Peak_1": 14700,
    "Senescence_1": 14720,
    "MidGreendown_1": 14740,
    "Dormancy_1": 14760,
    "EVI_Minimum_1": 1000,
    "EVI_Amplitude_1": 3000,
    "EVI_Area_1": 300,
    "QA_Overall_1": 0,
}
DATE_COLUMNS = [
    f"phen_{name}_{part}"
    for name in ("greenup", "midgreenup", "maturity", "peak", "senescence", "midgreendown", "dormancy")
    for part in ("sin", "cos")
]
CYCLE_COLUMNS = [
    *DATE_COLUMNS,
    "phen_season_days",
    "phen_midseason_days",
    "phen_evi2_min",
    "phen_evi2_amplitude",
    "phen_evi2_area",
]
ALWAYS_COLUMNS = ["phen_num_cycles", "phen_cycle_share", "phen_years_used"]


def _no_cycle(site="A", year=2010, land=1):
    row = {"site_name": site, "year": year, "land": land, "NumCycles": FILL}
    for band in CYCLE_BANDS:
        row[f"{band}_1"] = FILL
        row[f"{band}_2"] = FILL
    return row


def _cycle(site="A", year=2010, land=1, num_cycles=1, **bands):
    return {**_no_cycle(site, year, land), "NumCycles": num_cycles, **A_CYCLE, **bands}


def _summary(rows, **kwargs):
    return summarise(pd.DataFrame(rows), **kwargs)


def test_peak_dates_across_new_year_average_to_january():
    result = _summary([_cycle(Peak_1=14610), _cycle(Peak_1=14974)])  # 2010-01-01 and 2010-12-31

    assert result.values.loc["A", "phen_peak_cos"] > 0.99
    assert abs(result.values.loc["A", "phen_peak_sin"]) < 0.02


def test_angle_is_day_of_year_over_the_mean_year():
    result = _summary([_cycle(Peak_1=14610)])  # day of year 1
    assert result.values.loc["A", "phen_peak_sin"] == pytest.approx(math.sin(2 * math.pi / 365.25))
    assert result.values.loc["A", "phen_peak_cos"] == pytest.approx(math.cos(2 * math.pi / 365.25))


def test_main_cycle_is_the_larger_not_the_first():
    row = _cycle(num_cycles=2, EVI_Amplitude_1=1000, Peak_1=14700, EVI_Amplitude_2=3000, Peak_2=14800)
    picked = main_cycle(pd.DataFrame([row]))

    assert picked.loc[0, "Peak"] == 14800
    assert picked.loc[0, "EVI_Amplitude"] == 3000


def test_main_cycle_tie_takes_the_first():
    row = _cycle(num_cycles=2, EVI_Amplitude_1=3000, Peak_1=14700, EVI_Amplitude_2=3000, Peak_2=14800)
    assert main_cycle(pd.DataFrame([row])).loc[0, "Peak"] == 14700


def test_main_cycle_of_one_cycle_ignores_the_empty_second():
    assert main_cycle(pd.DataFrame([_cycle()])).loc[0, "Peak"] == 14700


def test_main_cycle_turns_fill_into_missing():
    picked = main_cycle(pd.DataFrame([_cycle(Greenup_1=FILL), _no_cycle()]))

    assert np.isnan(picked.loc[0, "Greenup"])
    assert picked.loc[0, "Peak"] == 14700
    assert np.isnan(picked.loc[1, "Peak"])
    assert picked["has_cycle"].tolist() == [True, False]
    assert picked["num_cycles"].tolist() == [1, 0]


def test_season_days_per_pixel_before_averaging():
    result = _summary([_cycle(Greenup_1=14600, Dormancy_1=14700), _cycle(Greenup_1=14600, Dormancy_1=14900)])
    assert result.values.loc["A", "phen_season_days"] == 200.0


def test_season_days_skip_a_pixel_missing_one_date():
    # Averaging the dates first would give 14700 - 14605 = 95.
    result = _summary([_cycle(Greenup_1=14600, Dormancy_1=14700), _cycle(Greenup_1=14610, Dormancy_1=FILL)])
    assert result.values.loc["A", "phen_season_days"] == 100.0


def test_midseason_days_use_the_midpoints():
    result = _summary([_cycle(MidGreenup_1=14650, MidGreendown_1=14730)])
    assert result.values.loc["A", "phen_midseason_days"] == 80.0


def test_averages_over_pixels_then_over_years():
    rows = [
        _cycle(year=2010, Greenup_1=14600, Dormancy_1=14700),
        _cycle(year=2010, Greenup_1=14600, Dormancy_1=14800),
        _cycle(year=2011, Greenup_1=15000, Dormancy_1=15300),
    ]
    # Years: 150 and 300. Pooling the three pixel-years would give 200.
    assert _summary(rows).values.loc["A", "phen_season_days"] == 225.0


def test_no_cycle_counts_as_zero_for_share_and_num_cycles():
    rows = [_cycle(num_cycles=1), _no_cycle(), _no_cycle(year=2011), _no_cycle(year=2011)]
    result = _summary(rows)

    assert result.values.loc["A", "phen_cycle_share"] == pytest.approx(0.25)
    assert result.values.loc["A", "phen_num_cycles"] == pytest.approx(0.25)


def test_num_cycles_counts_two_cycle_pixels_as_two():
    assert _summary([_cycle(num_cycles=2), _no_cycle()]).values.loc["A", "phen_num_cycles"] == 1.0


def test_water_pixels_are_ignored_everywhere():
    rows = [_cycle(), _cycle(land=0, Peak_1=14900, Greenup_1=14000, EVI_Minimum_1=9000), _no_cycle(land=0)]
    result = _summary(rows)
    alone = _summary([_cycle()])

    assert result.values.loc["A", "phen_cycle_share"] == 1.0
    assert result.values.loc["A", "phen_years_used"] == 1
    assert result.values.loc["A", "phen_season_days"] == alone.values.loc["A", "phen_season_days"]
    assert result.values.loc["A", "phen_evi2_min"] == alone.values.loc["A", "phen_evi2_min"]
    assert result.values.loc["A", "phen_peak_sin"] == alone.values.loc["A", "phen_peak_sin"]


def test_no_land_is_no_data_for_every_column():
    result = _summary([_cycle(land=0), _no_cycle(land=0)])

    assert result.status.loc["A"].eq(NO_DATA).all()
    assert result.values.loc["A"].isna().all()
    assert result.coverage.loc["A", "phen_cycle_share"] == 0.0
    validate(result, ["A"])


def test_land_without_cycle_keeps_share_zero_and_dates_no_data():
    result = _summary([_no_cycle(), _no_cycle(year=2011)])

    for column in CYCLE_COLUMNS:
        assert np.isnan(result.values.loc["A", column]), column
        assert result.status.loc["A", column] == NO_DATA, column
    for column in ALWAYS_COLUMNS:
        assert result.values.loc["A", column] == 0, column
        assert result.status.loc["A", column] == OK, column
    validate(result, ["A"])


def test_every_column_is_present_in_order():
    assert list(_summary([_cycle()]).values.columns) == CYCLE_COLUMNS + ALWAYS_COLUMNS


def test_evi2_scaling():
    result = _summary([_cycle(EVI_Minimum_1=1000, EVI_Amplitude_1=3000, EVI_Area_1=300)])

    assert result.values.loc["A", "phen_evi2_min"] == pytest.approx(0.1)
    assert result.values.loc["A", "phen_evi2_amplitude"] == pytest.approx(0.3)
    assert result.values.loc["A", "phen_evi2_area"] == pytest.approx(30.0)


def test_years_used_counts_years_with_a_cycle():
    rows = [_cycle(year=2010), _no_cycle(year=2010), _cycle(year=2011), _no_cycle(year=2012)]
    assert _summary(rows).values.loc["A", "phen_years_used"] == 2


def test_years_outside_window_reach_only_by_year():
    rows = [_cycle(year=2010, Peak_1=14700), _cycle(year=2021, Peak_1=18800, EVI_Minimum_1=9000)]
    result = _summary(rows)

    assert result.values.loc["A", "phen_years_used"] == 1
    assert result.values.loc["A", "phen_evi2_min"] == pytest.approx(0.1)
    table = result.by_year[result.by_year["column"] == "phen_evi2_min"].set_index("year")
    assert table.loc[2021, "value"] == pytest.approx(0.9)
    assert table.loc[2010, "status"] == OK
    validate(result, ["A"])


def test_by_year_holds_share_at_zero_and_empty_dates_for_a_year_without_cycle():
    result = _summary([_cycle(year=2010), _no_cycle(year=2011)])
    table = result.by_year.set_index(["column", "year"])

    assert table.loc[("phen_cycle_share", 2011), "value"] == 0.0
    assert table.loc[("phen_cycle_share", 2011), "status"] == OK
    assert np.isnan(table.loc[("phen_peak_sin", 2011), "value"])
    assert table.loc[("phen_peak_sin", 2011), "status"] == NO_DATA
    assert "phen_years_used" not in set(result.by_year["column"])
    validate(result, ["A"])


def test_coverage_counts_pixel_years_holding_the_value_over_all_pixel_years():
    rows = [_cycle(), _no_cycle(), _no_cycle(land=0), _cycle(Greenup_1=FILL)]
    result = _summary(rows)

    assert result.coverage.loc["A", "phen_peak_sin"] == 0.5
    assert result.coverage.loc["A", "phen_greenup_sin"] == 0.25
    assert result.coverage.loc["A", "phen_cycle_share"] == 0.75  # land share of the cell


def test_extras_keep_the_sampled_table_and_provenance_carries_the_guides_warning():
    table = pd.DataFrame([_cycle(), _no_cycle()])
    result = summarise(table)

    assert result.extras["phenology_pixels"].equals(table)
    assert "high latitude" in result.provenance["caution"]
    assert result.provenance["time_window"] == "2001-2020"


def test_sites_stay_apart_and_in_order():
    result = _summary([_no_cycle(site="B"), _cycle(site="A")])

    assert list(result.values.index) == ["B", "A"]
    assert result.values.loc["A", "phen_cycle_share"] == 1.0
    assert result.values.loc["B", "phen_cycle_share"] == 0.0
    validate(result, ["A", "B"])


def _land_table(rows):
    return pd.DataFrame(rows, columns=["site_name", "longitude", "latitude", "land"])


def _year_table(rows):
    return pd.DataFrame(rows, columns=["site_name", "longitude", "latitude", "NumCycles"])


def test_land_joins_by_pixel_not_by_row_order():
    pixels = _year_table([("A", 1.0, 2.0, 1), ("A", 1.5, 2.0, 2), ("B", 9.0, 9.0, 3)])
    land = _land_table([("B", 9.0, 9.0, 0), ("A", 1.5, 2.0, 1), ("A", 1.0, 2.0, 0)])

    joined = with_land(pixels, land)

    assert joined["land"].tolist() == [0, 1, 0]
    assert joined["NumCycles"].tolist() == [1, 2, 3]


def test_land_join_tolerates_coordinate_noise_far_below_a_pixel():
    pixels = _year_table([("A", 1.00000001, 2.0, 1)])
    land = _land_table([("A", 1.00000002, 2.0, 1)])

    assert with_land(pixels, land)["land"].tolist() == [1]


def test_a_pixel_without_land_stops_naming_its_site():
    pixels = _year_table([("A", 1.0, 2.0, 1), ("B", 9.0, 9.0, 1)])
    land = _land_table([("A", 1.0, 2.0, 1)])

    with pytest.raises(ExtractionFailure, match="B"):
        with_land(pixels, land)


def test_a_land_pixel_absent_from_the_year_stops_naming_its_site():
    pixels = _year_table([("A", 1.0, 2.0, 1)])
    land = _land_table([("A", 1.0, 2.0, 1), ("C", 5.0, 5.0, 1)])

    with pytest.raises(ExtractionFailure, match="C"):
        with_land(pixels, land)


def test_sites_sharing_a_cell_keep_their_own_land_rows():
    pixels = _year_table([("A", 1.0, 2.0, 1), ("B", 1.0, 2.0, 1)])
    land = _land_table([("A", 1.0, 2.0, 1), ("B", 1.0, 2.0, 0)])

    assert with_land(pixels, land)["land"].tolist() == [1, 0]

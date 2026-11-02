from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.static_features.assemble import SOIL_VALUE_COLUMNS
from src.static_features.checks import compare
from src.static_features.checks.compare import column_report, main
from src.static_features.results import RunContext


def _series(values, names="ABCD"):
    return pd.Series(values, index=pd.Index(list(names), name="site_name"), dtype=float)


def test_column_report_names_largest_movers():
    report = column_report(_series([1.0, 2.0, 3.0, 4.0]), _series([1.0, 2.5, 1.0, 4.0]), top=2)

    assert report["n"] == 4
    assert [mover["site_name"] for mover in report["largest_movers"]] == ["C", "B"]
    assert report["largest_movers"][0]["old"] == 3.0
    assert report["largest_movers"][0]["new"] == 1.0
    assert report["largest_movers"][0]["change"] == -2.0


def test_column_report_r_hand_computed():
    report = column_report(_series([1.0, 2.0, 3.0], "ABC"), _series([2.0, 4.0, 6.0], "ABC"))
    assert report["r"] == pytest.approx(1.0)


def test_column_report_keeps_only_sites_holding_both_values():
    report = column_report(_series([1.0, np.nan, 3.0, 4.0]), _series([1.0, 2.0, np.nan, 5.0]))

    assert report["n"] == 2
    assert {mover["site_name"] for mover in report["largest_movers"]} == {"A", "D"}


def test_column_report_aligns_on_the_site_not_the_position():
    report = column_report(_series([1.0, 2.0], "AB"), _series([2.0, 1.0], "BA"))

    assert report["n"] == 2
    assert all(mover["change"] == 0.0 for mover in report["largest_movers"])


def test_column_report_constant_series_gives_nan_r_not_a_crash():
    report = column_report(_series([1.0, 1.0, 1.0], "ABC"), _series([1.0, 2.0, 3.0], "ABC"))

    assert report["n"] == 3
    assert np.isnan(report["r"])


def test_column_report_of_no_shared_site_is_empty_not_a_crash():
    report = column_report(_series([np.nan], "A"), _series([1.0], "A"))

    assert report["n"] == 0
    assert np.isnan(report["r"])
    assert report["largest_movers"] == []


def _files(tmp_path):
    root = tmp_path / "data" / "raw" / "extracted_data"
    output = root / "static_1km" / "scale"
    output.mkdir(parents=True)
    (root / "env_site_data" / "scale").mkdir(parents=True)
    (root / "terrain_site_data" / "scale").mkdir(parents=True)
    env = pd.DataFrame(
        {
            "site_name": ["A", "B", "C"],
            "elevation_m": [100.0, 200.0, 300.0],
            "slope_deg": [1.0, 2.0, 3.0],
            "aspect_sin": [0.1, 0.2, 0.3],
            "aspect_cos": [0.3, 0.2, 0.1],
            "canopy_height_m": [10.0, 20.0, 30.0],
        }
    )
    soil = pd.DataFrame({"site_name": ["A", "B", "C"], **{column: [1.0, 2.0, 3.0] for column in SOIL_VALUE_COLUMNS}})
    env.to_csv(output / "site_info_with_env_data.csv", index=False)
    soil.to_csv(output / "soilgrids_data.csv", index=False)
    env.assign(elevation_m=[100.0, 250.0, 300.0]).to_csv(
        root / "env_site_data" / "scale" / "site_info_with_env_data.csv", index=False
    )
    soil.to_csv(root / "terrain_site_data" / "scale" / "soilgrids_data.csv", index=False)
    probe = tmp_path / "probes" / "canopy_height_ground"
    probe.mkdir(parents=True)
    pd.DataFrame({"site_name": ["A", "B", "C"], "metasq1km_h": [11.0, 21.0, 29.0]}).to_csv(
        probe / "canopy_stats_sites.csv", index=False
    )
    sites = pd.DataFrame({"site_name": ["A", "B", "C"]})
    return RunContext(sites=sites, raw_root=tmp_path, output_root=output, project="p")


def _run(context, monkeypatch):
    monkeypatch.setattr(compare.RunContext, "for_scale", classmethod(lambda cls, scale, project: context))
    return main(["--scale", "scale"])


def test_main_writes_old_against_new_for_each_column(tmp_path, monkeypatch):
    context = _files(tmp_path)

    assert _run(context, monkeypatch) == 0
    summary = pd.read_csv(context.output_root / "checks" / "compare_summary.csv")
    columns = set(summary["column"])
    assert {"elevation_m", "slope_deg", "aspect_sin", "aspect_cos", "canopy_height_m"} <= columns
    assert set(SOIL_VALUE_COLUMNS) <= columns
    movers = pd.read_csv(context.output_root / "checks" / "compare_movers.csv")
    top = movers[(movers["column"] == "elevation_m")].iloc[0]
    assert (top["site_name"], top["old"], top["new"]) == ("B", 250.0, 200.0)


def test_main_compares_canopy_with_the_earlier_probe(tmp_path, monkeypatch):
    context = _files(tmp_path)

    assert _run(context, monkeypatch) == 0
    summary = pd.read_csv(context.output_root / "checks" / "compare_summary.csv")
    probe = summary[summary["against"].str.contains("metasq1km_h")]
    assert probe["column"].tolist() == ["canopy_height_m"]
    assert probe["n"].tolist() == [3]


def test_main_returns_nonzero_when_a_current_file_is_absent(tmp_path, monkeypatch):
    context = _files(tmp_path)
    (context.output_root.parents[1] / "terrain_site_data" / "scale" / "soilgrids_data.csv").unlink()

    assert _run(context, monkeypatch) == 1


def test_main_reads_the_probe_from_the_given_root(tmp_path, monkeypatch):
    context = _files(tmp_path)
    code = tmp_path / "code"
    code.mkdir()
    (tmp_path / "probes").rename(code / "probes")
    monkeypatch.setattr(compare.RunContext, "for_scale", classmethod(lambda cls, scale, project: context))

    assert main(["--scale", "scale", "--probe-root", str(code)]) == 0

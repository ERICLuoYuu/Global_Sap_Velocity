"""The merge writes one file per biome and no all-biome aggregate (owner, 2026-11-04: not needed).

The aggregate stacked biome files with different column sets under the first file's header, so values landed
under the wrong names, and every reader already skipped it by name.
"""

import pandas as pd

from notebooks.merge_gap_filled_hourly_orginal import save_biome_collections


def test_one_file_per_biome_and_no_all_biome_aggregate(tmp_path):
    temperate = pd.DataFrame({"site_name": ["A", "A"], "x": [1.0, 2.0]})
    boreal = pd.DataFrame({"site_name": ["B"], "y": [3.0]})

    save_biome_collections(
        {"Temperate forest": [temperate], "Boreal/taiga": [boreal], "Desert": []}, tmp_path, "merged_daily"
    )

    written = sorted(p.name for p in (tmp_path / "by_biome").iterdir())
    assert written == ["Boreal_taiga_merged_daily.csv", "Temperate_forest_merged_daily.csv"]  # nosec B101
    assert not list(tmp_path.rglob("all_biomes*"))  # nosec B101
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "by_biome" / "Boreal_taiga_merged_daily.csv"), boreal)


def test_a_biome_with_several_sites_stacks_them_in_one_file(tmp_path):
    first = pd.DataFrame({"site_name": ["A"], "x": [1.0]})
    second = pd.DataFrame({"site_name": ["C"], "x": [2.0]})

    save_biome_collections({"Temperate forest": [first, second]}, tmp_path, "merged_hourly")

    stacked = pd.read_csv(tmp_path / "by_biome" / "Temperate_forest_merged_hourly.csv")
    assert stacked["site_name"].tolist() == ["A", "C"]  # nosec B101


def test_an_aggregate_left_by_an_earlier_run_is_deleted(tmp_path):
    stale = tmp_path / "all_biomes_merged_daily.csv"
    stale.write_text("site_name,x\n-1593.53,A\n")

    save_biome_collections({"Temperate forest": [pd.DataFrame({"site_name": ["A"]})]}, tmp_path, "merged_daily")

    assert not stale.exists()  # nosec B101

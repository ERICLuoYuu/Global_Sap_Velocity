# src/sm_vpd_decoupling/tests/test_error_paths.py
"""Phase 5 Round 2 — negative-path and error-handling tests.

Each test asserts the ACTUAL documented behaviour of the source (confirmed by
reading the implementation before writing the assertion).  No behaviour is
invented; if the code is silent (logs a warning and continues), the test verifies
that rather than expecting a raise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.sm_vpd_decoupling.aggregate import decouple_all_sites
from src.sm_vpd_decoupling.decoupling import decouple_site
from src.sm_vpd_decoupling.loader import (
    apply_day_filter,
    load_table,
    resolve_daily_dir,
    standardise_site_frame,
)

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

_N = 400  # rows per synthetic site — enough for binning


def _write_valid_site_csv(path, site: str = "S1", n: int = _N, seed: int = 0) -> None:
    """Write a fully-valid per-site daily CSV (all required columns present)."""
    rng = np.random.default_rng(seed)
    sm = rng.uniform(0.1, 0.35, n)
    df = pd.DataFrame(
        {
            "site_name": [site] * n,
            "TIMESTAMP": pd.date_range("2008-01-01", periods=n, freq="D"),
            "sap_velocity": rng.uniform(1.0, 10.0, n),
            "ta": rng.uniform(16.0, 28.0, n),
            "vpd": rng.uniform(0.6, 2.5, n),
            "ppfd_in": rng.uniform(550.0, 1200.0, n),
            "sw_in": rng.uniform(150.0, 500.0, n),
            "surface_solar_radiation_downwards_hourly": rng.uniform(150.0, 500.0, n),
            "volumetric_soil_water_layer_1": sm,
            "volumetric_soil_water_layer_2": sm,
            "volumetric_soil_water_layer_3": sm,
            "volumetric_soil_water_layer_4": sm,
            "temperature_2m": rng.uniform(16.0, 28.0, n) + 273.15,
            "dewpoint_2m": rng.uniform(5.0, 15.0, n) + 273.15,
            "elevation": [100.0] * n,
            "pft": ["ENF"] * n,
            "biome": ["temperate"] * n,
            "prcip/PET": [0.6] * n,
            "canopy_height": [20.0] * n,
            "latitude_x": [45.0] * n,
            "longitude_x": [7.0] * n,
        }
    )
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# loader.resolve_daily_dir — error paths
# ---------------------------------------------------------------------------


def test_resolve_daily_dir_nonexistent_explicit_dir_raises(tmp_path):
    """An explicit data_dir path that does not exist raises FileNotFoundError.

    Source (loader.py ~l.154-156): if not (d.exists() and ...) → raises.
    """
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="no \\*.csv files"):
        resolve_daily_dir(str(missing))


def test_resolve_daily_dir_exists_but_no_csv_raises(tmp_path):
    """An explicit data_dir that exists but contains no *.csv raises FileNotFoundError.

    The condition checks both existence AND any(glob("*.csv")); no CSVs → raises.
    """
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    (empty_dir / "readme.txt").write_text("not a csv")
    with pytest.raises(FileNotFoundError, match="no \\*.csv files"):
        resolve_daily_dir(str(empty_dir))


def test_resolve_daily_dir_explicit_wins_over_auto(tmp_path):
    """Explicit data_dir is returned as-is; the auto-resolution candidates are ignored.

    Concretely: even if processed_root has no candidate, the explicit dir is used.
    Source: the ``if data_dir is not None`` branch returns immediately.
    """
    explicit = tmp_path / "explicit_daily"
    explicit.mkdir()
    _write_valid_site_csv(explicit / "S1.csv", "S1")
    # processed_root points to a path with no candidates at all.
    fake_root = tmp_path / "nowhere"
    fake_root.mkdir()
    result = resolve_daily_dir(str(explicit), processed_root=fake_root)
    assert result == explicit


# ---------------------------------------------------------------------------
# loader.load_table — error paths
# ---------------------------------------------------------------------------


def test_load_table_empty_directory_raises_file_not_found(tmp_path):
    """A data_dir directory with no *.csv files causes FileNotFoundError.

    Path: resolve_daily_dir succeeds (dir exists + has CSVs would fail), but
    actually resolve_daily_dir checks for CSVs and raises before load_table
    touches `files`.  Result: FileNotFoundError from resolve_daily_dir.
    """
    empty = tmp_path / "daily"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        load_table(str(empty))


def test_load_table_csv_missing_sap_velocity_column_is_skipped(tmp_path):
    """A CSV that lacks the `sap_velocity` column is silently skipped.

    Source (loader.py ~l.186): if raw.empty or "sap_velocity" not in raw.columns: continue.
    When the ONLY file is skipped → ValueError raised at the end.
    """
    d = tmp_path / "daily"
    d.mkdir()
    # Write a CSV with no sap_velocity column.
    pd.DataFrame(
        {
            "site_name": ["S1"] * 5,
            "TIMESTAMP": pd.date_range("2010-01-01", periods=5, freq="D"),
            "ta": [20.0] * 5,
        }
    ).to_csv(d / "S1_daily.csv", index=False)

    with pytest.raises(ValueError, match="No site files survived"):
        load_table(str(d))


def test_load_table_all_files_lacking_sap_velocity_raises_value_error(tmp_path):
    """When every CSV in the directory is missing sap_velocity, ValueError is raised.

    This tests that the message distinguishes the 'all skipped' path from the
    'empty dir' path (FileNotFoundError vs ValueError).
    """
    d = tmp_path / "daily"
    d.mkdir()
    for i in range(3):
        pd.DataFrame({"col_a": [i] * 5}).to_csv(d / f"site_{i}.csv", index=False)

    with pytest.raises(ValueError, match="No site files survived"):
        load_table(str(d))


def test_load_table_garbage_csv_is_skipped_run_continues(tmp_path):
    """A file that cannot be parsed/standardised is skipped; valid files still load.

    Source (~l.195): broad except catches ValueError/KeyError/... and logs a warning.
    The valid site file should still appear in the returned table.
    """
    d = tmp_path / "daily"
    d.mkdir()
    # Garbage file: binary noise that looks like a CSV name.
    (d / "corrupt_site.csv").write_bytes(b"\xff\xfe" + b"\x00" * 100)
    # Valid file that should survive.
    _write_valid_site_csv(d / "valid_site.csv", site="VALID", seed=42)

    table = load_table(str(d), climate_source="site", tair_min=15.0)
    assert "VALID" in table["site_name"].values


# ---------------------------------------------------------------------------
# loader.standardise_site_frame — ERA5 VPD path
# ---------------------------------------------------------------------------


def _raw_era5_frame(n: int = 20) -> pd.DataFrame:
    """Minimal raw frame with ERA5 temperature_2m / dewpoint_2m columns."""
    return pd.DataFrame(
        {
            "site_name": ["ERA5_SITE"] * n,
            "TIMESTAMP": pd.date_range("2012-06-01", periods=n, freq="D"),
            "sap_velocity": np.linspace(1.0, 5.0, n),
            # Kelvin values so the Kelvin-to-Celsius branch is exercised.
            "temperature_2m": np.linspace(283.15, 303.15, n),  # 10–30 °C
            "dewpoint_2m": np.linspace(275.15, 290.15, n),  # 2–17 °C
            "ppfd_in": np.linspace(600.0, 1100.0, n),
            "sw_in": np.linspace(200.0, 500.0, n),
            "surface_solar_radiation_downwards_hourly": np.linspace(200.0, 500.0, n),
            "volumetric_soil_water_layer_1": np.linspace(0.1, 0.3, n),
            "volumetric_soil_water_layer_2": np.linspace(0.1, 0.3, n),
            "volumetric_soil_water_layer_3": np.linspace(0.1, 0.3, n),
            "volumetric_soil_water_layer_4": np.linspace(0.1, 0.3, n),
        }
    )


def test_standardise_era5_vpd_is_finite_and_nonnegative():
    """ERA5 path: VPD derived via Tetens (es - ea).clip(lower=0) must be >= 0 everywhere.

    Source (loader.py ~l.83-87): _vpd_from_era5 clips to 0 → no negative values.
    Also confirms the resulting vpd series is fully finite (no NaN for valid inputs).
    """
    raw = _raw_era5_frame()
    out = standardise_site_frame(raw, climate_source="era5")
    assert out["vpd"].notna().all(), "VPD must be non-NaN for valid ERA5 inputs"
    assert (out["vpd"] >= 0.0).all(), "Tetens VPD must be >= 0 (clipped at 0)"


def test_standardise_era5_vpd_physically_plausible():
    """ERA5 VPD values must be positive when dewpoint < temperature (typical case).

    When T_air > T_dew, es > ea so VPD > 0.
    """
    raw = _raw_era5_frame()
    # All rows: T_air > T_dew (by construction: linspace 283..303 > 275..290)
    out = standardise_site_frame(raw, climate_source="era5")
    # At least some rows should have vpd > 0 (not all zero).
    assert (out["vpd"] > 0.0).any(), "At least some ERA5 VPD values should be positive"


# ---------------------------------------------------------------------------
# loader.apply_day_filter — no rows pass
# ---------------------------------------------------------------------------


def test_apply_day_filter_no_rows_pass_returns_empty_dataframe():
    """When no rows satisfy the three thresholds the result is an empty DataFrame.

    Source (loader.py ~l.126-138): mask is all False → table[mask] is empty.
    Empty result must NOT raise.
    """
    n = 10
    # Deliberately set values well below all thresholds.
    df = pd.DataFrame(
        {
            "tair": [5.0] * n,  # below default tair_min=15
            "vpd": [0.1] * n,  # below VPD_MIN_KPA=0.5
            "ppfd": [100.0] * n,  # below PPFD_MIN=500
            "E": [1.0] * n,
            "Gc": [0.5] * n,
        }
    )
    result = apply_day_filter(df, tair_min=15.0)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0, "No rows should pass the three-way AND filter"


# ---------------------------------------------------------------------------
# aggregate.decouple_all_sites — all sites below min_valid_days
# ---------------------------------------------------------------------------


def _small_table(n_per_site: int = 10, seed: int = 7) -> pd.DataFrame:
    """Table with 3 sites, each having n_per_site rows of random valid data."""
    rng = np.random.default_rng(seed)
    frames = []
    for site in ["X", "Y", "Z"]:
        sm = rng.uniform(0.1, 0.35, n_per_site)
        frames.append(
            pd.DataFrame(
                {
                    "site_name": [site] * n_per_site,
                    "vpd": rng.uniform(0.6, 2.5, n_per_site),
                    "swvl1": sm,
                    "E_norm": rng.uniform(0.1, 1.0, n_per_site),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_decouple_all_sites_all_below_min_valid_days_returns_empty():
    """When every site has fewer rows than min_valid_days, an empty DataFrame is returned.

    Source (aggregate.py ~l.28-29): if len(valid) < min_valid_days: continue.
    No rows are appended → pd.DataFrame.from_records([]) → empty.
    """
    table = _small_table(n_per_site=5)
    result = decouple_all_sites(table, sm_col="swvl1", response="E_norm", n_bins=5, min_valid_days=100)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0, "No site should survive the min_valid_days filter"


def test_decouple_all_sites_dropna_excludes_nan_response_rows(tmp_path):
    """Rows where the response is NaN are excluded before the min_valid_days check.

    Source (aggregate.py ~l.28): valid = g.dropna(subset=[vpd_col, sm_col, response]).
    If all response rows are NaN, len(valid)==0 < any min_valid_days → site skipped.
    """
    rng = np.random.default_rng(99)
    n = 200
    # One site where the response column is entirely NaN.
    nan_site = pd.DataFrame(
        {
            "site_name": ["NAN_SITE"] * n,
            "vpd": rng.uniform(0.6, 2.5, n),
            "swvl1": rng.uniform(0.1, 0.35, n),
            "E_norm": [np.nan] * n,
        }
    )
    # One site with valid data.
    sm = rng.uniform(0.1, 0.35, n)
    good_site = pd.DataFrame(
        {
            "site_name": ["GOOD_SITE"] * n,
            "vpd": rng.uniform(0.6, 2.5, n),
            "swvl1": sm,
            "E_norm": rng.uniform(0.1, 1.0, n),
        }
    )
    table = pd.concat([nan_site, good_site], ignore_index=True)
    result = decouple_all_sites(table, sm_col="swvl1", response="E_norm", n_bins=5, min_valid_days=10)
    # NaN_SITE must be excluded; GOOD_SITE must be present.
    assert "NAN_SITE" not in result["site_name"].values
    assert "GOOD_SITE" in result["site_name"].values


# ---------------------------------------------------------------------------
# decoupling.decouple_site — all-NaN response
# ---------------------------------------------------------------------------


def _site_frame_all_nan_response(n: int = 200, seed: int = 5) -> pd.DataFrame:
    """Site-level DataFrame where the response column is all NaN."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "vpd": rng.uniform(0.6, 2.5, n),
            "sm": rng.uniform(0.1, 0.35, n),
            "response": [np.nan] * n,
        }
    )


def test_decouple_site_all_nan_response_returns_nan_effects():
    """decouple_site with all-NaN response must return NaN for both effects.

    Source (decoupling.py ~l.86): dropna(subset=[..., response]) removes all rows →
    decouple_effect sees empty groups → len(effects) < MIN_COND_BINS → NaN.
    """
    df = _site_frame_all_nan_response()
    result = decouple_site(df, response="response", n_bins=5, vpd_col="vpd", sm_col="sm")
    assert "sm_given_vpd" in result
    assert "vpd_given_sm" in result
    assert np.isnan(result["sm_given_vpd"]), "all-NaN response → sm_given_vpd must be NaN"
    assert np.isnan(result["vpd_given_sm"]), "all-NaN response → vpd_given_sm must be NaN"

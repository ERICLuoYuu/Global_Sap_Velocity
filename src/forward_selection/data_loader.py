"""Load all site CSVs, compute all features, and cache to .npz.

Reuses helper functions from the training script without modification:
  - add_sap_flow_features()
  - apply_feature_engineering()
  - add_time_features()
  - create_spatial_groups()
  - calculate_soil_hydraulics_sr2006()

The data_loader extends apply_feature_engineering() rolling and lag groups
to include rh alongside ta/vpd/sw_in.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.forward_selection.feature_registry import (
    ADDITIONAL_FEATURES,
    ALL_FEATURE_ENGINEERING_GROUPS,
    PFT_ONEHOT_COLS,
)

logger = logging.getLogger(__name__)


def _import_training_helpers() -> tuple:
    """Lazy-import heavy helpers from the training script.

    Deferred to avoid pulling in cartopy, tensorflow, shap, etc.
    at module load time (they are only needed for cache building).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.hyperparameter_optimization.test_hyperparameter_tuning_ML_spatial_stratified_prediction import (
        add_sap_flow_features,
        add_time_features,
        apply_feature_engineering,
        create_spatial_groups,
    )

    return add_sap_flow_features, add_time_features, apply_feature_engineering, create_spatial_groups


# ── rh extensions for rolling / lag groups ──────────────────────────────────
def _add_rh_rolling_and_lag(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add rh rolling statistics and lag that the base FE doesn't compute.

    The training script's ``apply_feature_engineering()`` computes rolling
    stats for ta/vpd/sw_in only.  We extend with rh here.
    """
    df = df.copy()
    new_features: list[str] = []

    if "rh" not in df.columns:
        return df, new_features

    for window in [3, 7, 14]:
        mn = f"rh_roll{window}d_mean"
        sd = f"rh_roll{window}d_std"
        df[mn] = df["rh"].rolling(window, min_periods=1).mean()
        df[sd] = df["rh"].rolling(window, min_periods=2).std()
        new_features.extend([mn, sd])

    df["rh_lag1d"] = df["rh"].shift(1)
    new_features.append("rh_lag1d")

    return df, new_features


def load_and_cache_features(
    data_dir: Path,
    cache_path: Path,
    *,
    target_col: str = "sap_velocity",
    time_scale: str = "daily",
    is_only_day: bool = False,
    grid_size: float = 0.05,
) -> dict[str, Any]:
    """Load all site CSVs, compute every feature, and save to .npz cache.

    Parameters
    ----------
    data_dir : Path
        Directory containing per-site merged CSV files.
    cache_path : Path
        Output path for the .npz cache file.
    target_col : str
        Name of the target variable column.
    time_scale : str
        'daily' or 'hourly'.
    is_only_day : bool
        If True, filter to daytime records (sw_in > 10).
    grid_size : float
        Spatial grid size for grouping (degrees).

    Returns
    -------
    dict with keys: X, y, groups, pfts_encoded, feature_names, pft_categories
    """
    add_sap_flow_features, add_time_features, apply_feature_engineering, create_spatial_groups = (
        _import_training_helpers()
    )

    data_list = sorted(data_dir.glob(f"*{time_scale}.csv"))
    data_list = [f for f in data_list if "all_biomes_merged" not in f.name]

    if not data_list:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    logger.info("Found %d site CSV files in %s", len(data_list), data_dir)

    # Build the full used_cols list (base + additional)
    base_features = [
        target_col,
        "sw_in",
        "ws",
        "precip",
        "ta",
        "ta_max",
        "ta_min",
        "vpd",
        "vpd_max",
        "vpd_min",
        "ext_rad",
        "ppfd_in",
        "pft",
        "canopy_height",
        "elevation",
        "LAI",
        "prcip/PET",
        "volumetric_soil_water_layer_1",
        "soil_temperature_level_1",
        "day_length",
    ]
    daily_only_cols = {"day_length", "ta_max", "ta_min", "vpd_max", "vpd_min"}
    if time_scale == "hourly":
        base_features = [f for f in base_features if f not in daily_only_cols]

    used_cols = list(set(base_features + ADDITIONAL_FEATURES))
    all_possible_pft_types = PFT_ONEHOT_COLS

    site_data_dict: dict[str, pd.DataFrame] = {}
    site_info_dict: dict[str, dict] = {}
    final_feature_names: list[str] | None = None

    for data_file in data_list:
        try:
            df = pd.read_csv(data_file, parse_dates=["TIMESTAMP"])
            logger.info("Loaded %s, shape %s", data_file.name, df.shape)

            if is_only_day:
                if "sw_in" not in df.columns:
                    continue
                df = df[df["sw_in"] > 10]
                if df.empty:
                    continue

            site_id = df["site_name"].iloc[0]

            lat_col = next((c for c in df.columns if c.lower() in ["lat", "latitude_x"]), None)
            lon_col = next((c for c in df.columns if c.lower() in ["lon", "longitude_x"]), None)
            pft_col = next(
                (c for c in df.columns if c.lower() in ["pft", "plant_functional_type", "biome"]),
                None,
            )
            if not (lat_col and lon_col and pft_col):
                logger.warning("Missing lat/lon/pft for %s, skipping", site_id)
                continue

            latitude = df[lat_col].median()
            longitude = df[lon_col].median()

            if site_id.startswith("CZE"):
                logger.info("Skipping site %s", site_id)
                continue

            df["latitude"] = latitude
            df["longitude"] = longitude

            # Step 1: add_sap_flow_features (always computed)
            df = add_sap_flow_features(df, verbose=False)

            pft_value = df[pft_col].mode()[0]

            # Step 2: set index + time features (all 8)
            df.set_index("solar_TIMESTAMP", inplace=True)
            df.sort_index(inplace=True)
            df = add_time_features(df, datetime_column=None)

            # Step 3: apply ALL feature engineering groups
            engineered_names: list[str] = []
            df, engineered_names = apply_feature_engineering(
                df, ALL_FEATURE_ENGINEERING_GROUPS, time_scale, verbose=False
            )
            for fname in engineered_names:
                if fname not in used_cols:
                    used_cols.append(fname)

            # Step 4: add rh rolling/lag extensions
            df, rh_extra = _add_rh_rolling_and_lag(df)
            for fname in rh_extra:
                if fname not in used_cols:
                    used_cols.append(fname)

            # Step 5: all 8 time features
            time_features = [
                "Day sin",
                "Day cos",
                "Week sin",
                "Week cos",
                "Month sin",
                "Month cos",
                "Year sin",
                "Year cos",
            ]

            feature_cols = used_cols + time_features
            missing_cols = [c for c in feature_cols if c not in df.columns]

            if missing_cols:
                logger.warning(
                    "Site %s missing %d cols: %s — skipping",
                    site_id,
                    len(missing_cols),
                    missing_cols[:10],
                )
                continue

            df = df[feature_cols].copy()

            # Step 6: one-hot encode PFT
            if "pft" in df.columns:
                orig_cols = [c for c in feature_cols if c != "pft"]
                pft_cat = pd.Categorical(df["pft"], categories=all_possible_pft_types)
                pft_df = pd.get_dummies(pft_cat)
                pft_df.index = df.index
                df = df[orig_cols].join(pft_df)

            final_feature_names = [c for c in df.columns if c != target_col]

            # Step 7: convert + clean
            df = df.astype(np.float32)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)

            if df.empty:
                continue

            site_data_dict[site_id] = df
            site_info_dict[site_id] = {
                "latitude": latitude,
                "longitude": longitude,
                "pft": pft_value,
                "data_count": len(df),
            }

        except Exception:
            logger.exception("Error processing %s", data_file.name)

    if not site_data_dict:
        raise RuntimeError("No valid site data could be processed")

    # --- Spatial grouping ---
    site_ids = np.array(list(site_info_dict.keys()))
    latitudes = [site_info_dict[s]["latitude"] for s in site_ids]
    longitudes = [site_info_dict[s]["longitude"] for s in site_ids]

    spatial_groups, _stats = create_spatial_groups(
        lat=latitudes,
        lon=longitudes,
        method="grid",
        lat_grid_size=grid_size,
        lon_grid_size=grid_size,
    )
    site_to_group = dict(zip(site_ids, spatial_groups))
    site_to_pft = {sid: site_info_dict[sid]["pft"] for sid in site_ids}

    # --- Concatenate into record-level arrays ---
    list_X, list_y, list_groups, list_pfts = [], [], [], []

    for site_id, df in site_data_dict.items():
        y_site = df.pop(target_col).values
        X_site = df.values
        n = len(y_site)

        list_X.append(X_site)
        list_y.append(y_site)
        list_groups.append(np.full(n, site_to_group[site_id]))
        list_pfts.append(np.full(n, site_to_pft[site_id]))

    del site_data_dict

    X_all = np.vstack(list_X)
    y_all = np.concatenate(list_y)
    groups_all = np.concatenate(list_groups)
    pfts_all = np.concatenate(list_pfts)

    pfts_encoded, pft_categories = pd.factorize(pfts_all)
    logger.info(
        "Assembled matrix: X=%s, y=%s, %d PFT classes",
        X_all.shape,
        y_all.shape,
        len(pft_categories),
    )

    # --- Save cache ---
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        X=X_all,
        y=y_all,
        groups=groups_all,
        pfts_encoded=pfts_encoded,
        feature_names=np.array(final_feature_names),
        pft_categories=np.array(list(pft_categories)),
    )
    logger.info("Cache saved to %s (%.1f MB)", cache_path, cache_path.stat().st_size / 1e6)

    return {
        "X": X_all,
        "y": y_all,
        "groups": groups_all,
        "pfts_encoded": pfts_encoded,
        "feature_names": final_feature_names,
        "pft_categories": list(pft_categories),
    }


def load_cache(cache_path: Path) -> dict[str, Any]:
    """Load a previously saved .npz feature cache.

    Parameters
    ----------
    cache_path : Path
        Path to the .npz file.

    Returns
    -------
    dict with keys: X, y, groups, pfts_encoded, feature_names, pft_categories
    """
    cache_path = Path(cache_path)
    data = np.load(cache_path, allow_pickle=True)
    result = {
        "X": data["X"],
        "y": data["y"],
        "groups": data["groups"],
        "pfts_encoded": data["pfts_encoded"],
        "feature_names": list(data["feature_names"]),
        "pft_categories": list(data["pft_categories"]),
    }
    logger.info(
        "Loaded cache: X=%s, %d features, %d PFT classes",
        result["X"].shape,
        len(result["feature_names"]),
        len(result["pft_categories"]),
    )
    return result

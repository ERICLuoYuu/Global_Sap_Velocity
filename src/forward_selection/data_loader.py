"""Load all site CSVs, compute all features, and cache to .npz.

Reuses helper functions from the training script without modification:
  - apply_all_feature_engineering()
  - add_time_features()
  - create_spatial_groups()
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.feature_contract.definitions import (
    DEFAULT_CONTRACT,
    assert_dataset_contract,
    assert_no_retired_swc_features,
    assert_no_servable_duplicate_features,
    servable_duplicate_features,
    validate_contract,
)
from src.forward_selection.feature_registry import (
    ADDITIONAL_FEATURES,
    INTERMEDIATE_ONLY,
    PFT_ONEHOT_COLS,
)

_INTERMEDIATE_SET = frozenset(INTERMEDIATE_ONLY)


def _intermediate_columns(contract: str) -> frozenset:
    """Columns loaded as feature-engineering inputs but kept OUT of the design matrix.

    Under ``servable`` this also covers the ERA5-Land daily total in metres, because ``precip`` is
    that same total in mm and the two would be one variable under two names. Feature engineering's
    ``cwd`` group still reads the metres column, so it is dropped here rather than never loaded.
    """
    return _INTERMEDIATE_SET | frozenset(servable_duplicate_features(contract))

logger = logging.getLogger(__name__)


def _import_training_helpers() -> tuple:
    """Lazy-import heavy helpers from the training script.

    Deferred to avoid pulling in cartopy, tensorflow, shap, etc.
    at module load time (they are only needed for cache building).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.hyperparameter_optimization.feature_engineering import (
        apply_all_feature_engineering,
    )
    from src.hyperparameter_optimization.training_utils import (
        add_time_features,
        create_spatial_groups,
    )

    return apply_all_feature_engineering, add_time_features, create_spatial_groups


#: Column names that can carry a site's vegetation class, in the order they
#: should be preferred. ``pft`` is the real plant functional type; ``biome`` is
#: a coarser fallback for sites that never received one.
PFT_COLUMN_PRIORITY = ("pft", "plant_functional_type", "biome")


def load_and_cache_features(
    data_dir: Path,
    cache_path: Path,
    *,
    target_col: str = "sap_velocity",
    time_scale: str = "daily",
    is_only_day: bool = False,
    grid_size: float = 0.05,
    spatial_method: str = "default",
    contract: str = DEFAULT_CONTRACT,
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
    contract : str
        Feature contract, ``"site"`` (default, legacy) or ``"servable"``; see
        :mod:`src.feature_contract.definitions`. The tree (the parent of ``data_dir``) must have
        been built under it, every site is engineered under it, and the cache records it.

    Returns
    -------
    dict with keys: X, y, groups, pfts_encoded, feature_names, pft_categories
    """
    contract = validate_contract(contract)
    # Before anything heavy: a servable cache from a legacy tree would read per-SITE soil-moisture
    # z-scores as climatological, and a site cache from a servable tree the reverse.
    assert_dataset_contract(Path(data_dir).parent, contract)
    apply_all_feature_engineering, add_time_features, create_spatial_groups = _import_training_helpers()

    data_list = sorted(data_dir.glob(f"*{time_scale}.csv"))
    data_list = [f for f in data_list if "all_biomes_merged" not in f.name]

    if not data_list:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    logger.info("Found %d site CSV files in %s (feature contract: %s)", len(data_list), data_dir, contract)

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
        "soil_temperature_level_1",
        "day_length",
    ]
    daily_only_cols = {"day_length", "ta_max", "ta_min", "vpd_max", "vpd_min"}
    if time_scale == "hourly":
        base_features = [f for f in base_features if f not in daily_only_cols]

    # dict.fromkeys, not set(): a set of strings iterates in an order drawn from the process's hash
    # seed, so every cache build laid the design matrix out differently -- and forward selection breaks
    # ties by column position, as do XGBoost's column subsampling and split finding. The same rows and
    # seeds could select a different feature set. Declared order; the first occurrence of a name wins.
    used_cols = list(dict.fromkeys(base_features + ADDITIONAL_FEATURES))
    assert_no_retired_swc_features(used_cols, "the FFS column list (base features + ADDITIONAL_FEATURES)")
    _intermediates = _intermediate_columns(contract)
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
            # Resolve by walking the PRIORITY LIST, not df.columns. Iterating the
            # columns hands precedence to whichever name appears first in the CSV,
            # and in the merged daily files "biome" sits at column 39 and "pft" at
            # 40 -- so every site was stratified by biome while the logs said PFT.
            _by_lower = {c.lower(): c for c in df.columns}
            pft_col = next(
                (_by_lower[name] for name in PFT_COLUMN_PRIORITY if name in _by_lower),
                None,
            )
            if not (lat_col and lon_col and pft_col):
                logger.warning("Missing lat/lon/pft for %s, skipping", site_id)
                continue

            latitude = df[lat_col].median()
            longitude = df[lon_col].median()

            df["latitude"] = latitude
            df["longitude"] = longitude

            pft_value = df[pft_col].mode()[0]

            # Step 1: set index + time features (all 8)
            df.set_index("solar_TIMESTAMP", inplace=True)
            df.sort_index(inplace=True)
            df = add_time_features(df, datetime_column=None)

            # Step 2: apply ALL feature engineering (single call)
            df, engineered_names = apply_all_feature_engineering(df, time_scale, contract=contract)
            for fname in engineered_names:
                if fname not in used_cols:
                    used_cols.append(fname)

            # Step 3: all 8 time features
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

            feature_cols = [c for c in used_cols + time_features if c not in _intermediates]
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
        method=spatial_method,
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
        feature_contract=np.array(contract),
    )
    logger.info("Cache saved to %s (%.1f MB)", cache_path, cache_path.stat().st_size / 1e6)

    return {
        "X": X_all,
        "y": y_all,
        "groups": groups_all,
        "pfts_encoded": pfts_encoded,
        "feature_names": final_feature_names,
        "pft_categories": list(pft_categories),
        "feature_contract": contract,
    }


def load_cache(cache_path: Path) -> dict[str, Any]:
    """Load a previously saved .npz feature cache.

    Parameters
    ----------
    cache_path : Path
        Path to the .npz file.

    Returns
    -------
    dict with keys: X, y, groups, pfts_encoded, feature_names, pft_categories, feature_contract
        (``None`` for a cache written before the contract was recorded)

    Raises
    ------
    ValueError
        The cache names a soil-moisture feature retired on 2026-09-14. Such a cache also lacks the
        level and anomaly columns, so selection on it would silently run without soil moisture.
    """
    cache_path = Path(cache_path)
    data = np.load(cache_path, allow_pickle=True)
    # Names first: an .npz member is read on access, so a stale cache fails before X is decompressed.
    feature_names = list(data["feature_names"])
    cached_contract = data["feature_contract"].item() if "feature_contract" in data.files else None
    _source = f"feature cache {cache_path} (rebuild it with run_selection --build_cache)"
    assert_no_retired_swc_features(feature_names, _source)
    if cached_contract:
        # A servable cache built before precip became the ERA5 total carries both names for it.
        assert_no_servable_duplicate_features(feature_names, _source, cached_contract)
    result = {
        "X": data["X"],
        "y": data["y"],
        "groups": data["groups"],
        "pfts_encoded": data["pfts_encoded"],
        "feature_names": feature_names,
        "pft_categories": list(data["pft_categories"]),
        "feature_contract": cached_contract,
    }
    logger.info(
        "Loaded cache: X=%s, %d features, %d PFT classes, feature contract %s",
        result["X"].shape,
        len(result["feature_names"]),
        len(result["pft_categories"]),
        result["feature_contract"],
    )
    return result

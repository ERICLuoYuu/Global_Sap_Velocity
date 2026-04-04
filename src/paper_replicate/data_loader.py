"""Data loading and preprocessing for Loritz et al. (2024) replication.

Faithful port of their gauged Cell 3-9 and ungauged Cell 6-10.
Replaces dynamic variable creation with a dict of DataFrames.
"""

import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    FEATURES_ONEHOT,
    FEATURES_WITHOUT_ONEHOT,
    MONTH_LIST,
    USECOLS_ENV,
    USECOLS_EO,
    USECOLS_PL,
    USECOLS_SI,
    cyclical_encode,
)


def load_all_data(
    data_dir: Path,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[str]]]:
    """Load all per-tree-year DataFrames from the data.zip structure.

    Ports their Cell 6 logic: iterate plant_md/, load env, eo, site, plant,
    sapf data per site, split into per-tree per-year DataFrames.

    Returns:
        all_data: dict mapping plant_year_id -> DataFrame
        igbp_class: dict mapping IGBP type -> list of site codes
    """
    data_dir = Path(data_dir)
    all_data: Dict[str, pd.DataFrame] = {}
    igbp_class: Dict[str, List[str]] = {}

    plant_md_dir = data_dir / "plant_md"
    plant_md_files = sorted(plant_md_dir.glob("*_plant_md.csv"))

    for md_file in plant_md_files:
        si_code = md_file.stem.replace("_plant_md", "")

        # Load environmental features
        env_path = data_dir / "env_data" / f"{si_code}_env_data.csv"
        df_env = pd.read_csv(
            env_path,
            index_col="TIMESTAMP",
            usecols=USECOLS_ENV,
            parse_dates=True,
        )
        df_env = df_env.tz_localize(None)
        df_env = df_env.sort_index(axis=1)
        df_env = df_env.resample("h").mean()
        df_env.fillna(0, inplace=True)
        df_env = df_env[df_env.index.month.isin(MONTH_LIST)]
        index_year = df_env.index.year.unique().tolist()

        # Load earth observation (LAI)
        eo_path = data_dir / "eo_data" / f"{si_code}_eo_data.csv"
        df_eo = pd.read_csv(
            eo_path,
            index_col="date",
            usecols=USECOLS_EO,
            parse_dates=True,
        )
        df_eo = df_eo.sort_index(axis=1)
        df_eo = df_eo.resample("h").mean()
        df_eo.fillna(0, inplace=True)
        df_eo = df_eo[
            df_eo.index.month.isin(MONTH_LIST) & df_eo.index.year.isin(index_year)
        ]

        if df_eo.empty:
            warnings.warn(f"Warning: EO DataFrame empty for {si_code}")

        df_env = df_env.join(df_eo)

        # Load site features
        si_path = data_dir / "site_md" / f"{si_code}_site_md.csv"
        df_site = pd.read_csv(si_path, usecols=USECOLS_SI, index_col=False)
        df_site = df_site[USECOLS_SI]

        # Load plant features
        pl_path = data_dir / "plant_md" / f"{si_code}_plant_md.csv"
        df_plant = pd.read_csv(pl_path, usecols=USECOLS_PL, index_col=False)

        # Load sap flow target
        sapf_path = data_dir / "sapf_data" / f"{si_code}_sapf_data.csv"
        df_sapf = pd.read_csv(sapf_path, index_col="TIMESTAMP", parse_dates=True)

        # Drop solar timestamp columns if present
        for col in ("solar_TIMESTAMP", "TIMESTAMP_solar"):
            if col in df_sapf.columns:
                df_sapf = df_sapf.drop(columns=[col], axis=1)

        df_sapf = df_sapf.resample("h").mean()
        df_sapf.fillna(0, inplace=True)
        df_sapf = df_sapf[df_sapf.index.month.isin(MONTH_LIST)]

        # Replace negative sap flow with 0, set nighttime to 0
        df_sapf[df_sapf < 0] = 0
        df_sapf[(df_sapf.index.hour >= 22) | (df_sapf.index.hour <= 6)] = 0

        si_plants = df_sapf.columns.tolist()

        # Track IGBP class for ungauged splits
        igbp_name = df_site["si_igbp"].tolist()[0]
        if igbp_name not in igbp_class:
            igbp_class[igbp_name] = []
        if si_code not in igbp_class[igbp_name]:
            igbp_class[igbp_name].append(si_code)

        for pl_code in si_plants:
            try:
                # Get genus name
                pl_genus = (
                    df_plant[df_plant["pl_code"] == pl_code]["pl_species"]
                    .iloc[0]
                    .split()[0]
                )
                df_si_single = df_site.copy()

                # Extract plant features for single plant
                df_p_single = df_plant[df_plant["pl_code"] == pl_code]
                df_p_single = df_p_single.drop(["pl_code", "pl_species"], axis=1)
                df_p_single = df_p_single.reset_index(drop=True)

                # One-hot encode genus and IGBP
                df_si_single["pl_genus"] = pl_genus
                df_onehot = pd.get_dummies(
                    df_si_single[FEATURES_ONEHOT].astype(str)
                )
                df_si_single_onehot = pd.concat(
                    [df_si_single, df_onehot], axis=1
                ).drop(columns=FEATURES_ONEHOT, axis=1)

                # Combine all static features
                df_static = pd.concat([df_p_single, df_si_single_onehot], axis=1)
                df_static = df_static.iloc[np.full(df_env.shape[0], 0)]
                df_static = df_static.set_index(df_env.index)

                # Extract sap flow for single plant
                df_sapf_single = df_sapf.loc[:, df_sapf.columns == pl_code]
                df_sapf_single = df_sapf_single.set_index(df_env.index)
                df_sapf_single.columns = ["sapf"]
                hours = df_sapf_single.index.hour
                df_sapf_single["time_hours"] = np.hstack(
                    [cyclical_encode(hour) for hour in hours]
                )

                # Combine everything
                df_all_single = df_env.join([df_static, df_sapf_single])

                # Split by year
                for year in df_all_single.index.year.unique():
                    df_oneyear = df_all_single[df_all_single.index.year == year]
                    plant_year_id = f"{pl_code}_{year}"
                    all_data[plant_year_id] = df_oneyear

            except Exception:
                print(f"Plant with ID: {pl_code} did not work")

    return all_data, igbp_class


def compute_standardization_stats(
    all_data: Dict[str, pd.DataFrame],
    plant_ids: Optional[List[str]] = None,
) -> Tuple[pd.Series, pd.Series, float, float, float]:
    """Compute mean/std over ALL data (matching their Cell 7).

    Note: Their code computes stats over ALL data, not just training data.
    This is technically data leakage, but we replicate it faithfully.

    Args:
        all_data: dict of plant_year_id -> DataFrame
        plant_ids: optional subset of IDs to compute stats over.
                   If None, uses all data (matching their code).

    Returns:
        all_mean: per-feature mean (for features_without_onehot)
        all_std: per-feature std (for features_without_onehot)
        sapf_mean: mean of non-zero sap flow
        sapf_std: std of non-zero sap flow
        min_obs: threshold for excluding zero sap flow in z-score space
    """
    if plant_ids is None:
        plant_ids = sorted(all_data.keys())

    total_sum = 0
    total_count = 0
    total_sapf_sum = 0
    total_sapf_count = 0

    for pid in plant_ids:
        df = all_data[pid]
        total_sum += df[FEATURES_WITHOUT_ONEHOT].sum()
        total_count += len(df)

        sapf_pos = df[df["sapf"] > 0]["sapf"]
        total_sapf_sum += sapf_pos.sum()
        total_sapf_count += len(sapf_pos)

    all_mean = total_sum / total_count
    sapf_mean = total_sapf_sum / total_sapf_count

    # Second pass for std (population std, not sample std)
    total_squared_diff = 0
    total_squared_sapf_diff = 0

    for pid in plant_ids:
        df = all_data[pid]
        diff = df[FEATURES_WITHOUT_ONEHOT] - all_mean
        total_squared_diff += (diff**2).sum()

        sapf_pos = df[df["sapf"] > 0]["sapf"]
        total_squared_sapf_diff += ((sapf_pos - sapf_mean) ** 2).sum()

    all_std = np.sqrt(total_squared_diff / total_count)
    sapf_std = np.sqrt(total_squared_sapf_diff / total_sapf_count)

    min_obs = (-sapf_mean) / sapf_std

    return all_mean, all_std, sapf_mean, sapf_std, min_obs


def get_all_columns(all_data: Dict[str, pd.DataFrame]) -> Tuple[List[str], List[str]]:
    """Collect all unique columns across all DataFrames (Cell 7-8).

    Returns:
        all_columns: sorted list of all columns
        features: all_columns minus 'sapf'
    """
    unique_columns = set()
    for df in all_data.values():
        unique_columns.update(df.columns.tolist())

    all_columns = sorted(unique_columns)
    features = [c for c in all_columns if c != "sapf"]
    return all_columns, features


def split_gauged(
    all_plant_ids: List[str],
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    """Split data for gauged model: 50% train, 10% val, 40% test (Cell 9).

    Args:
        all_plant_ids: sorted list of plant_year IDs
        seed: random seed for shuffle

    Returns:
        train_plants, val_plants, test_plants
    """
    rand_plants = sorted(all_plant_ids)
    random.Random(seed).shuffle(rand_plants)

    train_split = round(len(all_plant_ids) * 0.5)
    val_split = round(len(all_plant_ids) * 0.6)

    train_plants = rand_plants[:train_split]
    val_plants = rand_plants[train_split:val_split]
    test_plants = rand_plants[val_split:]

    return train_plants, val_plants, test_plants


def split_ungauged(
    all_plant_ids: List[str],
    igbp_class: Dict[str, List[str]],
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Split data for ungauged model: IGBP-stratified stand split (ungauged Cell 9).

    Each IGBP class is split 50/50 by stands. The second half of each
    shuffled class is used for training (matching their code: half_counts[category]:).

    Args:
        all_plant_ids: sorted list of plant_year IDs
        igbp_class: dict mapping IGBP type -> list of site codes
        seed: random seed

    Returns:
        train_plants, test_plants
    """
    # IGBP-stratified split of stands
    half_counts = {
        cat: int(len(elements) / 2) for cat, elements in igbp_class.items()
    }
    shuffled_class = {
        cat: random.Random(seed).sample(elements, len(elements))
        for cat, elements in igbp_class.items()
    }
    # Training stands = second half (half_counts[cat]:)
    igbp_selected = {
        cat: shuffled[half_counts[cat]:]
        for cat, shuffled in shuffled_class.items()
    }
    train_stand = [item for sublist in igbp_selected.values() for item in sublist]

    train_plants = []
    test_plants = []

    for plant_id in all_plant_ids:
        parts = plant_id.split("_")
        if len(parts) < 5:
            warnings.warn(f"An error in plant: {plant_id}")
            continue
        plant_stand = "_".join(parts[:-4])

        if plant_stand in train_stand:
            train_plants.append(plant_id)
        else:
            test_plants.append(plant_id)

    return train_plants, test_plants


def load_baseline_data(
    data_dir: Path,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[str]]]:
    """Load data for baseline model (simplified, no env/eo features).

    Ports their baseline Cell 4. Naming convention differs:
    stand_genus_idx_year instead of pl_code_year.
    """
    data_dir = Path(data_dir)
    all_data: Dict[str, pd.DataFrame] = {}
    igbp_class: Dict[str, List[str]] = {}

    plant_md_dir = data_dir / "plant_md"
    plant_md_files = sorted(plant_md_dir.glob("*_plant_md.csv"))

    for md_file in plant_md_files:
        si_code = md_file.stem.replace("_plant_md", "")

        # Load plant features
        df_plant = pd.read_csv(
            md_file,
            usecols=["pl_code", "pl_species"],
            index_col=False,
        )

        # Load site features (for IGBP)
        si_path = data_dir / "site_md" / f"{si_code}_site_md.csv"
        df_site = pd.read_csv(si_path, usecols=USECOLS_SI, index_col=False)

        # Load sap flow
        sapf_path = data_dir / "sapf_data" / f"{si_code}_sapf_data.csv"
        df_sapf = pd.read_csv(sapf_path, index_col="TIMESTAMP", parse_dates=True)

        for col in ("solar_TIMESTAMP", "TIMESTAMP_solar"):
            if col in df_sapf.columns:
                df_sapf = df_sapf.drop(columns=[col], axis=1)

        df_sapf = df_sapf.resample("h").mean()
        df_sapf.fillna(0, inplace=True)
        df_sapf = df_sapf[df_sapf.index.month.isin(MONTH_LIST)]
        df_sapf[df_sapf < 0] = 0
        # Baseline uses strict inequality: hour < 22 and hour > 6
        df_sapf = df_sapf[(df_sapf.index.hour < 22) & (df_sapf.index.hour > 6)]

        si_plants = df_sapf.columns.tolist()

        # IGBP tracking
        igbp_name = df_site["si_igbp"].tolist()[0]
        if igbp_name not in igbp_class:
            igbp_class[igbp_name] = []
        if si_code not in igbp_class[igbp_name]:
            igbp_class[igbp_name].append(si_code)

        for j, pl_code in enumerate(si_plants):
            try:
                pl_genus = (
                    df_plant[df_plant["pl_code"] == pl_code]["pl_species"]
                    .iloc[0]
                    .split()[0]
                )

                df_sapf_single = df_sapf.loc[:, df_sapf.columns == pl_code]
                df_sapf_single = df_sapf_single.set_index(df_sapf.index)
                df_sapf_single.columns = ["sapf"]

                for year in df_sapf_single.index.year.unique():
                    df_oneyear = df_sapf_single[df_sapf_single.index.year == year]
                    # Baseline naming: stand_genus_idx_year
                    stand_name = pl_code.rsplit("_", 3)[0]
                    plant_year_id = f"{stand_name}_{pl_genus}_{j}_{year}"
                    all_data[plant_year_id] = df_oneyear

            except Exception:
                print(f"Plant with ID: {pl_code} did not work")

    return all_data, igbp_class


def split_baseline_gauged(
    all_plant_ids: List[str],
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Split for gauged baseline: 50% train, skip 10% val, 40% test."""
    rand_plants = sorted(all_plant_ids)
    random.Random(seed).shuffle(rand_plants)

    n = len(all_plant_ids)
    train_plants = rand_plants[: round(n * 0.5)]
    test_plants = rand_plants[round(n * 0.6) :]

    return train_plants, test_plants


def split_baseline_ungauged(
    all_plant_ids: List[str],
    igbp_class: Dict[str, List[str]],
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Split for ungauged baseline: IGBP-stratified stand split."""
    half_counts = {
        cat: int(len(elements) / 2) for cat, elements in igbp_class.items()
    }
    shuffled_class = {
        cat: random.Random(seed).sample(elements, len(elements))
        for cat, elements in igbp_class.items()
    }
    igbp_selected = {
        cat: shuffled[half_counts[cat]:]
        for cat, shuffled in shuffled_class.items()
    }
    train_stand = [item for sublist in igbp_selected.values() for item in sublist]

    train_plants = []
    test_plants = []

    for plant_id in all_plant_ids:
        parts = plant_id.split("_")
        if len(parts) < 5:
            warnings.warn(f"An error in plant: {plant_id}")
            continue
        # Baseline naming: stand_genus_idx_year -> stand = parts[:-3]
        plant_stand = "_".join(parts[:-3])

        if plant_stand in train_stand:
            train_plants.append(plant_id)
        else:
            test_plants.append(plant_id)

    return train_plants, test_plants

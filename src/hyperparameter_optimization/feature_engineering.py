"""
Feature engineering functions for sap velocity prediction.

Extracted from test_hyperparameter_tuning_ML_spatial_stratified.py for reuse
across training, prediction, and standalone SHAP analysis scripts.
"""

import logging

import numpy as np
import pandas as pd


def add_sap_flow_features(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Add scientifically-grounded features for sap velocity prediction.
    """
    df = df.copy()
    new_features = []

    # Priority 1: Highest Impact Features
    if "vpd" in df.columns and "sw_in" in df.columns:
        df["vpd_x_sw_in"] = df["vpd"] * df["sw_in"]
        new_features.append("vpd_x_sw_in")

    if "vpd" in df.columns:
        df["vpd_squared"] = df["vpd"] ** 2
        new_features.append("vpd_squared")

    if "vpd" in df.columns:
        df["vpd_lag_1h"] = df["vpd"].shift(1)
        new_features.append("vpd_lag_1h")

    if "sw_in" in df.columns:
        df["is_daytime"] = (df["sw_in"] > 10).astype(np.float32)
        new_features.append("is_daytime")

    if "latitude" in df.columns:
        df["southern_hemisphere"] = (df["latitude"] < 0).astype(np.float32)
        new_features.append("southern_hemisphere")

    # Priority 2: High Impact Features
    if "vpd" in df.columns:
        df["vpd_mean_6h"] = df["vpd"].rolling(6, min_periods=1).mean()
        new_features.append("vpd_mean_6h")

    if "vpd" in df.columns:
        df["vpd_high"] = (df["vpd"] > 2.5).astype(np.float32)
        new_features.append("vpd_high")

    if "sw_in" in df.columns and "ext_rad" in df.columns:
        df["clear_sky_index"] = (df["sw_in"] / (df["ext_rad"] + 1)).clip(0, 1)
        new_features.append("clear_sky_index")

    if "canopy_height" in df.columns and "vpd" in df.columns:
        df["height_x_vpd"] = df["canopy_height"] * df["vpd"]
        new_features.append("height_x_vpd")

    if "ta" in df.columns:
        df["ta_lag_1h"] = df["ta"].shift(1)
        new_features.append("ta_lag_1h")

    if "ta" in df.columns:
        df["gdd"] = (df["ta"] - 5).clip(lower=0)
        new_features.append("gdd")

    # Priority 3: Additional Useful Features
    if "vpd" in df.columns:
        df["vpd_log"] = np.log1p(df["vpd"])
        new_features.append("vpd_log")

    if "sw_in" in df.columns and "LAI" in df.columns:
        k = 0.5
        df["absorbed_radiation"] = df["sw_in"] * (1 - np.exp(-k * df["LAI"]))
        new_features.append("absorbed_radiation")

    if "ws" in df.columns and "vpd" in df.columns:
        df["wind_x_vpd"] = df["ws"] * df["vpd"]
        new_features.append("wind_x_vpd")

    if "ta" in df.columns and "vpd" in df.columns:
        df["ta_x_vpd"] = df["ta"] * df["vpd"]
        new_features.append("ta_x_vpd")

    if "vpd" in df.columns and "prcip/PET" in df.columns:
        df["demand_x_supply"] = df["vpd"] * df["prcip/PET"]
        new_features.append("demand_x_supply")

    if "sw_in" in df.columns:
        df["sw_in_cumsum_day"] = df["sw_in"].rolling(24, min_periods=1).sum()
        new_features.append("sw_in_cumsum_day")

    if "vpd" in df.columns:
        df["vpd_cumsum_6h"] = df["vpd"].rolling(6, min_periods=1).sum()
        new_features.append("vpd_cumsum_6h")

    if "ta" in df.columns:
        df["ta_change_1h"] = df["ta"].diff(1)
        new_features.append("ta_change_1h")

    if "latitude" in df.columns:
        df["tropical"] = (abs(df["latitude"]) < 23.5).astype(np.float32)
        df["boreal"] = (abs(df["latitude"]) > 55).astype(np.float32)
        new_features.extend(["tropical", "boreal"])

    # Conditional Features
    soil_cols = [c for c in df.columns if any(x in c.lower() for x in ["swc", "soil_moisture", "sm_", "vwc"])]
    if soil_cols:
        sm_col = soil_cols[0]
        sm_min, sm_max = df[sm_col].quantile([0.05, 0.95])
        df["soil_moisture_rel"] = ((df[sm_col] - sm_min) / (sm_max - sm_min + 0.01)).clip(0, 1)
        df["soil_dry"] = (df["soil_moisture_rel"] < 0.3).astype(np.float32)
        new_features.extend(["soil_moisture_rel", "soil_dry"])

    if "rh" in df.columns and "ta" in df.columns:
        a, b = 17.27, 237.7
        alpha = (a * df["ta"] / (b + df["ta"])) + np.log(df["rh"] / 100 + 0.01)
        df["dew_point"] = b * alpha / (a - alpha)
        df["dew_point_depression"] = df["ta"] - df["dew_point"]
        new_features.extend(["dew_point", "dew_point_depression"])

    precip_col = None
    for col in ["precip", "precipitation", "rain", "prcp"]:
        if col in df.columns:
            precip_col = col
            break

    if precip_col:
        df["precip_sum_24h"] = df[precip_col].rolling(24, min_periods=1).sum()
        df["precip_sum_72h"] = df[precip_col].rolling(72, min_periods=1).sum()
        rain_events = df[precip_col] > 1
        df["hours_since_rain"] = (~rain_events).groupby((~rain_events).cumsum()).cumcount()
        new_features.extend(["precip_sum_24h", "precip_sum_72h", "hours_since_rain"])

    if "dbh" in df.columns:
        df["dbh_log"] = np.log1p(df["dbh"])
        df["sapwood_area_est"] = 0.5 * df["dbh"] ** 1.8
        new_features.extend(["dbh_log", "sapwood_area_est"])

    if verbose:
        print(f"Added {len(new_features)} new features: {new_features}")

    return df


def apply_feature_engineering(df, groups, time_scale="daily", verbose=False):
    """
    Apply selected feature engineering groups to a per-site DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Site data with original columns (already has TIMESTAMP index).
    groups : list of str
        Feature group names to apply.
    time_scale : str
        'daily' or 'hourly' — adapts rolling window sizes.
    verbose : bool

    Returns
    -------
    (df, new_feature_names) : tuple
    """
    df = df.copy()
    new_features = []

    # ── interactions ────────────────────────────────────────────────
    if "interactions" in groups:
        if "vpd" in df.columns and "sw_in" in df.columns:
            df["vpd_x_sw_in"] = df["vpd"] * df["sw_in"]
            new_features.append("vpd_x_sw_in")
        if "vpd" in df.columns:
            df["vpd_squared"] = df["vpd"] ** 2
            new_features.append("vpd_squared")
        if "ta" in df.columns and "vpd" in df.columns:
            df["ta_x_vpd"] = df["ta"] * df["vpd"]
            new_features.append("ta_x_vpd")
        if "canopy_height" in df.columns and "vpd" in df.columns:
            df["height_x_vpd"] = df["canopy_height"] * df["vpd"]
            new_features.append("height_x_vpd")
        if "ws" in df.columns and "vpd" in df.columns:
            df["wind_x_vpd"] = df["ws"] * df["vpd"]
            new_features.append("wind_x_vpd")
        if "vpd" in df.columns and "prcip/PET" in df.columns:
            df["demand_x_supply"] = df["vpd"] * df["prcip/PET"]
            new_features.append("demand_x_supply")

    # ── lags_1d ─────────────────────────────────────────────────────
    if "lags_1d" in groups:
        for col_name in ["ta", "vpd", "sw_in", "precip"]:
            if col_name in df.columns:
                feat = f"{col_name}_lag1d"
                df[feat] = df[col_name].shift(1)
                new_features.append(feat)

    # ── rolling statistics ──────────────────────────────────────────
    rolling_groups = {
        "rolling_3d": 3,
        "rolling_7d": 7,
        "rolling_14d": 14,
    }
    for grp, window in rolling_groups.items():
        if grp in groups:
            for col_name in ["ta", "vpd", "sw_in"]:
                if col_name in df.columns:
                    mn = f"{col_name}_roll{window}d_mean"
                    sd = f"{col_name}_roll{window}d_std"
                    df[mn] = df[col_name].rolling(window, min_periods=1).mean()
                    df[sd] = df[col_name].rolling(window, min_periods=2).std()
                    new_features.extend([mn, sd])

    # ── physics ─────────────────────────────────────────────────────
    if "physics" in groups:
        if "sw_in" in df.columns and "ext_rad" in df.columns:
            df["clear_sky_index"] = (df["sw_in"] / (df["ext_rad"] + 1)).clip(0, 1)
            new_features.append("clear_sky_index")
        if "ta" in df.columns:
            df["gdd"] = (df["ta"] - 5).clip(lower=0)
            new_features.append("gdd")
        if "sw_in" in df.columns and "LAI" in df.columns:
            df["absorbed_radiation"] = df["sw_in"] * (1 - np.exp(-0.5 * df["LAI"]))
            new_features.append("absorbed_radiation")

    # ── precip_memory ───────────────────────────────────────────────
    if "precip_memory" in groups:
        if "precip" in df.columns:
            df["precip_sum_3d"] = df["precip"].rolling(3, min_periods=1).sum()
            df["precip_sum_7d"] = df["precip"].rolling(7, min_periods=1).sum()
            rain_events = df["precip"] > 0.5
            no_rain = ~rain_events
            df["days_since_rain"] = no_rain.groupby(rain_events.cumsum()).cumcount().astype(np.float32)
            new_features.extend(["precip_sum_3d", "precip_sum_7d", "days_since_rain"])

    # ── indicators ──────────────────────────────────────────────────
    if "indicators" in groups:
        if "vpd" in df.columns:
            df["vpd_high"] = (df["vpd"] > 2.5).astype(np.float32)
            new_features.append("vpd_high")
        swc_col = next(
            (c for c in df.columns if any(x in c.lower() for x in ["volumetric_soil_water", "swc", "sm_"])),
            None,
        )
        if swc_col:
            sm_min, sm_max = df[swc_col].quantile([0.05, 0.95])
            df["soil_moisture_rel"] = ((df[swc_col] - sm_min) / (sm_max - sm_min + 0.01)).clip(0, 1)
            df["soil_dry"] = (df["soil_moisture_rel"] < 0.3).astype(np.float32)
            new_features.extend(["soil_moisture_rel", "soil_dry"])

    # ── static_enrich ───────────────────────────────────────────────
    if "static_enrich" in groups:
        for col_name in ["stand_age", "slope", "mean_annual_temp", "precip_seasonality"]:
            if col_name in df.columns:
                new_features.append(col_name)

    # Drop NaNs introduced by lag/rolling (first few rows)
    if any(g in groups for g in ["lags_1d", "rolling_3d", "rolling_7d", "rolling_14d", "precip_memory"]):
        before = len(df)
        df.dropna(subset=[f for f in new_features if f in df.columns], inplace=True)
        if verbose:
            logging.info(f"  Feature engineering dropped {before - len(df)} NaN rows")

    if verbose:
        logging.info(f"  Feature engineering added {len(new_features)} features: {new_features}")

    return df, new_features


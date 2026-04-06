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


def calculate_soil_hydraulics_sr2006(sand, clay, organic_matter, coarse_fragments_vol_percent):
    """
    Saxton & Rawls (2006) two-step soil hydraulic property estimation.

    Parameters
    ----------
    sand, clay : float
        Fractions (0-1), NOT percentages.
    organic_matter : float
        Percentage (0-5 typical).
    coarse_fragments_vol_percent : float
        Volumetric coarse fragment percentage (0-100).

    Returns
    -------
    (theta_wp, theta_fc, theta_sat) in m3/m3
    """
    S, C, OM = sand, clay, organic_matter

    t1500t = -0.024 * S + 0.487 * C + 0.006 * OM + 0.005 * (S * OM) - 0.013 * (C * OM) + 0.068 * (S * C) + 0.031
    theta_wp = t1500t + (0.14 * t1500t - 0.02)

    t33t = -0.251 * S + 0.195 * C + 0.011 * OM + 0.006 * (S * OM) - 0.027 * (C * OM) + 0.452 * (S * C) + 0.299
    theta_fc = t33t + (1.283 * t33t**2 - 0.374 * t33t - 0.015)

    ts33t = 0.278 * S + 0.034 * C + 0.022 * OM - 0.018 * (S * OM) - 0.027 * (C * OM) - 0.584 * (S * C) + 0.078
    ts33 = ts33t + (0.636 * ts33t - 0.107)
    theta_sat = theta_fc + ts33 - 0.097 * S + 0.043

    cf_frac = coarse_fragments_vol_percent / 100.0
    return (theta_wp * (1 - cf_frac), theta_fc * (1 - cf_frac), theta_sat * (1 - cf_frac))


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
        for col_name in ["ta", "vpd", "sw_in", "precip", "rh"]:
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
            for col_name in ["ta", "vpd", "sw_in", "rh"]:
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

    # ── root_zone_swc ──────────────────────────────────────────────
    if "root_zone_swc" in groups:
        for _layer in [2, 3, 4]:
            swc_col = f"volumetric_soil_water_layer_{_layer}"
            if swc_col in df.columns:
                _vsw = df[swc_col]
                _vsw_std = _vsw.std()
                feat = f"swc_layer{_layer}_norm"
                df[feat] = _vsw / _vsw_std if _vsw_std > 1e-10 else 0.0
                new_features.append(feat)
            st_col = f"soil_temperature_level_{_layer}"
            if st_col in df.columns:
                new_features.append(st_col)

    # ── rew (Relative Extractable Water) ───────────────────────────
    if "rew" in groups:
        _sand = df["soil_sand"].iloc[0] / 100.0 if "soil_sand" in df.columns else None
        _clay = df["soil_clay"].iloc[0] / 100.0 if "soil_clay" in df.columns else None
        _soc = df["soil_soc"].iloc[0] if "soil_soc" in df.columns else 0
        _cfvo = df["soil_cfvo"].iloc[0] if "soil_cfvo" in df.columns else 0
        if _sand is not None and _clay is not None and not pd.isna(_sand) and not pd.isna(_clay):
            _om = (_soc / 10.0) * 1.72 if not pd.isna(_soc) else 0
            _cfvo = _cfvo if not pd.isna(_cfvo) else 0
            _wp, _fc, _ = calculate_soil_hydraulics_sr2006(_sand, _clay, _om, _cfvo)
            _swc_col = "volumetric_soil_water_layer_2" if "volumetric_soil_water_layer_2" in df.columns else None
            if _swc_col and (_fc - _wp) > 0.01:
                df["rew"] = ((df[_swc_col] - _wp) / (_fc - _wp)).clip(0, 1.5)
                new_features.append("rew")

    # ── et0 (FAO-56 Penman-Monteith) ──────────────────────────────
    if "et0" in groups:
        _et0_cols = ["ta", "ta_max", "ta_min", "rh", "ws", "sw_in", "ext_rad", "surface_pressure", "elevation"]
        if all(c in df.columns for c in _et0_cols):
            _T = df["ta"]
            _Tmax, _Tmin = df["ta_max"], df["ta_min"]
            _RH = df["rh"]
            _u2 = df["ws"]
            _Rs = df["sw_in"] * 0.0864
            _Ra = df["ext_rad"] * 0.0864
            _elev = df["elevation"].iloc[0]

            _P = 101.3 * ((293 - 0.0065 * _elev) / 293) ** 5.26
            _gamma = 0.000665 * _P

            _es_max = 0.6108 * np.exp(17.27 * _Tmax / (_Tmax + 237.3))
            _es_min = 0.6108 * np.exp(17.27 * _Tmin / (_Tmin + 237.3))
            _es = (_es_max + _es_min) / 2
            _es_T = 0.6108 * np.exp(17.27 * _T / (_T + 237.3))
            _ea = _es_T * _RH / 100.0

            _delta = 4098 * _es_T / (_T + 237.3) ** 2

            _Rns = 0.77 * _Rs
            _Rso = (0.75 + 2e-5 * _elev) * _Ra
            _Rs_Rso = (_Rs / _Rso.clip(lower=0.1)).clip(0, 1)
            _sigma = 4.903e-9
            _Rnl = (
                _sigma
                * ((_Tmax + 273.16) ** 4 + (_Tmin + 273.16) ** 4)
                / 2
                * (0.34 - 0.14 * np.sqrt(_ea.clip(lower=0.001)))
                * (1.35 * _Rs_Rso - 0.35)
            )
            _Rn = _Rns - _Rnl

            _num = 0.408 * _delta * _Rn + _gamma * (900 / (_T + 273)) * _u2 * (_es - _ea)
            _den = _delta + _gamma * (1 + 0.34 * _u2)
            df["et0"] = (_num / _den).clip(lower=0)
            new_features.append("et0")

    # ── psi_soil (Campbell 1974 + Cosby et al. 1984) ───────────────
    if "psi_soil" in groups:
        _sand_pct = df["soil_sand"].iloc[0] if "soil_sand" in df.columns else None
        _clay_pct = df["soil_clay"].iloc[0] if "soil_clay" in df.columns else None
        if _sand_pct is not None and _clay_pct is not None and not pd.isna(_sand_pct) and not pd.isna(_clay_pct):
            _psi_sat_cm = -(10 ** (1.88 - 0.0131 * _sand_pct))
            _b = 2.91 + 0.159 * _clay_pct
            _theta_sat = (50.5 - 0.142 * _sand_pct - 0.037 * _clay_pct) / 100

            _swc_col = "volumetric_soil_water_layer_2" if "volumetric_soil_water_layer_2" in df.columns else None
            if _swc_col and _theta_sat > 0:
                _ratio = (df[_swc_col] / _theta_sat).clip(0.01, 1.0)
                _psi_cm = _psi_sat_cm * _ratio ** (-_b)
                df["psi_soil"] = (_psi_cm * 0.000098).clip(-10, 0)
                new_features.append("psi_soil")

    # ── cwd (Cumulative Water Deficit) ─────────────────────────────
    if "cwd" in groups:
        _pet_col = "potential_evaporation_hourly_sum"
        _precip_col = "total_precipitation_hourly_sum"
        if _pet_col in df.columns and _precip_col in df.columns:
            _pet_mm = -df[_pet_col] * 1000
            _precip_mm = df[_precip_col] * 1000
            _deficit = _pet_mm - _precip_mm
            _cwd_vals = []
            _running = 0.0
            for _d in _deficit:
                if pd.isna(_d):
                    _cwd_vals.append(np.nan)
                else:
                    _running = max(0.0, _running + _d)
                    _cwd_vals.append(_running)
            df["cwd"] = _cwd_vals
            new_features.append("cwd")

    # Drop NaNs introduced by lag/rolling (first few rows)
    if any(g in groups for g in ["lags_1d", "rolling_3d", "rolling_7d", "rolling_14d", "precip_memory"]):
        before = len(df)
        df.dropna(subset=[f for f in new_features if f in df.columns], inplace=True)
        if verbose:
            logging.info(f"  Feature engineering dropped {before - len(df)} NaN rows")

    if verbose:
        logging.info(f"  Feature engineering added {len(new_features)} features: {new_features}")

    return df, new_features

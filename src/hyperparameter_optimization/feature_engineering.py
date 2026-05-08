"""
Feature engineering functions for sap velocity prediction.

Extracted from test_hyperparameter_tuning_ML_spatial_stratified.py for reuse
across training, prediction, and standalone SHAP analysis scripts.
"""

import logging

import numpy as np
import pandas as pd


def compute_canopy_conductance(
    sap_velocity: pd.Series,
    vpd: pd.Series,
    temperature: pd.Series,
    elevation: float = 0.0,
    vpd_min: float = 0.3,
) -> pd.Series:
    """Compute canopy conductance per unit sapwood area from sap velocity.

    Full Köstner / Phillips & Oren (1998) / Flo et al. (2021, 2022) formula:

        G_Asw (mol m⁻² s⁻¹) = K_G × SFD × n_air / VPD

    where K_G = 115.8 + 0.4236 × T (kPa m³ kg⁻¹) lumps the temperature-
    dependent psychrometric constant, latent heat, and specific heat
    (Phillips & Oren 1998), and n_air corrects molar air density for
    local temperature and altitude.

    Parameters
    ----------
    sap_velocity : pd.Series
        Sap flux density in cm³ cm⁻² h⁻¹ (SAPFLUXNET standard).
    vpd : pd.Series
        Vapour pressure deficit in kPa.
    temperature : pd.Series
        Air temperature in °C.
    elevation : float
        Site altitude in metres (default 0 = sea level).
    vpd_min : float
        Minimum VPD threshold (kPa). Rows below produce NaN.

    Returns
    -------
    pd.Series
        Canopy conductance G_Asw in mol m⁻²_sapwood s⁻¹.
        NaN where VPD < vpd_min or inputs are NaN.
    """
    # --- unit conversion: cm³ cm⁻² h⁻¹  →  kg m⁻² s⁻¹ ---
    # 1 cm³ cm⁻² h⁻¹ = 1 cm h⁻¹ = 0.01 m h⁻¹
    # × 1000 kg m⁻³ (ρ_water) / 3600 s h⁻¹ = 10 / 3600 ≈ 0.002778
    _SFD_CONV = 10.0 / 3600.0  # cm³ cm⁻² h⁻¹ → kg m⁻² s⁻¹
    sfd_kg = sap_velocity * _SFD_CONV

    # --- K_G: temperature-dependent coefficient (kPa m³ kg⁻¹) ---
    # Phillips & Oren (1998); absorbs λ(T), γ(T), ρ_a(T), c_p
    k_g = 115.8 + 0.4236 * temperature

    # --- n_air: molar density of air (mol m⁻³) ---
    # Standard molar density corrected for T and altitude
    _ETA = 44.6  # mol m⁻³ at STP (0 °C, 101.325 kPa)
    _T0 = 273.0  # K
    n_air = _ETA * (_T0 / (_T0 + temperature)) * np.exp(-0.00012 * elevation)

    # --- VPD filter ---
    vpd_safe = vpd.where(vpd >= vpd_min)

    # --- G_Asw (mol m⁻² s⁻¹) = K_G × SFD_kg × n_air / VPD ---
    g_sw = k_g * sfd_kg * n_air / vpd_safe

    return g_sw


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

    # ── lags ───────────────────────────────────────────────────────
    if "lags_1d" in groups:
        lag_cols = ["ta", "vpd", "sw_in", "precip", "rh"]
        if time_scale == "hourly":
            for col_name in lag_cols:
                if col_name in df.columns:
                    for lag_h in [1, 3, 6, 12, 24]:
                        feat = f"{col_name}_lag{lag_h}h"
                        df[feat] = df[col_name].shift(lag_h)
                        new_features.append(feat)
        else:
            for col_name in lag_cols:
                if col_name in df.columns:
                    feat = f"{col_name}_lag1d"
                    df[feat] = df[col_name].shift(1)
                    new_features.append(feat)

    # ── rolling statistics ──────────────────────────────────────────
    rolling_cols = ["ta", "vpd", "sw_in", "rh"]
    if time_scale == "hourly":
        # Hourly: 3h, 6h, 12h, 24h windows (triggered by any rolling group)
        if any(g in groups for g in ["rolling_3d", "rolling_7d", "rolling_14d"]):
            for col_name in rolling_cols:
                if col_name in df.columns:
                    for window in [3, 6, 12, 24]:
                        mn = f"{col_name}_roll{window}h_mean"
                        sd = f"{col_name}_roll{window}h_std"
                        df[mn] = df[col_name].rolling(window, min_periods=1).mean()
                        df[sd] = df[col_name].rolling(window, min_periods=2).std()
                        new_features.extend([mn, sd])
    else:
        # Daily: 3d, 7d, 14d windows
        rolling_groups_map = {"rolling_3d": 3, "rolling_7d": 7, "rolling_14d": 14}
        for grp, window in rolling_groups_map.items():
            if grp in groups:
                for col_name in rolling_cols:
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
            rain_events = df["precip"] > 0.5
            no_rain = ~rain_events
            if time_scale == "hourly":
                df["precip_sum_24h"] = df["precip"].rolling(24, min_periods=1).sum()
                df["precip_sum_72h"] = df["precip"].rolling(72, min_periods=1).sum()
                df["hours_since_rain"] = no_rain.groupby(rain_events.cumsum()).cumcount().astype(np.float32)
                new_features.extend(["precip_sum_24h", "precip_sum_72h", "hours_since_rain"])
            else:
                df["precip_sum_3d"] = df["precip"].rolling(3, min_periods=1).sum()
                df["precip_sum_7d"] = df["precip"].rolling(7, min_periods=1).sum()
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
                _vsw_mean = _vsw.mean()
                _vsw_std = _vsw.std()
                feat = f"swc_layer{_layer}_norm"
                df[feat] = (_vsw - _vsw_mean) / _vsw_std if _vsw_std > 1e-10 else 0.0
                new_features.append(feat)
            st_col = f"soil_temperature_level_{_layer}"
            if st_col in df.columns:
                new_features.append(st_col)

    # ── rew (Relative Extractable Water) ───────────────────────────
    # Uses merge's pre-computed Saxton-Rawls θ_wp/θ_fc (soil_theta_wp/fc)
    # and RAW SWC (volumetric_soil_water_layer_2_raw) to avoid using the
    # variance-normalised column that breaks the physical formula.
    if "rew" in groups:
        _wp = df["soil_theta_wp"].iloc[0] if "soil_theta_wp" in df.columns else None
        _fc = df["soil_theta_fc"].iloc[0] if "soil_theta_fc" in df.columns else None
        if _wp is not None and _fc is not None and not pd.isna(_wp) and not pd.isna(_fc):
            # Prefer _raw column (actual m³/m³); fall back to main column
            _swc_col = next(
                (c for c in ["volumetric_soil_water_layer_2_raw", "volumetric_soil_water_layer_2"] if c in df.columns),
                None,
            )
            if _swc_col and (_fc - _wp) > 0.01:
                df["rew"] = ((df[_swc_col] - _wp) / (_fc - _wp)).clip(0, 1.5)
                new_features.append("rew")

    # ── et0 (FAO-56 Penman-Monteith) ────────────────────────────────
    # Daily: Allen et al. 1998 Eq. 6  (Cn=900, Cd=0.34, G≈0)
    # Hourly: Allen et al. 1998 Eq. 53 (Cn=37, Cd=0.24 day/0.96 night)
    if "et0" in groups:
        _is_daily = time_scale == "daily"
        _et0_base = ["ta", "rh", "ws", "sw_in", "ext_rad", "elevation"]
        _et0_cols = _et0_base + (["ta_max", "ta_min"] if _is_daily else [])
        if all(c in df.columns for c in _et0_cols):
            _T = df["ta"]
            _u2 = df["ws"]
            _elev = df["elevation"].iloc[0]

            # Scale-aware radiation conversion (W/m² → MJ/m²/period)
            _rad_factor = 0.0864 if _is_daily else 0.0036  # /day vs /hour
            _Rs = df["sw_in"] * _rad_factor
            _Ra = df["ext_rad"] * _rad_factor

            _P = 101.3 * ((293 - 0.0065 * _elev) / 293) ** 5.26
            _gamma = 0.000665 * _P

            _es_T = 0.6108 * np.exp(17.27 * _T / (_T + 237.3))

            if _is_daily:
                _Tmax, _Tmin = df["ta_max"], df["ta_min"]
                _es_max = 0.6108 * np.exp(17.27 * _Tmax / (_Tmax + 237.3))
                _es_min = 0.6108 * np.exp(17.27 * _Tmin / (_Tmin + 237.3))
                _es = (_es_max + _es_min) / 2

                # Actual vapour pressure: FAO-56 Eq. 17 when RH extremes
                # available, else Eq. 19 fallback.
                if "rh_max" in df.columns and "rh_min" in df.columns:
                    _RH_max = df["rh_max"].clip(0, 100)
                    _RH_min = df["rh_min"].clip(0, 100)
                    _ea = (_es_min * _RH_max / 100.0 + _es_max * _RH_min / 100.0) / 2
                else:
                    _ea = _es_T * df["rh"] / 100.0
            else:
                # Hourly: es = e°(T), ea = e°(T) × RH/100
                _es = _es_T
                _ea = _es_T * df["rh"] / 100.0

            _delta = 4098 * _es_T / (_T + 237.3) ** 2

            # Net radiation
            _Rns = 0.77 * _Rs
            _Rso = (0.75 + 2e-5 * _elev) * _Ra

            if _is_daily:
                _Tmax, _Tmin = df["ta_max"], df["ta_min"]
                _Rs_Rso = (_Rs / _Rso.clip(lower=0.1)).clip(0, 1)
                _sigma = 4.903e-9  # MJ/K⁴/m²/day
                _Rnl = (
                    _sigma
                    * ((_Tmax + 273.16) ** 4 + (_Tmin + 273.16) ** 4)
                    / 2
                    * (0.34 - 0.14 * np.sqrt(_ea.clip(lower=0.001)))
                    * (1.35 * _Rs_Rso - 0.35)
                )
            else:
                _Rs_Rso = (_Rs / _Rso.clip(lower=0.001)).clip(0, 1)
                _sigma_h = 2.042e-10  # MJ/K⁴/m²/hour  (σ/24)
                _Rnl = (
                    _sigma_h
                    * ((_T + 273.16) ** 4)
                    * (0.34 - 0.14 * np.sqrt(_ea.clip(lower=0.001)))
                    * (1.35 * _Rs_Rso - 0.35)
                )
            _Rn = _Rns - _Rnl

            # Soil heat flux G
            if _is_daily:
                _G = 0.0  # FAO-56: G ≈ 0 for daily
            else:
                # FAO-56 Eq. 45/46: G = 0.1 Rn (day), G = 0.5 Rn (night)
                _is_day = df["sw_in"] > 10
                _G = np.where(_is_day, 0.1 * _Rn, 0.5 * _Rn)

            # Wind function constants (FAO-56 Table 1 / Eq. 53)
            _Cn = 900 if _is_daily else 37
            if _is_daily:
                _Cd = 0.34
            else:
                _Cd = np.where(df["sw_in"] > 10, 0.24, 0.96)

            _num = 0.408 * _delta * (_Rn - _G) + _gamma * (_Cn / (_T + 273)) * _u2 * (_es - _ea)
            _den = _delta + _gamma * (1 + _Cd * _u2)
            df["et0"] = (_num / _den).clip(lower=0)
            new_features.append("et0")

    # ── psi_soil (Campbell 1974 + Cosby et al. 1984) ───────────────
    # Cosby parameters (ψ_sat, b, θ_sat) are a self-consistent set —
    # do NOT mix with Saxton-Rawls θ_sat.  Uses RAW SWC (_raw column)
    # to avoid the variance-normalised main column.
    if "psi_soil" in groups:
        _sand_pct = df["soil_sand"].iloc[0] if "soil_sand" in df.columns else None
        _clay_pct = df["soil_clay"].iloc[0] if "soil_clay" in df.columns else None
        if _sand_pct is not None and _clay_pct is not None and not pd.isna(_sand_pct) and not pd.isna(_clay_pct):
            _psi_sat_cm = -(10 ** (1.88 - 0.0131 * _sand_pct))
            _b = 2.91 + 0.159 * _clay_pct
            _theta_sat = (50.5 - 0.142 * _sand_pct - 0.037 * _clay_pct) / 100

            # Prefer _raw column (actual m³/m³); fall back to main column
            _swc_col = next(
                (c for c in ["volumetric_soil_water_layer_2_raw", "volumetric_soil_water_layer_2"] if c in df.columns),
                None,
            )
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

    # ── soil_hydraulics_extended ──────────────────────────────────────
    # Uses merge's pre-computed Saxton-Rawls θ_wp/θ_fc and raw SWC.
    if "soil_hydraulics_extended" in groups:
        _wp = df["soil_theta_wp"].iloc[0] if "soil_theta_wp" in df.columns else None
        _fc = df["soil_theta_fc"].iloc[0] if "soil_theta_fc" in df.columns else None

        if _wp is not None and _fc is not None and not pd.isna(_wp) and not pd.isna(_fc):
            # AWC: available water capacity (mm per metre of soil)
            df["awc"] = (_fc - _wp) * 1000.0
            new_features.append("awc")

            # Available water & deficit require RAW SWC (m³/m³).
            # The variance-normalised column (x/σ) is NOT in physical units
            # and would produce garbage values when subtracted from θ_wp/θ_fc.
            _swc_raw = (
                "volumetric_soil_water_layer_2_raw" if "volumetric_soil_water_layer_2_raw" in df.columns else None
            )
            if _swc_raw:
                # Available water above wilting point (mm/m)
                df["available_water"] = ((df[_swc_raw] - _wp) * 1000.0).clip(lower=0)
                # Soil water deficit below field capacity (mm/m)
                df["soil_water_deficit"] = ((_fc - df[_swc_raw]) * 1000.0).clip(lower=0)
                new_features.extend(["available_water", "soil_water_deficit"])

        # Root-zone weighted SWC (ERA5-Land: 0-7, 7-28, 28-100, 100-289 cm)
        # Weights approximate root density distribution (Jackson et al. 1996)
        _layer_weights = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}
        _wsum = None
        _wtotal = 0.0
        for _lyr, _w in _layer_weights.items():
            _col = f"volumetric_soil_water_layer_{_lyr}_raw"
            if _col not in df.columns:
                _col = f"volumetric_soil_water_layer_{_lyr}"
            if _col in df.columns:
                _wsum = df[_col] * _w if _wsum is None else _wsum + df[_col] * _w
                _wtotal += _w
        if _wsum is not None and _wtotal > 0:
            df["root_zone_swc_weighted"] = _wsum / _wtotal
            new_features.append("root_zone_swc_weighted")

        # Soil temperature gradient (surface minus deep → heat flux direction)
        if "soil_temperature_level_1" in df.columns and "soil_temperature_level_4" in df.columns:
            df["soil_temp_gradient"] = df["soil_temperature_level_1"] - df["soil_temperature_level_4"]
            new_features.append("soil_temp_gradient")

        # Soil frozen indicator (auto-detect K vs °C: ERA5-Land native = K)
        if "soil_temperature_level_1" in df.columns:
            _st1 = df["soil_temperature_level_1"]
            _thresh = 273.15 if _st1.median() > 100 else 0.0
            df["soil_frozen"] = (_st1 < _thresh).astype(np.float32)
            new_features.append("soil_frozen")

    # ── atm_demand_extended ───────────────────────────────────────────
    if "atm_demand_extended" in groups:
        _Rn_computed = None

        # Step 1: Net radiation (Rn = Rns - Rnl, FAO-56 Eq. 38-39)
        # Requires rh for actual vapour pressure ea in Rnl.
        if all(c in df.columns for c in ["sw_in", "ext_rad", "ta", "rh"]):
            _elev = df["elevation"].iloc[0] if "elevation" in df.columns else 0.0
            _rad_factor = 0.0864 if time_scale == "daily" else 0.0036
            _Rns = 0.77 * df["sw_in"] * _rad_factor  # FAO-56 Eq. 38
            _Rso = (0.75 + 2e-5 * _elev) * df["ext_rad"] * _rad_factor
            _Rs_Rso = (df["sw_in"] * _rad_factor / _Rso.clip(lower=0.1)).clip(0, 1)

            _ea = 0.6108 * np.exp(17.27 * df["ta"] / (df["ta"] + 237.3)) * df["rh"] / 100.0

            # Rnl: daily uses (Tmax_K⁴ + Tmin_K⁴)/2, hourly uses T_K⁴
            if time_scale == "daily" and "ta_max" in df.columns and "ta_min" in df.columns:
                _T_K4 = ((df["ta_max"] + 273.16) ** 4 + (df["ta_min"] + 273.16) ** 4) / 2
                _sigma = 4.903e-9  # MJ K⁻⁴ m⁻² day⁻¹
            else:
                _T_K4 = (df["ta"] + 273.16) ** 4
                _sigma = 4.903e-9 if time_scale == "daily" else 2.042e-10

            _Rnl = _sigma * _T_K4 * (0.34 - 0.14 * np.sqrt(_ea.clip(lower=0.001))) * (1.35 * _Rs_Rso - 0.35)
            _Rn_computed = _Rns - _Rnl
            df["net_radiation"] = _Rn_computed
            new_features.append("net_radiation")

        # Step 2: Priestley-Taylor PET using full Rn
        # PT: ET = 0.408 × α × [Δ/(Δ+γ)] × (Rn - G)
        # 0.408 = 1/λ converts MJ/m² → mm (λ ≈ 2.45 MJ/kg)
        # α = 1.26 (Priestley & Taylor, 1972)
        if _Rn_computed is not None and "elevation" in df.columns:
            _T = df["ta"]
            _elev = df["elevation"].iloc[0]
            _P = 101.3 * ((293 - 0.0065 * _elev) / 293) ** 5.26
            _gamma = 0.000665 * _P
            _es = 0.6108 * np.exp(17.27 * _T / (_T + 237.3))
            _delta = 4098 * _es / (_T + 237.3) ** 2

            if time_scale == "daily":
                _G = 0.0  # FAO-56: G ≈ 0 for daily
            else:
                _is_day = df["sw_in"] > 10
                _G = np.where(_is_day, 0.1 * _Rn_computed, 0.5 * _Rn_computed)

            df["priestley_taylor_pet"] = (0.408 * 1.26 * (_delta / (_delta + _gamma)) * (_Rn_computed - _G)).clip(
                lower=0
            )
            new_features.append("priestley_taylor_pet")

        # Step 3: Diurnal ranges (daily only)
        if time_scale == "daily":
            if "ta_max" in df.columns and "ta_min" in df.columns:
                df["diurnal_temp_range"] = df["ta_max"] - df["ta_min"]
                new_features.append("diurnal_temp_range")
            if "vpd_max" in df.columns and "vpd_min" in df.columns:
                df["vpd_diurnal_range"] = df["vpd_max"] - df["vpd_min"]
                new_features.append("vpd_diurnal_range")

    # ── plant_hydraulics ──────────────────────────────────────────────
    if "plant_hydraulics" in groups:
        if "LAI" in df.columns:
            # fAPAR — Beer-Lambert with k=0.5 (Monsi & Saeki 1953)
            df["fAPAR"] = 1.0 - np.exp(-0.5 * df["LAI"])
            new_features.append("fAPAR")
            # LAI change rate (phenology signal)
            df["lai_change_rate"] = df["LAI"].diff(1)
            new_features.append("lai_change_rate")

        if "sw_in" in df.columns and "LAI" in df.columns:
            # Light per unit leaf area
            df["radiation_per_leaf"] = df["sw_in"] / (df["LAI"] + 0.01)
            new_features.append("radiation_per_leaf")

    # ── swc_memory ────────────────────────────────────────────────────
    if "swc_memory" in groups:
        _swc_col = next(
            (c for c in ["volumetric_soil_water_layer_1"] if c in df.columns),
            None,
        )
        if _swc_col:
            if time_scale == "hourly":
                for _lag, _name in [(1, "swc_lag1h"), (6, "swc_lag6h"), (24, "swc_lag24h")]:
                    df[_name] = df[_swc_col].shift(_lag)
                    new_features.append(_name)
                df["swc_change_1h"] = df[_swc_col].diff(1)
                new_features.append("swc_change_1h")
            else:
                for _lag, _name in [(1, "swc_lag1d"), (3, "swc_lag3d"), (7, "swc_lag7d")]:
                    df[_name] = df[_swc_col].shift(_lag)
                    new_features.append(_name)
                df["swc_change_1d"] = df[_swc_col].diff(1)
                new_features.append("swc_change_1d")

    # ── temporal_anomalies ────────────────────────────────────────────
    # Deviation from running mean captures short-term departures from
    # seasonal baseline — important for drought / heatwave detection.
    if "temporal_anomalies" in groups:
        _anom_window = 720 if time_scale == "hourly" else 30  # ~30 days
        _min_periods = max(1, _anom_window // 4)

        if "ta" in df.columns:
            df["ta_anomaly"] = df["ta"] - df["ta"].rolling(_anom_window, min_periods=_min_periods).mean()
            new_features.append("ta_anomaly")
        if "vpd" in df.columns:
            df["vpd_anomaly"] = df["vpd"] - df["vpd"].rolling(_anom_window, min_periods=_min_periods).mean()
            new_features.append("vpd_anomaly")

        _swc_col = next(
            (c for c in ["volumetric_soil_water_layer_1"] if c in df.columns),
            None,
        )
        if _swc_col:
            df["swc_anomaly"] = df[_swc_col] - df[_swc_col].rolling(_anom_window, min_periods=_min_periods).mean()
            new_features.append("swc_anomaly")

        # Cumulative GDD (does not reset annually — acceptable for ML
        # since Year sin/cos encode seasonality separately)
        if "ta" in df.columns:
            _gdd = df.get("gdd", (df["ta"] - 5).clip(lower=0))
            df["cumulative_gdd"] = _gdd.cumsum()
            new_features.append("cumulative_gdd")

    # ── cross_interactions (SPAC cross-terms) ─────────────────────────
    # Must come AFTER rew/et0 groups so those columns exist in df.
    if "cross_interactions" in groups:
        _swc_col = next(
            (c for c in ["volumetric_soil_water_layer_1"] if c in df.columns),
            None,
        )
        if "vpd" in df.columns and _swc_col:
            df["vpd_x_swc"] = df["vpd"] * df[_swc_col]
            new_features.append("vpd_x_swc")
        if "vpd" in df.columns and "rew" in df.columns:
            df["vpd_x_rew"] = df["vpd"] * df["rew"]
            new_features.append("vpd_x_rew")
        if "LAI" in df.columns and "vpd" in df.columns:
            df["lai_x_vpd"] = df["LAI"] * df["vpd"]
            new_features.append("lai_x_vpd")
        if "LAI" in df.columns and "sw_in" in df.columns:
            df["lai_x_sw_in"] = df["LAI"] * df["sw_in"]
            new_features.append("lai_x_sw_in")
        if "ta" in df.columns and _swc_col:
            df["ta_x_swc"] = df["ta"] * df[_swc_col]
            new_features.append("ta_x_swc")
        if "et0" in df.columns and "rew" in df.columns:
            df["et0_x_rew"] = df["et0"] * df["rew"]
            new_features.append("et0_x_rew")

    # ── bioclimatic ───────────────────────────────────────────────────
    if "bioclimatic" in groups:
        # De Martonne Aridity Index: MAP / (MAT + 10)
        # De Martonne (1926), La Météorologie 2:449-458
        if "mean_annual_precip" in df.columns and "mean_annual_temp" in df.columns:
            _mat = df["mean_annual_temp"].iloc[0]
            _map_val = df["mean_annual_precip"].iloc[0]
            if not pd.isna(_mat) and not pd.isna(_map_val) and (_mat + 10) > 0:
                df["de_martonne_aridity"] = _map_val / (_mat + 10)
                new_features.append("de_martonne_aridity")

    # ── tree_metadata (driver analysis — plant-level features) ─────
    if "tree_metadata" in groups:
        for _col in ["pl_dbh"]:
            if _col in df.columns and not df[_col].isna().all():
                new_features.append(_col)

        if "pl_sens_meth" in df.columns:
            _all_meths = ["HD", "CHP", "HR", "TSHB", "CHD", "HPTM", "HFD", "SHB"]
            for _m in _all_meths:
                df[f"meth_{_m}"] = (df["pl_sens_meth"] == _m).astype(np.float32)
            new_features.extend([f"meth_{_m}" for _m in _all_meths])

        if "pl_species" in df.columns:
            _all_genera = [
                "Pinus",
                "Acer",
                "Eucalyptus",
                "Picea",
                "Populus",
                "Fagus",
                "Abies",
                "Quercus",
                "Olea",
                "Larix",
                "Juniperus",
                "Acacia",
                "Malus",
                "Betula",
                "Fraxinus",
                "Nothofagus",
                "Tectona",
                "Elaeis",
                "Hevea",
                "Dicorynia",
                "Mangifera",
                "Vitellaria",
            ]
            _genus_series = (
                df["pl_species"]
                .fillna("Unknown")
                .apply(lambda x: x.split()[0] if isinstance(x, str) and " " in x else str(x))
            )
            for _g in _all_genera:
                df[f"genus_{_g}"] = (_genus_series == _g).astype(np.float32)
            df["genus_Other"] = (~_genus_series.isin(_all_genera)).astype(np.float32)
            new_features.extend([f"genus_{_g}" for _g in _all_genera] + ["genus_Other"])

    if verbose:
        logging.info(f"  Feature engineering added {len(new_features)} features: {new_features}")

    return df, new_features


# All feature engineering groups — single source of truth.
ALL_FEATURE_GROUPS: list[str] = [
    "interactions",
    "lags_1d",
    "rolling_3d",
    "rolling_7d",
    "rolling_14d",
    "physics",
    "precip_memory",
    "indicators",
    "root_zone_swc",
    "rew",
    "et0",
    "psi_soil",
    "cwd",
    "soil_hydraulics_extended",
    "atm_demand_extended",
    "plant_hydraulics",
    "swc_memory",
    "temporal_anomalies",
    # cross_interactions must be AFTER rew/et0 (uses their output columns)
    "cross_interactions",
    "bioclimatic",
]


def apply_all_feature_engineering(
    df: pd.DataFrame, time_scale: str = "daily", verbose: bool = False
) -> tuple[pd.DataFrame, list[str]]:
    """Apply ALL feature engineering unconditionally.

    Produces a symmetric feature set for both daily and hourly time scales:
    same feature names, window sizes adapted internally.  Only ``is_daytime``
    is hourly-exclusive.

    Parameters
    ----------
    df : pd.DataFrame
        Site data with raw columns and TIMESTAMP index.
    time_scale : str
        'daily' or 'hourly'.
    verbose : bool

    Returns
    -------
    (df, new_feature_names) : tuple
    """
    is_hourly = time_scale == "hourly"

    # ── 1. All configurable groups (interactions, rolling, lags, eco-hydro …) ──
    df, new_features = apply_feature_engineering(df, ALL_FEATURE_GROUPS, time_scale, verbose=False)

    # ── 2. Derived scalar features (both scales) ─────────────────────────────
    if "vpd" in df.columns:
        df["vpd_log"] = np.log1p(df["vpd"])
        new_features.append("vpd_log")

    if "rh" in df.columns and "ta" in df.columns:
        a, b = 17.27, 237.3  # Tetens (consistent with VPD / ET0 formulas)
        alpha = (a * df["ta"] / (b + df["ta"])) + np.log(df["rh"] / 100 + 0.01)
        df["dew_point"] = b * alpha / (a - alpha)
        df["dew_point_depression"] = df["ta"] - df["dew_point"]
        new_features.extend(["dew_point", "dew_point_depression"])

    if "latitude" in df.columns:
        df["tropical"] = (abs(df["latitude"]) < 23.5).astype(np.float32)
        df["boreal"] = (abs(df["latitude"]) > 55).astype(np.float32)
        df["southern_hemisphere"] = (df["latitude"] < 0).astype(np.float32)
        new_features.extend(["tropical", "boreal", "southern_hemisphere"])

    # ── 3. Temporal derivative — scale-aware naming ────────────────────────────
    if "ta" in df.columns:
        suffix = "1h" if is_hourly else "1d"
        feat = f"ta_change_{suffix}"
        df[feat] = df["ta"].diff(1)
        new_features.append(feat)

    # ── 4. Cumulative sums — scale-aware window + naming ─────────────────────
    if "sw_in" in df.columns:
        w, suffix = (24, "24h") if is_hourly else (7, "7d")
        feat = f"sw_in_cumsum_{suffix}"
        df[feat] = df["sw_in"].rolling(w, min_periods=1).sum()
        new_features.append(feat)

    if "vpd" in df.columns:
        w, suffix = (6, "6h") if is_hourly else (3, "3d")
        feat = f"vpd_cumsum_{suffix}"
        df[feat] = df["vpd"].rolling(w, min_periods=1).sum()
        new_features.append(feat)

    # ── 5. VPD change — scale-aware ─────────────────────────────────────────
    if "vpd" in df.columns:
        suffix = "1h" if is_hourly else "1d"
        feat = f"vpd_change_{suffix}"
        df[feat] = df["vpd"].diff(1)
        new_features.append(feat)

    # ── 6. Soil-atmosphere temperature difference ────────────────────────────
    # Auto-detect soil temp units (K vs °C) since GEE path keeps Kelvin,
    # CDS prediction path converts to Celsius.
    if "ta" in df.columns and "soil_temperature_level_1" in df.columns:
        _st1 = df["soil_temperature_level_1"]
        if _st1.median() > 100:
            # Soil temp in Kelvin → convert ta to Kelvin for consistent subtraction
            df["soil_atm_temp_diff"] = (df["ta"] + 273.15) - _st1
        else:
            # Both in Celsius
            df["soil_atm_temp_diff"] = df["ta"] - _st1
        new_features.append("soil_atm_temp_diff")

    # ── 7. Hourly-only: is_daytime ───────────────────────────────────────────
    if is_hourly and "sw_in" in df.columns:
        df["is_daytime"] = (df["sw_in"] > 10).astype(np.float32)
        new_features.append("is_daytime")

    if verbose:
        logging.info("apply_all_feature_engineering (%s) added %d features", time_scale, len(new_features))

    return df, new_features

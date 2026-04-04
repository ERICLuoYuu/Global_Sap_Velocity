import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

# Assuming path_config is in your local environment
from path_config import get_default_paths

paths = get_default_paths()


def calculate_soil_hydraulics(sand, clay, organic_matter, coarse_fragments_vol_percent):
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
    (theta_wp, theta_fc, theta_sat) : tuple of float
        Volumetric water content (m³/m³) at wilting point, field capacity,
        and saturation, adjusted for coarse fragments.
    """
    S, C, OM = sand, clay, organic_matter

    # 1. Wilting point (1500 kPa) — first estimate then adjustment
    t1500t = -0.024 * S + 0.487 * C + 0.006 * OM + 0.005 * (S * OM) - 0.013 * (C * OM) + 0.068 * (S * C) + 0.031
    theta_wp = t1500t + (0.14 * t1500t - 0.02)

    # 2. Field capacity (33 kPa) — first estimate then adjustment
    #    Correction uses t33t itself, NOT theta_wp
    t33t = -0.251 * S + 0.195 * C + 0.011 * OM + 0.006 * (S * OM) - 0.027 * (C * OM) + 0.452 * (S * C) + 0.299
    theta_fc = t33t + (1.283 * t33t**2 - 0.374 * t33t - 0.015)

    # 3. Saturation (porosity) — three steps
    ts33t = 0.278 * S + 0.034 * C + 0.022 * OM - 0.018 * (S * OM) - 0.027 * (C * OM) - 0.584 * (S * C) + 0.078
    ts33 = ts33t + (0.636 * ts33t - 0.107)
    theta_sat = theta_fc + ts33 - 0.097 * S + 0.043

    # Adjust for coarse fragments (rocks reduce pore volume)
    cf_frac = coarse_fragments_vol_percent / 100.0
    return (
        theta_wp * (1 - cf_frac),
        theta_fc * (1 - cf_frac),
        theta_sat * (1 - cf_frac),
    )


def get_weighted_soil_props(soil_row, depths=["0_5", "5_15", "15_30", "30_60", "60_100"]):
    """
    Calculate thickness-weighted average for soil properties down to 100cm.
    """
    thickness_map = {"0_5": 5, "5_15": 10, "15_30": 15, "30_60": 30, "60_100": 40}
    properties = ["sand", "clay", "soc", "bdod", "cfvo"]
    weighted_sums = {p: 0.0 for p in properties}
    total_thickness = 0.0

    for depth in depths:
        thick = thickness_map.get(depth, 0)
        if f"sand_{depth}" not in soil_row.index:
            continue
        if pd.isna(soil_row.get(f"sand_{depth}")):
            continue

        total_thickness += thick
        for p in properties:
            val = soil_row.get(f"{p}_{depth}", 0)
            if pd.isna(val):
                val = 0
            weighted_sums[p] += val * thick

    if total_thickness == 0:
        return {p: np.nan for p in properties}

    return {p: weighted_sums[p] / total_thickness for p in properties}


def calculate_daytime_mask(timestamps, lat, lon, elevation=0, elevation_threshold=0):
    """
    Calculate a boolean mask indicating daytime based on solar elevation.

    Args:
        timestamps: DatetimeIndex or datetime column
        lat: Latitude
        lon: Longitude
        elevation: Site elevation in meters
        elevation_threshold: Solar elevation angle threshold (0 = horizon)

    Returns:
        Boolean Series (True = daytime, False = nighttime)
    """
    if not isinstance(timestamps, pd.DatetimeIndex):
        timestamps = pd.DatetimeIndex(timestamps)

    if timestamps.tzinfo is None:
        timestamps = timestamps.tz_localize("UTC")

    solar_position = pvlib.solarposition.get_solarposition(timestamps, lat, lon, altitude=elevation)
    return solar_position["elevation"] > elevation_threshold


def load_stand_age_map(base_path):
    """
    Load the global stand age CSV and create a dictionary mapping site_name -> mean_age.
    """
    age_file_path = base_path / "extracted_data" / "terrain_site_data" / "sapwood" / "stand_age_data.csv"

    if not age_file_path.exists():
        age_file_path = (
            paths.base_data_dir / "raw" / "extracted_data" / "terrain_site_data" / "sapwood" / "stand_age_data.csv"
        )

    if age_file_path.exists():
        print(f"Loading Stand Age data from: {age_file_path}")
        df = pd.read_csv(age_file_path)
        tc_cols = [c for c in df.columns if "stand_age_TC" in c]
        if tc_cols:
            df["mean_age"] = df[tc_cols].mean(axis=1)
            return df.set_index("site_name")["mean_age"].to_dict()

    print(f"WARNING: Stand Age file not found at {age_file_path}")
    return {}


def merge_sap_env_data_site(output_base_dir, daytime_only=False, plant_level=False):
    """
    Merge sap flow and environmental data for all sites.

    Args:
        output_base_dir: Base directory for output files
        daytime_only: If True, filter to daytime data only before daily aggregation
        plant_level: If True, output one row per tree per timestamp (long format)
                     instead of site-averaged sap velocity. Joins plant_md.csv
                     to add tree metadata (pl_dbh, pl_species, pl_sens_meth, etc.).
    """
    if daytime_only:
        print("\n" + "=" * 60)
        print("DAYTIME-ONLY MODE ENABLED")
        print("Nighttime data will be filtered out before daily aggregation")
        print("=" * 60 + "\n")

    if plant_level:
        print("\n" + "=" * 60)
        print("PLANT-LEVEL MODE ENABLED")
        print("Output: one row per tree per timestamp (long format)")
        print("Tree metadata from plant_md.csv will be joined")
        print("=" * 60 + "\n")

    # --- SETUP DIRECTORIES ---
    if not output_base_dir.exists():
        output_base_dir.mkdir(parents=True, exist_ok=True)

    hourly_out_dir = output_base_dir / "hourly"
    hourly_out_dir.mkdir(exist_ok=True)

    daily_out_dir = output_base_dir / "daily"
    daily_out_dir.mkdir(exist_ok=True)

    # Collections for final merge
    biome_merged_hourly = {}
    biome_merged_daily = {}

    all_pft_types = set()
    site_biome_mapping = {}

    # --- LOAD STATIC DATASETS ---
    site_info = pd.read_csv(paths.env_extracted_data_path)

    soil_file_path = paths.terrain_attributes_data_path.parent / "soilgrids_data.csv"
    if soil_file_path.exists():
        soil_data = pd.read_csv(soil_file_path)
        print("Loaded SoilGrids data.")
    else:
        print("WARNING: SoilGrids data not found.")
        soil_data = pd.DataFrame()

    stand_age_map = load_stand_age_map(paths.raw_csv_dir.parent)

    era5_data = pd.read_csv(paths.era5_discrete_data_path)
    # Avoid division by zero
    pet = np.abs(era5_data["potential_evaporation_hourly"].values)
    precip = era5_data["total_precipitation_hourly"].values
    # Avoid division by zero
    pet = np.where(pet < 1e-10, 1e-10, pet)
    precip_PET = np.clip(precip / pet, 0, 10)
    era5_data["prcip/PET"] = precip_PET
    lai_data = pd.read_csv(paths.globmap_lai_data_path)
    pft_data = pd.read_csv(paths.pft_data_path)

    pft_mapping = {
        1: "Evergreen Needleleaf Forests",
        2: "Evergreen Broadleaf Forests",
        3: "Deciduous Needleleaf Forests",
        4: "Deciduous Broadleaf Forests",
        5: "Mixed Forests",
        6: "Closed Shrublands",
        7: "Open Shrublands",
        8: "Woody Savannas",
        9: "Savannas",
        10: "Grasslands",
        11: "Permanent Wetlands",
        12: "Cropland",
        13: "Urban and Built-up Lands",
        14: "Cropland/Natural Vegetation Mosaics",
        15: "Permanent Snow and Ice",
        16: "Areas with less than 10% vegetation",
        17: "Water Bodies",
    }
    pft_data["pft"] = pft_data["landcover_type"].map(pft_mapping)

    for df_temp in [pft_data, lai_data, era5_data]:
        df_temp.rename(columns={"timestamp": "TIMESTAMP", "datetime": "TIMESTAMP"}, inplace=True)
        df_temp["TIMESTAMP"] = pd.to_datetime(df_temp["TIMESTAMP"], utc=True)
        df_temp.sort_values("TIMESTAMP", inplace=True)

    count = 0

    # --- Always use outliers_removed (gap-filling is NOT part of the main pipeline) ---
    sap_input_dir = paths.sap_outliers_removed_dir
    env_input_dir = paths.env_outliers_removed_dir
    print(f"Sap input: {sap_input_dir}")
    print(f"Env input: {env_input_dir}")

    # --- MAIN LOOP ---
    for sap_data_file in sap_input_dir.glob("*.csv"):
        try:
            parts = sap_data_file.stem.split("_")
            # Handle both "_sapf_data_gap_filled" and "_sapf_data_outliers_removed" suffixes (4 words each)
            suffix_words = 4
            location_type = "_".join(parts[:-suffix_words])

            # --- LOAD SITE METADATA ---
            env_data_file = env_input_dir / f"{location_type}_env_data_outliers_removed.csv"
            biome_data_file = paths.raw_csv_dir / f"{location_type}_site_md.csv"

            try:
                biome_info = pd.read_csv(biome_data_file)
                biome_type = biome_info["si_biome"][0]
                igbp_type = biome_info["si_igbp"][0]
                lat = biome_info["si_lat"][0]
                lon = biome_info["si_long"][0]

                curr_site_info = site_info[site_info["site_name"] == location_type]
                if curr_site_info.empty:
                    continue

                ele = curr_site_info["elevation_m"].values[0]
                slope = curr_site_info["slope_deg"].values[0]
                aspect_sin = curr_site_info["aspect_sin"].values[0]
                aspect_cos = curr_site_info["aspect_cos"].values[0]
                mean_annual_temp = curr_site_info["bio1"].values[0]
                mean_annual_precip = curr_site_info["bio12"].values[0]
                canopy_height = curr_site_info["canopy_height_m"].values[0]
                temp_seasonality = curr_site_info["bio4"].values[0]
                precip_seasonality = curr_site_info["bio15"].values[0]

                # --- 1. EXTRACT STAND AGE ---
                stand_age = np.nan
                if location_type in stand_age_map:
                    stand_age = stand_age_map[location_type]

                if pd.isna(stand_age):
                    stand_md_file = paths.raw_csv_dir / f"{location_type}_stand_md.csv"
                    if stand_md_file.exists():
                        try:
                            s_md = pd.read_csv(stand_md_file)
                            if "st_age" in s_md.columns:
                                val = s_md["st_age"].iloc[0]
                                stand_age = pd.to_numeric(val, errors="coerce")
                        except Exception:
                            pass

                # --- 2. EXTRACT SOIL PROPS ---
                avg_props = {"sand": np.nan, "clay": np.nan, "soc": np.nan, "bdod": np.nan, "cfvo": np.nan}
                raw_soil_props = {}
                theta_wp, theta_fc, theta_sat = np.nan, np.nan, np.nan

                if not soil_data.empty and location_type in soil_data["site_name"].values:
                    soil_row = soil_data[soil_data["site_name"] == location_type].iloc[0]
                    try:
                        avg_props = get_weighted_soil_props(soil_row)

                        raw_props = ["sand", "clay", "soc", "bdod", "cfvo"]
                        raw_depths = ["0_5", "5_15", "15_30", "30_60", "60_100"]

                        for p in raw_props:
                            for d in raw_depths:
                                col_key = f"{p}_{d}"
                                val = soil_row.get(col_key, np.nan)
                                raw_soil_props[f"soil_{col_key}"] = val

                        if not pd.isna(avg_props["sand"]) and not pd.isna(avg_props["clay"]):
                            om_pct = (avg_props["soc"] / 10.0) * 1.72
                            theta_wp, theta_fc, theta_sat = calculate_soil_hydraulics(
                                sand=avg_props["sand"] / 100.0,
                                clay=avg_props["clay"] / 100.0,
                                organic_matter=om_pct,
                                coarse_fragments_vol_percent=avg_props["cfvo"],
                            )
                    except Exception as soil_e:
                        print(f"Error calculating soil hydraulics for {location_type}: {soil_e}")

                site_biome_mapping[location_type] = biome_type
                all_pft_types.add(igbp_type)

            except Exception:
                continue

            # --- LOAD TIME SERIES ---
            env_data = pd.read_csv(env_data_file, parse_dates=["TIMESTAMP"])
            sap_data = pd.read_csv(sap_data_file, parse_dates=["TIMESTAMP"])
            sap_data.drop(columns=[col for col in sap_data.columns if "Unnamed" in col], inplace=True)

            env_data.set_index("TIMESTAMP", inplace=True)
            # solar_TIMESTAMP comes from sap side (canonical); drop env duplicate to avoid merge conflict
            env_data.drop(columns=["solar_TIMESTAMP"], errors="ignore", inplace=True)

            col_names = [
                col for col in sap_data.columns if col not in ["solar_TIMESTAMP", "TIMESTAMP", "TIMESTAMP_LOCAL"]
            ]

            if plant_level:
                # --- PLANT-LEVEL: melt wide → long (one row per tree per timestamp) ---
                id_vars = ["TIMESTAMP"]
                if "solar_TIMESTAMP" in sap_data.columns:
                    id_vars.append("solar_TIMESTAMP")
                sap_long = sap_data.melt(
                    id_vars=id_vars,
                    value_vars=col_names,
                    var_name="pl_code",
                    value_name="sap_velocity",
                )
                # Drop rows where sap_velocity is NaN (tree not measured at this timestamp)
                sap_long = sap_long.dropna(subset=["sap_velocity"])

                # Join plant metadata from plant_md.csv
                plant_md_file = paths.raw_csv_dir / f"{location_type}_plant_md.csv"
                if plant_md_file.exists():
                    plant_md = pd.read_csv(plant_md_file)
                    # Keep only useful columns for driver analysis
                    md_cols = [
                        "pl_code",
                        "pl_dbh",
                        "pl_sens_meth",
                        "pl_species",
                    ]
                    md_cols = [c for c in md_cols if c in plant_md.columns]
                    sap_long = sap_long.merge(plant_md[md_cols], on="pl_code", how="left")

                plant_sap_data = sap_long.set_index("TIMESTAMP")
            else:
                # --- SITE-LEVEL: average across trees (original behaviour) ---
                sap_data["site"] = sap_data[col_names].mean(axis=1)
                sap_dict = {
                    "TIMESTAMP": sap_data["TIMESTAMP"],
                    "sap_velocity": sap_data["site"],
                    "TIMESTAMP_LOCAL": sap_data.get("TIMESTAMP_LOCAL"),
                }
                if "solar_TIMESTAMP" in sap_data.columns:
                    sap_dict["solar_TIMESTAMP"] = sap_data["solar_TIMESTAMP"]
                plant_sap_data = pd.DataFrame(sap_dict).set_index("TIMESTAMP")

            df = pd.merge(plant_sap_data, env_data, left_index=True, right_index=True)

            # --- PROCESS HOURLY DATA ---
            # Resample env/sap data to strictly hourly using solar_TIMESTAMP (local solar time)
            mean_cols = [
                c
                for c in ["vpd", "ta", "ws", "sw_in", "rh", "netrad", "ppfd_in", "ext_rad", "sap_velocity"]
                if c in df.columns
            ]
            sum_cols = [c for c in ["precip"] if c in df.columns]

            # Aggregation rules for hourly resample
            agg_dict = {**{col: "sum" for col in sum_cols}, **{col: "mean" for col in mean_cols}}
            # Keep solar_TIMESTAMP by taking the first value in each hour
            if "solar_TIMESTAMP" in df.columns:
                agg_dict["solar_TIMESTAMP"] = "first"

            # Plant-level: carry tree metadata through as "first" (static per tree)
            plant_meta_cols = []
            if plant_level:
                plant_meta_cols = [
                    c
                    for c in [
                        "pl_code",
                        "pl_dbh",
                        "pl_sens_meth",
                        "pl_species",
                    ]
                    if c in df.columns
                ]
                for c in plant_meta_cols:
                    agg_dict[c] = "first"

            # Resample to hourly using UTC TIMESTAMP
            if plant_level and "pl_code" in df.columns:
                # Group by pl_code first, then resample each tree independently
                df_hourly = (
                    df.groupby("pl_code")
                    .resample("h")
                    .agg({k: v for k, v in agg_dict.items() if k != "pl_code"})
                    .reset_index()
                )
            else:
                df_hourly = df.resample("h").agg(agg_dict).reset_index()
            df_hourly.rename(columns={"index": "TIMESTAMP"}, inplace=True)

            # --- ADD STATIC VARIABLES (Common function for DF population) ---
            def add_static_vars(target_df):
                target_df["biome"] = biome_type
                target_df["pft"] = igbp_type
                target_df["site_name"] = location_type
                target_df["latitude"] = lat
                target_df["longitude"] = lon
                target_df["elevation"] = ele
                target_df["slope"] = slope
                target_df["aspect_sin"] = aspect_sin
                target_df["aspect_cos"] = aspect_cos
                target_df["mean_annual_temp"] = mean_annual_temp
                target_df["mean_annual_precip"] = mean_annual_precip
                target_df["temp_seasonality"] = temp_seasonality
                target_df["precip_seasonality"] = precip_seasonality
                target_df["canopy_height"] = canopy_height
                target_df["stand_age"] = stand_age
                target_df["soil_sand"] = avg_props["sand"]
                target_df["soil_clay"] = avg_props["clay"]
                target_df["soil_bdod"] = avg_props["bdod"]
                target_df["soil_soc"] = avg_props["soc"]
                target_df["soil_cfvo"] = avg_props["cfvo"]
                for k, v in raw_soil_props.items():
                    target_df[k] = v
                target_df["soil_theta_wp"] = theta_wp
                target_df["soil_theta_fc"] = theta_fc
                target_df["soil_theta_sat"] = theta_sat
                return target_df

            df_hourly = add_static_vars(df_hourly)
            df_hourly = df_hourly.sort_values("TIMESTAMP")
            # 1. Ensure timezone consistency
            if df_hourly["TIMESTAMP"].dt.tz is None:
                df_hourly["TIMESTAMP"] = df_hourly["TIMESTAMP"].dt.tz_localize("UTC")
            # --- MERGE EXTERNAL DATA (LAI, ERA5) TO HOURLY ---
            df_hourly = pd.merge_asof(df_hourly, lai_data, on="TIMESTAMP", by="site_name", direction="nearest")
            df_hourly = pd.merge_asof(df_hourly, era5_data, on="TIMESTAMP", by="site_name", direction="nearest")

            # --- NORMALIZE VOLUMETRIC SOIL WATER BY SITE STD ---
            if "volumetric_soil_water_layer_1" in df_hourly.columns:
                vsw = df_hourly["volumetric_soil_water_layer_1"]
                # use Z-score: (x - mean) / std
                vsw_std = vsw.std()
                vsw_mean = vsw.mean()
                # vsw = (vsw - vsw_mean)
                if vsw_std > 1e-10:  # Avoid division by zero
                    df_hourly["volumetric_soil_water_layer_1"] = vsw / vsw_std
                else:
                    # If std is ~0 (constant values), set normalized to 0
                    df_hourly["volumetric_soil_water_layer_1"] = 0.0
                    print(f"  Warning: {location_type} has near-zero std for volumetric_soil_water_layer_1")

            # --- NORMALIZE DEEPER SOIL WATER LAYERS (same approach as layer 1) ---
            for _layer in [2, 3, 4]:
                _col = f"volumetric_soil_water_layer_{_layer}"
                if _col in df_hourly.columns:
                    _vsw = df_hourly[_col]
                    _vsw_std = _vsw.std()
                    if _vsw_std > 1e-10:
                        df_hourly[_col] = _vsw / _vsw_std
                    else:
                        df_hourly[_col] = 0.0
                        print(f"  Warning: {location_type} has near-zero std for {_col}")

            # --- CALCULATE DAY LENGTH ---
            # Day length in hours based on latitude, elevation, and date (astronomical calculation)
            # Note: Use UTC TIMESTAMP for pvlib calculations - it needs actual moment in time
            try:
                timestamps = pd.DatetimeIndex(df_hourly["TIMESTAMP"])
                if timestamps.tzinfo is None:
                    timestamps = timestamps.tz_localize("UTC")

                # Get unique dates for efficiency
                unique_dates = timestamps.normalize().unique()
                site_elevation = ele if not pd.isna(ele) else 0

                # Calculate day length for each unique date
                # Use sun_rise_set_transit_geometric (doesn't require ephem package)
                day_length_map = {}
                for date in unique_dates:
                    try:
                        # Create a time range for the day to find sunrise/sunset
                        times_of_day = pd.date_range(date, date + pd.Timedelta(hours=24), freq="1min", tz="UTC")
                        solar_pos = pvlib.solarposition.get_solarposition(
                            times_of_day, lat, lon, altitude=site_elevation
                        )

                        # Find when sun crosses horizon (elevation = 0)
                        elevations = solar_pos["elevation"].values
                        above_horizon = elevations > 0

                        if above_horizon.all():
                            # Polar day - sun never sets
                            day_length_hours = 24.0
                        elif not above_horizon.any():
                            # Polar night - sun never rises
                            day_length_hours = 0.0
                        else:
                            # Count minutes with sun above horizon
                            day_length_hours = above_horizon.sum() / 60.0

                        day_length_map[date] = day_length_hours
                    except Exception:
                        day_length_map[date] = np.nan

                # Map day length to each row based on date
                df_hourly["day_length"] = timestamps.normalize().map(day_length_map)
            except Exception as e:
                print(f"  Warning: Could not calculate day_length for {location_type}: {e}")
                df_hourly["day_length"] = np.nan

            # --- SAVE HOURLY ---
            df_hourly.to_csv(hourly_out_dir / f"{location_type}_hourly.csv", index=False)

            # --- PROCESS DAILY DATA (Aggregation from Hourly) ---

            # Optional: Filter to daytime only before daily aggregation
            if daytime_only:
                # Use UTC TIMESTAMP for solar position calculation (pvlib needs actual moment in time)
                daytime_mask = calculate_daytime_mask(
                    df_hourly["TIMESTAMP"], lat=lat, lon=lon, elevation=ele if not pd.isna(ele) else 0
                )
                n_before = len(df_hourly)
                df_hourly_for_daily = df_hourly[daytime_mask.values].copy()
                n_after = len(df_hourly_for_daily)
                print(
                    f"  {location_type}: Filtered {n_before - n_after}/{n_before} nighttime hours ({(n_before - n_after) / n_before * 100:.1f}%)"
                )
            else:
                df_hourly_for_daily = df_hourly

            # Define Static columns that should NOT be summed/min/maxed (just take first)
            static_prefixes = ["soil_", "biome", "pft", "site_name", "solar_TIMESTAMP"]
            static_exact = [
                "latitude",
                "longitude",
                "elevation",
                "slope",
                "aspect_sin",
                "aspect_cos",
                "mean_annual_temp",
                "mean_annual_precip",
                "temp_seasonality",
                "precip_seasonality",
                "canopy_height",
                "stand_age",
            ]
            if plant_level:
                # Tree metadata columns are static per tree — aggregate as "first"
                static_prefixes.append("pl_")
                static_exact.append("pl_code")

            daily_agg_rules = {}

            for col in df_hourly_for_daily.columns:
                # Skip TIMESTAMP - it's handled by resample's on= parameter
                if col == "TIMESTAMP":
                    continue

                # Check if column is static metadata
                is_static = (col in static_exact) or any(col.startswith(p) for p in static_prefixes)

                if is_static:
                    daily_agg_rules[col] = "first"
                elif pd.api.types.is_numeric_dtype(df_hourly_for_daily[col]):
                    # For all dynamic numeric variables (Env, Sap, ERA5, LAI), save all stats
                    daily_agg_rules[col] = ["mean", "min", "max", "sum"]
                else:
                    # Fallback for other non-numeric types
                    daily_agg_rules[col] = "first"

            # 2. Resample using UTC TIMESTAMP
            # This creates a MultiIndex column structure (e.g. ('vpd', 'mean'), ('vpd', 'sum'))
            df_hourly_for_daily = df_hourly_for_daily.set_index("TIMESTAMP")
            if plant_level and "pl_code" in df_hourly_for_daily.columns:
                # Group by tree, then resample each tree to daily
                agg_no_plcode = {k: v for k, v in daily_agg_rules.items() if k != "pl_code"}
                df_daily = df_hourly_for_daily.groupby("pl_code").resample("D").agg(agg_no_plcode)
                # pl_code is now in the index; move it back to a column
                df_daily = df_daily.reset_index(level="pl_code")
            else:
                df_daily = df_hourly_for_daily.resample("D").agg(daily_agg_rules)

            # 3. Flatten MultiIndex columns
            new_cols = []
            for col_info in df_daily.columns:
                if isinstance(col_info, tuple) and len(col_info) == 2:
                    col_name, agg_method = col_info
                    if not agg_method or agg_method == "first":
                        new_cols.append(col_name)
                    elif agg_method == "mean":
                        new_cols.append(f"{col_name}")
                    else:
                        new_cols.append(f"{col_name}_{agg_method}")
                else:
                    # Regular column (e.g. pl_code after groupby reset_index)
                    new_cols.append(col_info)

            df_daily.columns = new_cols
            df_daily = df_daily.reset_index()

            # 4. Derived Calculations (using the new suffixed column names)
            # Use 'sum' for precipitation related ratio, or adjust to 'mean' if preferred.
            # Here we use sum to stay consistent with typical daily totals.
            if (
                "total_precipitation_hourly_sum" in df_daily.columns
                and "potential_evaporation_hourly_sum" in df_daily.columns
            ):
                pet = np.abs(df_daily["potential_evaporation_hourly_sum"].values)
                precip = df_daily["total_precipitation_hourly_sum"].values
                # Avoid division by zero
                pet = np.where(pet < 1e-10, 1e-10, pet)
                precip_PET = np.clip(precip / pet, 0, 10)
                df_daily["prcip/PET"] = precip_PET

            # --- SAVE DAILY ---
            df_daily.to_csv(daily_out_dir / f"{location_type}_daily.csv", index=False)

            # --- COLLECT FOR BIOME MERGING ---
            if biome_type not in biome_merged_hourly:
                biome_merged_hourly[biome_type] = []
                biome_merged_daily[biome_type] = []

            biome_merged_hourly[biome_type].append(df_hourly)
            biome_merged_daily[biome_type].append(df_daily)

            count += 1
            if count % 10 == 0:
                print(f"Processed {count} sites...")

        except Exception as e:
            print(f"Error processing {sap_data_file}: {str(e)}")
            continue

    # --- FINAL MERGING (HOURLY & DAILY) ---
    def save_biome_collections(collection_dict, output_dir, file_suffix, skip_all_biomes_merge=False):
        biome_dir = output_dir / "by_biome"
        biome_dir.mkdir(exist_ok=True)

        last_biome_df = None
        for biome, dfs in collection_dict.items():
            if dfs:
                biome_df = pd.concat(dfs)
                safe_biome_name = "".join(c if c.isalnum() or c in [" ", "_", "-"] else "_" for c in biome).replace(
                    " ", "_"
                )
                biome_df.to_csv(biome_dir / f"{safe_biome_name}_{file_suffix}.csv", index=False)
                last_biome_df = biome_df
                # Free memory after saving each biome
                del biome_df

        # For hourly data, skip all_biomes merge to avoid memory issues
        if skip_all_biomes_merge:
            print(f"  Skipping all_biomes merge for {file_suffix} to save memory")
            return last_biome_df

        # For daily data, merge all biomes (smaller files)
        all_biomes_path = output_dir / f"all_biomes_{file_suffix}.csv"
        print(f"  Merging all biomes into {all_biomes_path.name}...")

        # Write header from first file, then append others
        biome_files = list(biome_dir.glob(f"*_{file_suffix}.csv"))
        if biome_files:
            first_file = True
            for bf in biome_files:
                chunk_iter = pd.read_csv(bf, chunksize=50000)
                for chunk in chunk_iter:
                    chunk.to_csv(all_biomes_path, mode="a" if not first_file else "w", header=first_file, index=False)
                    first_file = False
            return pd.read_csv(all_biomes_path, nrows=1)  # Return just header for reference
        return None

    if biome_merged_hourly:
        print("Saving merged hourly data...")
        save_biome_collections(biome_merged_hourly, hourly_out_dir, "merged_hourly", skip_all_biomes_merge=True)

        print("Saving merged daily data...")
        daily_all = save_biome_collections(
            biome_merged_daily, daily_out_dir, "merged_daily", skip_all_biomes_merge=False
        )

        # Save mapping
        pd.DataFrame({"site": list(site_biome_mapping.keys()), "biome": list(site_biome_mapping.values())}).to_csv(
            output_base_dir / "site_biome_mapping.csv", index=False
        )

        print(f"Processing complete. Data saved to {output_base_dir}")
        return daily_all

    raise ValueError("No data was successfully processed")


def main():
    parser = argparse.ArgumentParser(description="Merge sap flow and environmental data")
    parser.add_argument(
        "--daytime-only",
        action="store_true",
        help="Filter to daytime data only before daily aggregation (uses solar elevation > 0)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory (default: paths.merged_data_root)"
    )
    parser.add_argument(
        "--plant-level",
        action="store_true",
        help="Output one row per tree per timestamp (long format) instead of site-averaged. "
        "Joins plant_md.csv to add tree metadata (pl_dbh, pl_species, pl_sens_meth, etc.).",
    )
    args = parser.parse_args()
    # Defines the base directory for output
    if args.output_dir:
        output_base_dir = Path(args.output_dir)
    else:
        output_base_dir = paths.merged_data_root

    merge_sap_env_data_site(output_base_dir, daytime_only=args.daytime_only, plant_level=args.plant_level)


if __name__ == "__main__":
    main()

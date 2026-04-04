"""Map FFS selected features back to training script CLI arguments.

Reads FFS results JSON, determines which --additional_features and
--feature_groups the training script needs to reproduce the selected
feature set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ── Feature → training-arg mapping ────────────────────────────────────────

# Base features: always included by training script, no args needed
BASE_FEATURES = {
    "sw_in",
    "ppfd_in",
    "ta",
    "vpd",
    "ws",
    "ext_rad",  # mandatory
    "precip",
    "ta_max",
    "ta_min",
    "vpd_max",
    "vpd_min",
    "canopy_height",
    "elevation",
    "LAI",
    "prcip/PET",
    "volumetric_soil_water_layer_1",
    "soil_temperature_level_1",
    "day_length",
    "sap_velocity",
}

# PFT one-hot columns: included when "pft" is in used_cols (default)
PFT_COLS = {"MF", "DNF", "ENF", "EBF", "WSA", "WET", "DBF", "SAV"}

# Time features: added by add_time_features(), training uses only "Year sin"
# for daily; others need to be added to time_features list
TIME_FEATURES = {
    "Day sin",
    "Day cos",
    "Week sin",
    "Week cos",
    "Month sin",
    "Month cos",
    "Year sin",
    "Year cos",
}

# Features from add_sap_flow_features(): always computed, but only used
# if listed in additional_features
SAP_FLOW_FEATURES = {
    "is_daytime",
    "southern_hemisphere",
    "vpd_mean_6h",
    "vpd_log",
    "sw_in_cumsum_day",
    "vpd_cumsum_6h",
    "ta_change_1h",
    "tropical",
    "boreal",
    "dew_point",
    "dew_point_depression",
    "precip_sum_24h",
    "precip_sum_72h",
    "hours_since_rain",
}

# Additional features: passed via --additional_features
ADDITIONAL_FEATURES = {
    "slope",
    "aspect_sin",
    "aspect_cos",
    "stand_age",
    "soil_sand",
    "soil_clay",
    "soil_soc",
    "soil_cfvo",
    "temp_seasonality",
    "precip_seasonality",
    "mean_annual_temp",
    "mean_annual_precip",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "soil_temperature_level_4",
    "rh",
    "rh_max",
    "rh_min",
    "surface_pressure",
    "potential_evaporation_hourly_sum",
    "total_precipitation_hourly_sum",
}

# FE feature → feature_group name mapping
FE_FEATURE_TO_GROUP: dict[str, str] = {
    # interactions
    "vpd_x_sw_in": "interactions",
    "vpd_squared": "interactions",
    "ta_x_vpd": "interactions",
    "height_x_vpd": "interactions",
    "wind_x_vpd": "interactions",
    "demand_x_supply": "interactions",
    # lags
    "ta_lag1d": "lags_1d",
    "vpd_lag1d": "lags_1d",
    "sw_in_lag1d": "lags_1d",
    "precip_lag1d": "lags_1d",
    "rh_lag1d": "lags_1d",
    # rolling_3d
    "ta_roll3d_mean": "rolling_3d",
    "ta_roll3d_std": "rolling_3d",
    "vpd_roll3d_mean": "rolling_3d",
    "vpd_roll3d_std": "rolling_3d",
    "sw_in_roll3d_mean": "rolling_3d",
    "sw_in_roll3d_std": "rolling_3d",
    "rh_roll3d_mean": "rolling_3d",
    "rh_roll3d_std": "rolling_3d",
    # rolling_7d
    "ta_roll7d_mean": "rolling_7d",
    "ta_roll7d_std": "rolling_7d",
    "vpd_roll7d_mean": "rolling_7d",
    "vpd_roll7d_std": "rolling_7d",
    "sw_in_roll7d_mean": "rolling_7d",
    "sw_in_roll7d_std": "rolling_7d",
    "rh_roll7d_mean": "rolling_7d",
    "rh_roll7d_std": "rolling_7d",
    # rolling_14d
    "ta_roll14d_mean": "rolling_14d",
    "ta_roll14d_std": "rolling_14d",
    "vpd_roll14d_mean": "rolling_14d",
    "vpd_roll14d_std": "rolling_14d",
    "sw_in_roll14d_mean": "rolling_14d",
    "sw_in_roll14d_std": "rolling_14d",
    "rh_roll14d_mean": "rolling_14d",
    "rh_roll14d_std": "rolling_14d",
    # physics
    "clear_sky_index": "physics",
    "gdd": "physics",
    "absorbed_radiation": "physics",
    # precip_memory
    "precip_sum_3d": "precip_memory",
    "precip_sum_7d": "precip_memory",
    "days_since_rain": "precip_memory",
    # indicators
    "vpd_high": "indicators",
    "soil_moisture_rel": "indicators",
    "soil_dry": "indicators",
    # eco-hydro (each is its own group)
    "swc_layer2_norm": "root_zone_swc",
    "swc_layer3_norm": "root_zone_swc",
    "swc_layer4_norm": "root_zone_swc",
    "rew": "rew",
    "et0": "et0",
    "psi_soil": "psi_soil",
    "cwd": "cwd",
}

# rh rolling/lag: now supported by training script's apply_feature_engineering().
# Kept as empty set for backwards compatibility.
RH_EXTENSION_FEATURES: set[str] = set()


def map_features_to_args(
    selected_features: list[str],
) -> dict[str, list[str]]:
    """Map FFS selected features to training script arguments.

    Returns dict with keys:
        additional_features: list for --additional_features
        feature_groups: list for --feature_groups
        time_features: extra time features beyond "Year sin"
        rh_extensions: rh rolling/lag features (need custom handling)
        unmapped: features that couldn't be classified
    """
    additional = []
    fe_groups: set[str] = set()
    extra_time = []
    rh_ext = []
    unmapped = []

    for feat in selected_features:
        if feat in BASE_FEATURES or feat in PFT_COLS:
            continue  # already included by default
        elif feat in ADDITIONAL_FEATURES or feat in SAP_FLOW_FEATURES:
            additional.append(feat)
        elif feat in FE_FEATURE_TO_GROUP:
            fe_groups.add(FE_FEATURE_TO_GROUP[feat])
        elif feat in TIME_FEATURES:
            if feat != "Year sin":  # Year sin is default for daily
                extra_time.append(feat)
        elif feat in RH_EXTENSION_FEATURES:
            rh_ext.append(feat)
        else:
            unmapped.append(feat)

    # Eco-hydro groups need soil_soc and soil_cfvo as additional_features
    eco_groups = {"rew", "et0", "psi_soil", "cwd", "root_zone_swc"}
    if fe_groups & eco_groups:
        for dep in ["soil_soc", "soil_cfvo"]:
            if dep not in additional:
                additional.append(dep)
        # root_zone_swc needs deeper soil layers
        if "root_zone_swc" in fe_groups:
            for dep in [
                "volumetric_soil_water_layer_2",
                "volumetric_soil_water_layer_3",
                "volumetric_soil_water_layer_4",
            ]:
                if dep not in additional:
                    additional.append(dep)

    return {
        "additional_features": sorted(set(additional)),
        "feature_groups": sorted(fe_groups),
        "time_features": sorted(extra_time),
        "rh_extensions": sorted(rh_ext),
        "unmapped": unmapped,
    }


def generate_training_command(
    args: dict[str, list[str]],
    run_id: str = "ffs_best",
    hyperparams_json: str = "src/hyperparameter_optimization/JSON/XGB_hyperparameters_fixed.json",
    selected_features_json: str = "",
) -> str:
    """Generate a training script CLI command from mapped args.

    When selected_features_json is provided, uses ALL additional features and
    ALL feature engineering groups (matching FFS cache behavior) so that dropna
    removes the same rows as FFS. The --selected_features filter then keeps
    only the FFS-selected features.
    """
    from src.forward_selection.feature_registry import (
        ADDITIONAL_FEATURES as ALL_ADDITIONAL,
    )
    from src.forward_selection.feature_registry import (
        ALL_FEATURE_ENGINEERING_GROUPS as ALL_FE_GROUPS,
    )

    cmd_parts = [
        "python src/hyperparameter_optimization/test_hyperparameter_tuning_ML_spatial_stratified_prediction.py",
        "--model xgb --RANDOM_SEED 42 --n_groups 10",
        "--SPLIT_TYPE spatial_stratified --TIME_SCALE daily",
        "--IS_TRANSFORM True --TRANSFORM_METHOD log1p",
        "--IS_STRATIFIED True --IS_CV True",
        "--SHAP_SAMPLE_SIZE 50000",
        f"--hyperparameters {hyperparams_json}",
        f"--run_id {run_id}",
    ]

    if selected_features_json:
        # Use ALL features for matching FFS dropna, then filter to selected
        cmd_parts.append("--additional_features " + " ".join(sorted(ALL_ADDITIONAL)))
        cmd_parts.append("--feature_groups " + " ".join(sorted(ALL_FE_GROUPS)))
        cmd_parts.append(f"--selected_features {selected_features_json}")
    else:
        if args["additional_features"]:
            cmd_parts.append("--additional_features " + " ".join(args["additional_features"]))
        if args["feature_groups"]:
            cmd_parts.append("--feature_groups " + " ".join(args["feature_groups"]))

    return " \\\n  ".join(cmd_parts)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python ffs_to_training_args.py <ffs_results.json> [run_id]")
        sys.exit(1)

    results_path = Path(sys.argv[1])
    run_id = sys.argv[2] if len(sys.argv) > 2 else "ffs_best"

    with open(results_path) as f:
        results = json.load(f)

    selected = results["selected_features"]
    scoring = results["scoring"]
    best_score = results["best_score"]
    n_selected = results["n_selected"]

    print(f"FFS scoring: {scoring}")
    print(f"Best score: {best_score:.4f}")
    print(f"Selected features ({n_selected}): {selected}")
    print()

    args = map_features_to_args(selected)

    print(f"--additional_features ({len(args['additional_features'])}): {args['additional_features']}")
    print(f"--feature_groups ({len(args['feature_groups'])}): {args['feature_groups']}")
    if args["time_features"]:
        print(f"Extra time features: {args['time_features']}")
    if args["rh_extensions"]:
        print(f"WARNING: rh extensions not in training script: {args['rh_extensions']}")
    if args["unmapped"]:
        print(f"WARNING: unmapped features: {args['unmapped']}")
    print()

    # Save selected features JSON for --selected_features arg
    sf_json_file = results_path.parent / f"selected_features_{scoring}.json"
    with open(sf_json_file, "w") as f:
        json.dump({"selected_features": selected, "n_selected": n_selected, "scoring": scoring}, f, indent=2)
    print(f"Selected features JSON saved to {sf_json_file}")

    cmd = generate_training_command(args, run_id=run_id, selected_features_json=str(sf_json_file))
    print("Training command:")
    print(cmd)

    # Save command to file
    cmd_file = results_path.parent / f"training_cmd_{scoring}.sh"
    with open(cmd_file, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Auto-generated from FFS {scoring} results\n")
        f.write(f"# Best score: {best_score:.4f}, {n_selected} features\n\n")
        f.write(cmd + "\n")
    print(f"\nSaved to {cmd_file}")


if __name__ == "__main__":
    main()

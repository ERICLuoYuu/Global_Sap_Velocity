from __future__ import annotations

import gc
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
import seaborn as sns

parent_dir = str(Path(__file__).parent.parent.parent)
print(parent_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from path_config import PathConfig

from src.Analyzers.mannual_removal_processor import RemovalLogProcessor
from src.gap_filling.config import GapFillingConfig  # noqa: F811
from src.gap_filling.filler import GapFiller  # noqa: F811
from src.tools import adjust_time_to_local, create_timezone_mapping


class SapFlowAnalyzer:
    """
    Analyzer for Global SapFlow data with flag handling and day/night processing
    """

    def __init__(
        self, paths: PathConfig = None, scale: str = "sapwood", gap_filling_config: GapFillingConfig | None = None
    ):
        self.paths = paths if paths is not None else PathConfig(scale=scale)
        self.data_dir = self.paths.sap_data_dir
        self.data_outlier_removed_dir = self.paths.sap_outliers_removed_dir

        # Dictionaries are initialized but will be cleared after each batch.
        self.sapf_raw_data = {}
        self.sapflow_data = {}
        self.outlier_removed_data = {}
        self.timezone_map = create_timezone_mapping()

        # --- Gap-filling support ---
        self.gap_filling_config = gap_filling_config
        self._gap_filler: GapFiller | None = None
        if gap_filling_config is not None:
            self._gap_filler = GapFiller(gap_filling_config, target="sap")

        # --- Removal log support ---
        self.removal_processor: RemovalLogProcessor | None = None
        self._removal_log_path: Path | None = None
        self.removal_stats = {
            "sites_skipped": [],
            "columns_removed": {},
            "periods_removed": {},
            "total_points_removed": 0,
        }

    def set_removal_log(self, log_path: str | Path, validate: bool = True):
        """
        Set the removal log to use during data processing.

        This should be called BEFORE running any analysis methods like
        run_analysis_in_batches() or _load_sapflow_data().

        Args:
            log_path: Path to the removal log CSV file.
            validate: If True, validate entries and report issues.

        Example:
            analyzer = SapFlowAnalyzer()
            analyzer.set_removal_log('visual_inspection_removals.csv')
            analyzer.run_analysis_in_batches(batch_size=10)
        """
        self.removal_processor = RemovalLogProcessor()
        self.removal_processor.load_removal_log(log_path, validate=validate)
        self._removal_log_path = Path(log_path)

        print(f"\n{'=' * 60}")
        print("REMOVAL LOG ACTIVATED")
        print(f"{'=' * 60}")
        print(f"Log file: {self._removal_log_path.name}")
        print(f"Total entries: {len(self.removal_processor.removal_entries)}")
        print("Removals will be applied during data loading.")
        print(f"{'=' * 60}\n")

    def clear_removal_log(self):
        """Clear the current removal log (disable removal processing)."""
        self.removal_processor = None
        self._removal_log_path = None
        self.removal_stats = {
            "sites_skipped": [],
            "columns_removed": {},
            "periods_removed": {},
            "total_points_removed": 0,
        }
        print("Removal log cleared.")

    def _should_skip_file(self, file: Path) -> bool:
        """
        Check if a file should be skipped entirely based on removal log.

        Args:
            file: Path to the sapflow data file.

        Returns:
            True if the entire site should be skipped.
        """
        if self.removal_processor is None:
            return False

        parts = file.stem.split("_")
        location = "_".join(parts[:2])
        plant_type = "_".join(parts[2:])

        return self.removal_processor.should_skip_site(location, plant_type)

    def _apply_removal_log(self, df: pd.DataFrame, location: str, plant_type: str) -> pd.DataFrame:
        """
        Apply removal log entries to a DataFrame.

        This removes:
        - Entire columns marked for removal
        - Specific time periods for columns

        Args:
            df: DataFrame with TIMESTAMP index.
            location: Site location code.
            plant_type: Plant type identifier.

        Returns:
            DataFrame with removals applied.
        """
        if self.removal_processor is None:
            return df

        # Track original data points
        data_cols = [
            col
            for col in df.columns
            if col not in ["TIMESTAMP", "solar_TIMESTAMP", "TIMESTAMP_LOCAL", "lat", "long"]
            and "Unnamed:" not in str(col)
        ]
        points_before = df[data_cols].notna().sum().sum()

        # Apply removals using the processor
        df = self.removal_processor.apply_removals(df, location, plant_type, inplace=False, verbose=True)

        # Track statistics
        points_after = df[data_cols].notna().sum().sum()
        site_key = f"{location}_{plant_type}"

        if site_key not in self.removal_stats["columns_removed"]:
            self.removal_stats["columns_removed"][site_key] = self.removal_processor.get_columns_to_remove(
                location, plant_type
            )

        self.removal_stats["total_points_removed"] += points_before - points_after

        return df

    def get_removal_summary(self) -> dict:
        """
        Get a summary of all removals applied during processing.

        Returns:
            Dictionary with removal statistics.
        """
        return {
            "removal_log_path": str(self._removal_log_path) if self._removal_log_path else None,
            "sites_skipped": self.removal_stats["sites_skipped"],
            "columns_removed": self.removal_stats["columns_removed"],
            "total_points_removed": self.removal_stats["total_points_removed"],
        }

    def save_removal_report(self, output_path: str | Path = None):
        """
        Save a detailed report of all removals applied.

        Args:
            output_path: Path for the report CSV. If None, uses default directory.
        """
        if self.removal_processor is None:
            print("No removal log was set.")
            return

        if output_path is None:
            output_path = self.paths.sap_outliers_removed_dir / "removal_report.csv"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = self.removal_processor.get_removal_report()
        report.to_csv(output_path, index=False)
        print(f"Removal report saved to {output_path}")

    # NEW: A method to control the batch processing
    def run_analysis_in_batches(self, batch_size: int = 1, switch="both"):
        """
        Orchestrates the analysis by processing files in batches to manage memory.
        """
        # Sites with significant negative values to process
        sites_with_negatives = [  # 9.3% negative
            "ESP_MAJ_NOR_LM1",  # 5.9% negative
        ]

        # Collect files for these specific sites
        all_sapf_files = []
        for site in sites_with_negatives:
            matching = list(self.data_dir.glob(f"{site}*_sapf_data.csv"))
            all_sapf_files.extend(matching)

        # === NEW: Filter out files for sites that should be entirely skipped ===
        if self.removal_processor is not None:
            original_count = len(all_sapf_files)
            filtered_files = []
            for file in all_sapf_files:
                if self._should_skip_file(file):
                    parts = file.stem.split("_")
                    location = "_".join(parts[:2])
                    plant_type = "_".join(parts[2:])
                    self.removal_stats["sites_skipped"].append(f"{location}_{plant_type}")
                    print(f"[SKIP ENTIRE SITE] {file.stem}")
                else:
                    filtered_files.append(file)
            all_sapf_files = filtered_files

            if original_count != len(all_sapf_files):
                print(f"\nSkipped {original_count - len(all_sapf_files)} entire sites based on removal log")

        print(f"Found {len(all_sapf_files)} total files to process in batches of {batch_size}.")

        file_batches = [all_sapf_files[i : i + batch_size] for i in range(0, len(all_sapf_files), batch_size)]

        if switch == "both":
            for i, batch in enumerate(file_batches):
                print("-" * 80)
                print(
                    f"Loading and plotting Batch {i + 1}/{len(file_batches)} (files {i * batch_size + 1} to {i * batch_size + len(batch)})"
                )

                self._load_sapflow_data(files_to_load=batch)

                print("\nGenerating plots for the current batch...")
                self.plot_all_plants(save_dir=self.paths.sap_cleaned_figures_dir, progress_update=False)

                self._clear_batch_data()

        elif switch == "load":
            for i, batch in enumerate(file_batches):
                print("-" * 80)
                print(
                    f"Loading Batch {i + 1}/{len(file_batches)} (files {i * batch_size + 1} to {i * batch_size + len(batch)})"
                )
                self._load_sapflow_data(files_to_load=batch)
                self._clear_batch_data()

        elif switch == "plot":
            if not self.data_outlier_removed_dir.exists():
                print("No outlier removed data found. Please run the 'load' switch first to load data.")
                raise ValueError("No outlier removed data found. Please run the 'load' switch first to load data.")
            else:
                all_sapf_files_outliersremoved = list(self.data_outlier_removed_dir.glob("*_outliers_removed.csv"))
                print("Plotting outlier removed data in batches...")
                file_batches_outliersremoved = [
                    all_sapf_files_outliersremoved[i : i + batch_size]
                    for i in range(0, len(all_sapf_files_outliersremoved), batch_size)
                ]
                for i, batch in enumerate(file_batches_outliersremoved):
                    print("-" * 80)
                    print(
                        f"Plotting Batch {i + 1}/{len(file_batches)} (files {i * batch_size + 1} to {i * batch_size + len(batch)})"
                    )
                    print("-" * 80)
                    print(f"Plotting all plants from the loaded data (up to {len(all_sapf_files)} files)")
                    self.plot_all_plants(
                        save_dir=self.paths.sap_cleaned_figures_dir, files_to_load=batch, progress_update=False
                    )
                    self._clear_batch_data()
        else:
            raise ValueError("Invalid switch value. Use 'both', 'load', or 'plot'.")

        print("-" * 80)
        print("All batches processed successfully.")

        # === NEW: Save removal report if removal log was used ===
        if self.removal_processor is not None:
            self.save_removal_report()
            print("\nRemoval Summary:")
            print(f"  Sites skipped: {len(self.removal_stats['sites_skipped'])}")
            print(f"  Total points removed: {self.removal_stats['total_points_removed']}")

    def calculate_site_years(self, file_pattern: str = "*_sapf_data.csv", max_gap_days: int = 366) -> pd.DataFrame:
        """
        Calculates total site-years, deducting significant gaps (e.g., missing years).

        Args:
            file_pattern: Glob pattern to match files.
            max_gap_days: If a gap between records > this days, it is excluded from site-years.
                        Default is 30 days. Set to 366 to only exclude full missing years.
        """
        all_files = list(self.data_dir.glob(file_pattern))
        print(f"Found {len(all_files)} files. Calculating site-years (Gap threshold: {max_gap_days} days)...")

        stats = []

        for file in all_files:
            try:
                # OPTIMIZATION: Read ONLY the TIMESTAMP column
                df = pd.read_csv(file)
                # print(df.columns)
                df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
                df = df.dropna(subset=["TIMESTAMP"])
                # drop when all columns are NA ecept TIMESTAMP
                length_raw = len(df)
                df = df.dropna(
                    how="all", subset=[col for col in df.columns if col != "TIMESTAMP" and col != "solar_TIMESTAMP"]
                )
                length_dropped = len(df)
                print(f"Successfully dropped NA rows: {length_raw - length_dropped} rows removed.")
                if len(df) > 1:
                    # 1. Sort Data (Crucial for diff calculations)
                    df = df.sort_values("TIMESTAMP")

                    # 2. Calculate time difference between consecutive rows
                    # diff() gives us: [Row2-Row1, Row3-Row2, ...]
                    deltas = df["TIMESTAMP"].diff()

                    # 3. Define the threshold for a "gap"
                    gap_threshold = pd.Timedelta(days=max_gap_days)

                    # 4. Filter: Keep only durations that are connected (smaller than gap threshold)
                    # If the difference is 1 year (missing data), it is > gap_threshold, so we exclude it.
                    valid_durations = deltas[deltas <= gap_threshold]

                    # 5. Sum valid durations
                    total_duration_seconds = valid_durations.sum().total_seconds()

                    # Calculate simple Min/Max for reporting
                    start_date = df["TIMESTAMP"].min()
                    end_date = df["TIMESTAMP"].max()

                    # Convert to years
                    years = total_duration_seconds / (3600 * 24 * 365.25)

                    # Metadata extraction
                    parts = file.stem.split("_")
                    location = "_".join(parts[:2])
                    plant_type = "_".join(parts[2:-2])

                    stats.append(
                        {
                            "filename": file.name,
                            "location": location,
                            "plant_type": plant_type,
                            "start_date": start_date,
                            "end_date": end_date,
                            "years": years,
                            "coverage_percent": (years * 365.25) / ((end_date - start_date).days + 1) * 100,
                        }
                    )
                else:
                    print(f"Skipping {file.name}: Not enough data points.")

            except Exception as e:
                print(f"Error reading {file.name}: {e}")

        summary_df = pd.DataFrame(stats)

        if not summary_df.empty:
            total_site_years = summary_df["years"].sum()
            print("-" * 60)
            print("CALCULATION COMPLETE")
            print("-" * 60)
            print(f"Total Files:       {len(summary_df)}")
            print(f"TOTAL SITE-YEARS:  {total_site_years:.2f}")
            print("-" * 60)
        else:
            print("No valid data found.")

        return summary_df

    # NEW: A helper method to clear the dictionaries
    def _clear_batch_data(self):
        """
        Resets data dictionaries and calls the garbage collector to free memory.
        """
        print("\nClearing data from memory...")
        self.sapf_raw_data.clear()
        self.sapflow_data.clear()
        self.outlier_removed_data.clear()
        gc.collect()
        print("Memory cleared.")

    # MODIFIED: _load_sapflow_data now accepts a list of files to load
    def _load_sapflow_data(self, files_to_load: list[Path]):
        """Load a specific batch of sapflow data files and their corresponding flags"""
        print(f"Loading {len(files_to_load)} sapflow files for this batch...")

        # Check if removal log is active
        has_removal_log = self.removal_processor is not None

        if has_removal_log:
            print("\n[REMOVAL LOG ACTIVE] Removals will be applied after loading each file.\n")

        column_mapping = {"TIMESTAMP_solar": "solar_TIMESTAMP"}

        for file in files_to_load:
            parts = file.stem.split("_")
            location = "_".join(parts[:2])
            plant_type = "_".join(parts[2:])

            try:
                temp_df = pd.read_csv(file, nrows=1)
                numeric_cols = [col for col in temp_df.columns if "TIMESTAMP" not in col]
                dtype_map = {col: "float32" for col in numeric_cols}

                df = pd.read_csv(file, parse_dates=["TIMESTAMP"], dtype=dtype_map)
                df = df.rename(columns=column_mapping)
                if "solar_TIMESTAMP" in df.columns:
                    df["solar_TIMESTAMP"] = pd.to_datetime(df["solar_TIMESTAMP"])

                if location not in self.sapf_raw_data:
                    self.sapf_raw_data[location] = {}
                self.sapf_raw_data[location][plant_type] = df

                time_zone_file = file.parent / f"{file.stem.replace('sapf_data', 'env_md')}.csv"
                tz_df = pd.read_csv(time_zone_file)
                time_zone = tz_df["env_time_zone"].iloc[0]
                timestep = tz_df["env_timestep"].iloc[0]
                df["TIMESTAMP_LOCAL"] = df["TIMESTAMP"].apply(
                    lambda ts: adjust_time_to_local(ts, time_zone, self.timezone_map)
                )
                site_md_file = file.parent / f"{file.stem.replace('_sapf_data', '_site_md')}.csv"
                site_md = pd.read_csv(site_md_file)
                latitude = site_md["si_lat"].values[0]
                longitude = site_md["si_long"].values[0]
                elevation = site_md["si_elev"].values[0]

                flag_file = file.parent / f"{file.stem.replace('_data', '_flags')}.csv"
                if flag_file.exists():
                    flags = pd.read_csv(flag_file, parse_dates=["TIMESTAMP"])
                    flags = flags.rename(columns=column_mapping)
                    if "solar_TIMESTAMP" in flags.columns:
                        flags["solar_TIMESTAMP"] = pd.to_datetime(flags["solar_TIMESTAMP"])
                else:
                    flags = None

                # === NEW: Apply removal log BEFORE processing ===
                if has_removal_log:
                    print(f"\n--- Applying Removal Log: {location}_{plant_type} ---")
                    df = self._apply_removal_log(df, location, plant_type)

                df_processed = self._process_sapflow_data(
                    df, latitude, longitude, timestep, time_zone, elevation, location, plant_type, flags
                )

                if df_processed is not None:
                    df_processed["lat"] = latitude
                    df_processed["long"] = longitude

                    if location not in self.sapflow_data:
                        self.sapflow_data[location] = {}
                    self.sapflow_data[location][plant_type] = df_processed

            except Exception as e:
                print(f"Error loading {file.name}: {str(e)}")

    def _process_sapflow_data(
        self,
        df: pd.DataFrame,
        latitude: float,
        longitude: float,
        timestep: float,
        time_zone: str,
        elevation: float,
        location: str = None,
        plant_type: str = None,
        flags: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Process sapflow data with debugging information and day/night processing.

        UPDATED to include:
        - Growing season filtering
        - Incomplete day removal (>50% missing daytime data)
        """
        print("\nProcessing data shape:", df.shape)
        print("Data types before processing:")
        print(df.dtypes)
        print("\nSample of raw data:")
        print(df.head())

        # Convert string "NA" to numpy NaN
        df = df.replace("NA", np.nan)

        # Define sap flow variables (all non-timestamp columns)
        SAP_FLOW_VARIABLES = [
            col
            for col in df.columns
            if col not in ["TIMESTAMP", "solar_TIMESTAMP", "TIMESTAMP_LOCAL", "lat", "long"]
            and "Unnamed:" not in str(col)
        ]

        # Convert numeric columns to float, with error handling
        numeric_cols = [
            col for col in df.columns if col != "TIMESTAMP" and col != "solar_TIMESTAMP" and col != "TIMESTAMP_LOCAL"
        ]
        for col in numeric_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                print(f"\nColumn {col} range: {df[col].min()} to {df[col].max()}")
            except Exception as e:
                print(f"Error converting column {col}: {str(e)}")

        # Set index with error checking
        try:
            if "TIMESTAMP" not in df.columns:
                print("Error: TIMESTAMP column not found")
                return None

            is_datetime = pd.api.types.is_datetime64_any_dtype(df["TIMESTAMP"])
            print(f"TIMESTAMP column is already datetime: {is_datetime}")

            df = df.set_index("TIMESTAMP")
            df = df.sort_index()

            if df.index.isna().all():
                print("Error: All timestamp values are NaN")
                return None

        except Exception as e:
            print(f"Error setting timestamp index: {str(e)}")
            return None

        # Check for duplicates
        dup_mask = df.index.duplicated()
        dup_count = dup_mask.sum()

        if dup_count > 0:
            print(f"Found {dup_count} duplicate timestamps")
            dup_dir = self.paths.sap_duplicates_dir
            dup_dir.mkdir(parents=True, exist_ok=True)
            df[dup_mask].to_csv(dup_dir / f"{location}_{plant_type}_duplicates.csv")
            print(f"Removing {dup_count} duplicate timestamps ({dup_count / len(df) * 100:.1f}%)")
            df = df[~dup_mask]
        else:
            print("No duplicates found in timestamps")

        # Apply flag-based filtering
        if flags is not None:
            flags = flags.set_index("TIMESTAMP").sort_index()
            if flags.index.duplicated().any():
                flags = flags[~flags.index.duplicated(keep="first")]

            data_cols = [col for col in df.columns if col not in ["solar_TIMESTAMP", "TIMESTAMP_LOCAL", "lat", "long"]]
            output_dir = self.paths.sap_filtered_dir / "TEST"
            output_dir.mkdir(parents=True, exist_ok=True)
            for col in data_cols:
                flag_col = [fcol for fcol in flags.columns if col in fcol]
                if flag_col:
                    warn_mask = flags[flag_col[0]].astype(str).str.contains("RANGE_WARN", na=False)
                    negative_sap_mask = df[col] < 0
                    warn_mask = warn_mask & ~negative_sap_mask
                    df.loc[df[col] < 0, col] = np.nan
                    print(f"Filtering negative values in {col}: {negative_sap_mask.sum()} values set to NaN")
                    if warn_mask.any():
                        print(f"\nFiltering {warn_mask.sum()} values in {col} due to warnings")
                        df.loc[warn_mask, [col, "solar_TIMESTAMP"]].to_csv(output_dir / f"{col}_flagged.csv")
                    df.loc[warn_mask, col] = np.nan

            filter_path = output_dir / f"{location}_{plant_type}_filtered.csv"
            df.to_csv(filter_path, index=True)
            print(f"Saved filtered data to {filter_path}")

        try:
            # --- Parameters ---
            pre_timewindow = self._get_timewindow(30 * 24 * 60 * 60, df.index)
            DEFAULT_WINDOW_POINTS = pre_timewindow if pre_timewindow < len(df) else len(df)
            DEFAULT_N_STD = 3

            DAY_SAPFLOW_WINDOW_POINTS = pre_timewindow if pre_timewindow < len(df) else len(df)
            DAY_SAPFLOW_N_STD = 3
            NIGHT_SAPFLOW_N_STD = 5
            NIGHT_SAPFLOW_WINDOW_POINTS = DAY_SAPFLOW_WINDOW_POINTS

            VARIABILITY_WINDOW_POINTS = self._get_timewindow(2 * 24 * 60 * 60, df.index)
            VARIABILITY_WINDOW_POINTS = min(VARIABILITY_WINDOW_POINTS, len(df))

            DAY_CV_LOW = 0.08
            DAY_CV_HIGH = 3.5
            DAY_STD_LOW = 0.4
            DAY_STD_HIGH = 55.0
            NIGHT_CV_HIGH = 3.5
            NIGHT_STD_HIGH = 10.0

            print(f"Outlier window points: {DAY_SAPFLOW_WINDOW_POINTS}")
            print(f"Variability window points: {VARIABILITY_WINDOW_POINTS}")

            if location not in self.outlier_removed_data:
                self.outlier_removed_data[location] = {}

            # Create directories
            outlier_dir = self.paths.sap_outliers_dir
            outlier_dir.mkdir(parents=True, exist_ok=True)
            day_mask_dir = self.paths.sap_day_masks_dir
            day_mask_dir.mkdir(parents=True, exist_ok=True)
            outliers_removed_dir = self.paths.sap_outliers_removed_dir
            outliers_removed_dir.mkdir(parents=True, exist_ok=True)
            variability_filtered_dir = self.paths.sap_variability_filtered_dir
            variability_filtered_dir.mkdir(parents=True, exist_ok=True)
            variability_mask_dir = self.paths.sap_variability_masks_dir
            variability_mask_dir.mkdir(parents=True, exist_ok=True)
            reversed_dir = self.paths.sap_reversed_dir
            reversed_dir.mkdir(parents=True, exist_ok=True)

            current_df_latitude = latitude
            current_df_longitude = longitude

            data_cols = [
                col
                for col in df.columns
                if col not in ["solar_TIMESTAMP", "TIMESTAMP_LOCAL", "lat", "long"] and "Unnamed:" not in str(col)
            ]

            variability_stats = []

            # Calculate day/night mask ONCE for all columns
            try:
                day_mask = self._calculate_daytime_mask(
                    df.index, current_df_latitude, current_df_longitude, elevation=0, elevation_threshold=0
                )
                # === DEBUG: Check day_mask and coordinates ===
                print("\n[DEBUG DAY_MASK]")
                print(f"  Latitude: {current_df_latitude}")
                print(f"  Longitude: {current_df_longitude}")
                print(f"  Elevation: {elevation}")
                print(f"  Total points: {len(day_mask)}")
                print(f"  Daytime points: {day_mask.sum()} ({day_mask.sum() / len(day_mask) * 100:.1f}%)")
                print(f"  Nighttime points: {(~day_mask).sum()} ({(~day_mask).sum() / len(day_mask) * 100:.1f}%)")

                # Check a sample of timestamps and their day/night classification
                sample_idx = [0, len(df) // 4, len(df) // 2, 3 * len(df) // 4]
                print("\n  Sample day/night classification:")
                for idx in sample_idx:
                    ts = df.index[idx]
                    is_day = day_mask.iloc[idx]
                    val = df.iloc[idx][data_cols[0]] if data_cols else None
                    print(f"    {ts} -> {'DAY' if is_day else 'NIGHT'} (value: {val})")
                # === END DEBUG ===
                day_mask_raw_values = day_mask.values

                if len(day_mask_raw_values) != len(df.index):
                    raise ValueError(f"Day mask length mismatch: {len(day_mask_raw_values)} vs {len(df.index)}")

                day_mask = pd.Series(day_mask_raw_values, index=df.index)
                night_mask = ~day_mask

            except Exception as e:
                print(f"Error calculating day/night mask: {e}")
                return None

            for col in data_cols:
                if df[col].isna().all():
                    print(f"Skipping {col} at {location}_{plant_type} - all values are NA")
                    continue

                print(f"\n{'=' * 60}")
                print(f"Processing {col} at {location}_{plant_type}")
                print(f"{'=' * 60}")

                column_series = df[col].copy()

                # Save day mask
                day_mask_df = pd.DataFrame(
                    {"day_mask": day_mask, col: df[col], "lat": current_df_latitude, "long": current_df_longitude}
                )
                day_mask_path = day_mask_dir / f"{col}_day_mask.csv"
                day_mask_df.to_csv(day_mask_path, index=True)

                # STEP 1: REVERSED MEASUREMENT DETECTION
                print("\n[1/4] Reversed Measurement Detection")
                print("-" * 60)

                local_time_series = (
                    df["solar_TIMESTAMP"] if "solar_TIMESTAMP" in df.columns else pd.Series(df.index, index=df.index)
                )

                # A day is REMOVED if EITHER condition is met:
                #   1. Day average < Night average - 0.5 (original check)
                #   2. >30% of top 10% values occur during solar night (improved peak check)
                reversed_mask = self._detect_and_remove_reversed_days(
                    df_col=column_series,
                    day_mask=day_mask,
                    local_time=local_time_series,
                    peak_quantile=0.85,  # Top 15% values = "peaks"
                    max_night_peak_ratio=0.30,  # Flag if >30% of peaks at solar night
                )
                reversed_info_df = pd.DataFrame(
                    {
                        "timestamp": df.index,
                        "solar_TIMESTAMP": df["solar_TIMESTAMP"] if "solar_TIMESTAMP" in df.columns else df.index,
                        "value": column_series,
                        "is_reversed": reversed_mask,
                        "is_daytime": day_mask,
                    }
                )
                reversed_path = reversed_dir / f"{col}_reversed.csv"
                reversed_info_df.to_csv(reversed_path, index=False)
                if reversed_mask.any():
                    df.loc[reversed_mask, col] = np.nan
                    column_series = df[col].copy()
                    print(f"  Removed {reversed_mask.sum()} data points due to reversed day detection.")
                else:
                    print("  No reversed days detected.")

                # STEP 2: BASELINE DRIFT CORRECTION
                print("\n[2/4] Baseline Drift Correction")
                print("-" * 60)
                if location == "DEU_HIN" and plant_type == "OAK_sapf_data":
                    print(f"  Applying baseline drift correction for {location}_{plant_type}...")
                    corrected_series, baseline = self._correct_baseline_drift(
                        series=df[col], day_mask=day_mask, method="nighttime_quantile", window_days=1, quantile=0.05
                    )

                    baseline_dir = self.paths.sap_baseline_drift_corrected_dir
                    baseline_dir.mkdir(parents=True, exist_ok=True)

                    baseline_df = pd.DataFrame(
                        {
                            "timestamp": df.index,
                            "solar_TIMESTAMP": df["solar_TIMESTAMP"] if "solar_TIMESTAMP" in df.columns else df.index,
                            "original": df[col],
                            "baseline": baseline,
                            "corrected": corrected_series,
                        }
                    )
                    baseline_df.to_csv(baseline_dir / f"{col}_baseline_correction.csv", index=False)

                    df[col] = corrected_series

                    print("  Baseline correction applied")
                    print(f"  Mean baseline shift: {baseline.mean():.3f}")
                    print(f"  Max baseline shift: {baseline.max():.3f}")

                # STEP 3: OUTLIER DETECTION
                print("\n[3/4] Outlier Detection")
                print("-" * 60)

                final_col_outliers = pd.Series(False, index=column_series.index)

                day_series = column_series.copy()
                night_series = column_series.copy()
                day_series.loc[night_mask] = np.nan
                night_series.loc[day_mask] = np.nan

                if not day_series.dropna().empty:
                    print("  Processing daytime outliers...")
                    day_outliers = self._detect_outliers(
                        series=day_series, n_std=DAY_SAPFLOW_N_STD, time_window=DAY_SAPFLOW_WINDOW_POINTS, method="B"
                    )
                    final_col_outliers.loc[day_outliers[day_outliers].index] = True

                if not night_series.dropna().empty:
                    print("  Processing nighttime outliers...")
                    night_outliers = self._detect_outliers(
                        series=night_series,
                        n_std=NIGHT_SAPFLOW_N_STD,
                        time_window=NIGHT_SAPFLOW_WINDOW_POINTS,
                        method="B",
                    )
                    final_col_outliers.loc[night_outliers[night_outliers].index] = True

                outlier_info_df = pd.DataFrame(
                    {
                        "timestamp": df.index,
                        "solar_TIMESTAMP": df["solar_TIMESTAMP"] if "solar_TIMESTAMP" in df.columns else df.index,
                        "value": column_series,
                        "is_outlier": final_col_outliers,
                        "is_daytime": day_mask,
                    }
                )
                outlier_path = outlier_dir / f"{col}_outliers.csv"
                outlier_info_df.to_csv(outlier_path, index=False)
                df.loc[final_col_outliers, col] = np.nan
                print(f"  Removed {final_col_outliers.sum()} outliers")

                # STEP 4: VARIABILITY FILTERING
                print("\n[4/4] Variability Filtering")
                print("-" * 60)

                column_series = df[col].copy()

                variability_mask = self._detect_variability_issues(
                    series=column_series,
                    day_mask=day_mask,
                    window_points=VARIABILITY_WINDOW_POINTS,
                    day_cv_low=DAY_CV_LOW,
                    day_cv_high=DAY_CV_HIGH,
                    day_std_low=DAY_STD_LOW,
                    day_std_high=DAY_STD_HIGH,
                    night_cv_high=NIGHT_CV_HIGH,
                    night_std_high=NIGHT_STD_HIGH,
                    col_name=col,
                )

                df.loc[variability_mask, col] = np.nan
                print(f"  Removed {variability_mask.sum()} time points due to variability issues")

                variability_stats.append(
                    {
                        "sensor": col,
                        "location": location,
                        "plant_type": plant_type,
                        "outliers_removed": final_col_outliers.sum(),
                        "variability_removed": variability_mask.sum(),
                        "total_removed": final_col_outliers.sum() + variability_mask.sum(),
                        "total_points": len(column_series),
                        "removal_rate": (final_col_outliers.sum() + variability_mask.sum()) / len(column_series) * 100,
                    }
                )

                variability_info_df = pd.DataFrame(
                    {
                        "timestamp": df.index,
                        "solar_TIMESTAMP": df["solar_TIMESTAMP"] if "solar_TIMESTAMP" in df.columns else df.index,
                        "value": column_series,
                        "is_variability_issue": variability_mask,
                        "is_daytime": day_mask,
                    }
                )
                variability_mask_path = variability_mask_dir / f"{col}_variability_mask.csv"
                variability_info_df.to_csv(variability_mask_path, index=False)

                print(f"\n  Summary for {col}:")
                print(
                    f"    Outliers: {final_col_outliers.sum()} ({final_col_outliers.sum() / len(column_series) * 100:.2f}%)"
                )
                print(
                    f"    Variability: {variability_mask.sum()} ({variability_mask.sum() / len(column_series) * 100:.2f}%)"
                )
                print(
                    f"    Total removed: {final_col_outliers.sum() + variability_mask.sum()} ({(final_col_outliers.sum() + variability_mask.sum()) / len(column_series) * 100:.2f}%)"
                )

            # ====================================================================
            # STEP 5: INCOMPLETE DAY REMOVAL (NEW)
            # ====================================================================
            print(f"\n{'=' * 60}")
            print("[5/5] Incomplete Day Removal (>50% missing daytime data)")
            print(f"{'=' * 60}")

            df, daily_completeness = self._remove_incomplete_days(
                df=df,
                day_mask=day_mask,
                completeness_threshold=0.5,  # 60% threshold
                data_columns=data_cols,
            )

            # Save completeness report
            completeness_dir = outliers_removed_dir / "completeness_reports"
            completeness_dir.mkdir(parents=True, exist_ok=True)
            completeness_path = completeness_dir / f"{location}_{plant_type}_daily_completeness.csv"
            daily_completeness.to_csv(completeness_path, index=False)
            print(f"  Saved completeness report to {completeness_path}")

            # ====================================================================
            # Save results
            # ====================================================================

            if variability_stats:
                stats_df = pd.DataFrame(variability_stats)
                stats_path = variability_filtered_dir / f"{location}_{plant_type}_filtering_stats.csv"
                stats_df.to_csv(stats_path, index=False)
                print(f"\nSaved filtering statistics to {stats_path}")

            self.outlier_removed_data[location][plant_type] = df
            outliers_removed_path = outliers_removed_dir / f"{location}_{plant_type}_outliers_removed.csv"
            df.to_csv(outliers_removed_path)
            print(f"\nFinished processing for {location}_{plant_type}")

            # ====================================================================
            # OPTIONAL: Gap-filling (after QC, before daily resampling)
            # ====================================================================
            if self._gap_filler is not None:
                print(f"\n{'=' * 60}")
                print("[GAP-FILL] Hierarchical gap-filling")
                print(f"{'=' * 60}")
                data_cols_for_fill = [
                    col for col in df.columns if col not in ["solar_TIMESTAMP", "TIMESTAMP_LOCAL", "lat", "long"]
                ]
                pre_fill_df = df[data_cols_for_fill].copy()
                pre_nans = pre_fill_df.isna().sum().sum()
                df_filled = self._gap_filler.fill_dataframe(pre_fill_df)
                # Preserve non-data columns
                for col in df.columns:
                    if col not in data_cols_for_fill:
                        df_filled[col] = df[col]
                df = df_filled[df.columns]  # restore original column order
                post_nans = df[data_cols_for_fill].isna().sum().sum()
                print(f"  Gaps before: {pre_nans}, after: {post_nans}, filled: {pre_nans - post_nans}")
                # Save gap-filled version
                gap_filled_dir = outliers_removed_dir.parent / "gap_filled"
                gap_filled_dir.mkdir(parents=True, exist_ok=True)
                gap_filled_path = gap_filled_dir / f"{location}_{plant_type}_gap_filled.csv"
                df.to_csv(gap_filled_path)
                print(f"  Saved gap-filled data to {gap_filled_path}")
                # Save per-column gap-fill masks (for visualization)
                gap_masks_dir = outliers_removed_dir.parent / "gap_filled_masks"
                gap_masks_dir.mkdir(parents=True, exist_ok=True)
                solar_ts = df["solar_TIMESTAMP"] if "solar_TIMESTAMP" in df.columns else None
                for col in data_cols_for_fill:
                    was_nan = pre_fill_df[col].isna()
                    now_filled = df[col].notna()
                    is_gap_filled = was_nan & now_filled
                    if not is_gap_filled.any():
                        continue
                    mask_df_out = {"solar_TIMESTAMP": solar_ts, "value": df[col], "is_gap_filled": is_gap_filled}
                    if solar_ts is None:
                        del mask_df_out["solar_TIMESTAMP"]
                    import pandas as _pd
                    _pd.DataFrame(mask_df_out, index=df.index).to_csv(
                        gap_masks_dir / f"{col}_gap_filled.csv", index_label="timestamp"
                    )
                print(f"  Saved gap-fill masks to {gap_masks_dir}")

        except Exception as e:
            print(f"Error during processing: {str(e)}")
            import traceback

            traceback.print_exc()
            return None

        try:
            sap_daily_resampled_dir = self.paths.sap_daily_resampled_dir
            sap_daily_resampled_dir.mkdir(parents=True, exist_ok=True)

            data_cols = [col for col in df.columns if col not in ["solar_TIMESTAMP", "TIMESTAMP_LOCAL", "lat", "long"]]

            daily_df = df[data_cols].resample("D").mean()
            daily_df.columns = [f"{col}_mean" for col in daily_df.columns]

            sap_daily_resampled_path = sap_daily_resampled_dir / f"{location}_{plant_type}_daily.csv"
            daily_df.to_csv(sap_daily_resampled_path, index=True)
            print(f"Saved daily resampled data to {sap_daily_resampled_path}")

            return daily_df

        except Exception as e:
            print(f"Error during daily resampling: {str(e)}")
            return None

    def _correct_baseline_drift(
        self,
        series: pd.Series,
        day_mask: pd.Series,
        method: str = "nighttime_quantile",
        window_days: int = 7,
        quantile: float = 0.05,
    ) -> pd.Series:
        """
        Correct baseline drift in sap flow data

        Args:
            series: Sap flow data
            day_mask: Boolean mask for daytime
            method: 'nighttime_quantile', 'nighttime_min', or 'rolling_min'
            window_days: Window size in days for baseline tracking
            quantile: Quantile to use for nighttime baseline (0.05 = 5th percentile)

        Returns:
            Baseline-corrected series
        """
        corrected = series.copy()
        night_mask = ~day_mask

        if method == "nighttime_quantile":
            # Use nighttime quantile as baseline (more robust than min)
            night_values = series.copy()
            night_values[day_mask] = np.nan

            # Calculate rolling nighttime baseline
            window_points = self._get_timewindow(window_days * 24 * 3600, series.index)
            baseline = night_values.rolling(
                window=window_points, center=True, min_periods=max(1, window_points // 4)
            ).quantile(quantile)

            # Interpolate gaps in baseline
            baseline = baseline.interpolate(method="linear", limit_direction="both")

        elif method == "nighttime_min":
            # Use nighttime minimum as baseline
            night_values = series.copy()
            night_values[day_mask] = np.nan

            window_points = self._get_timewindow(window_days * 24 * 3600, series.index)
            baseline = night_values.rolling(
                window=window_points, center=True, min_periods=max(1, window_points // 4)
            ).min()

            baseline = baseline.interpolate(method="linear", limit_direction="both")

        elif method == "rolling_min":
            # Use overall rolling minimum (use cautiously)
            window_points = self._get_timewindow(window_days * 24 * 3600, series.index)
            baseline = series.rolling(
                window=window_points, center=True, min_periods=max(1, window_points // 4)
            ).quantile(quantile)

            baseline = baseline.interpolate(method="linear", limit_direction="both")

        # Subtract baseline
        corrected = series - baseline

        # Ensure no negative values after correction
        corrected[corrected < 0] = 0

        return corrected, baseline

    def _detect_and_remove_reversed_days(
        self,
        df_col: pd.Series,
        day_mask: pd.Series,
        local_time: pd.Series,
        peak_quantile: float = 0.90,
        max_night_peak_ratio: float = 0.30,
    ) -> pd.Series:
        """
        Detects days where sap flow appears reversed or inverted.

        A day is flagged as reversed if EITHER condition is met:
        1. Average Day Flow * 0.8 < Average Night Flow - 0.5 (original check)
        2. >X% of top Y% values occur during SOLAR night (improved check)

        Args:
            df_col: The sap flow density data series.
            day_mask: Boolean series (True=Day, False=Night) - SOLAR BASED from pvlib.
            local_time: Series containing solar datetime objects (solar_TIMESTAMP).
            peak_quantile: Quantile threshold for "high values" (default 0.90 = top 10%)
            max_night_peak_ratio: Maximum allowed ratio of high values at night
                                (default 0.30 = 30%). If more than 30% of top 10%
                                values occur at night, the peak check fails.

        Returns:
            A boolean mask (True = Reversed/Remove, False = Keep).
        """
        # === FIX: Use solar_TIMESTAMP for date grouping ===
        solar_ts = pd.to_datetime(local_time)
        solar_dates = pd.Series(solar_ts.dt.date.values, index=df_col.index)

        temp = pd.DataFrame({"val": df_col, "is_day": day_mask, "solar_date": solar_dates})
        # === END FIX ===

        reversed_mask = pd.Series(False, index=df_col.index)

        # Group by SOLAR Date (not UTC date)
        grouped = temp.groupby("solar_date")

        # Statistics
        total_days = 0
        days_failed_avg = 0
        days_failed_peak = 0
        days_removed = 0

        for date_obj, group in grouped:
            if group["val"].dropna().empty:
                continue

            total_days += 1

            # Get day and night data using SOLAR-BASED mask
            day_vals = group.loc[group["is_day"], "val"].dropna()
            night_vals = group.loc[~group["is_day"], "val"].dropna()

            # Skip if insufficient data
            if len(day_vals) < 3 or len(night_vals) < 3:
                continue

            # --- Calculate metrics ---
            mean_day = day_vals.mean()
            mean_night = night_vals.mean()

            # Handle edge cases
            if pd.isna(mean_day):
                mean_day = 0
            if pd.isna(mean_night):
                mean_night = 0

            # =====================================================================
            # CHECK 1: Day vs Night Average (ORIGINAL)
            # Day average should be >= night average (with 0.5 tolerance)
            # =====================================================================
            check1_passed = (mean_day * 0.8) >= (mean_night - 0.5)

            if not check1_passed:
                days_failed_avg += 1

            # =====================================================================
            # CHECK 2: Peak Distribution (IMPROVED - Solar-based quantile)
            # Uses pvlib day_mask, NOT fixed hours like 22:00-05:00
            # =====================================================================
            check2_passed = True
            night_peak_ratio = 0.0

            all_vals = group["val"].dropna()

            if len(all_vals) >= 10:  # Need enough data points
                # Find threshold for "high values" (e.g., top 10%)
                peak_threshold = all_vals.quantile(peak_quantile)

                # Get all high value points
                peak_mask = all_vals >= peak_threshold
                peak_indices = all_vals[peak_mask].index

                if len(peak_indices) > 0:
                    # Count how many peaks occur during SOLAR night
                    # Using the solar-based day_mask (not fixed hours!)
                    peaks_at_night = (~group.loc[peak_indices, "is_day"]).sum()
                    total_peaks = len(peak_indices)
                    night_peak_ratio = peaks_at_night / total_peaks

                    # Fail if too many peaks at night
                    check2_passed = night_peak_ratio <= max_night_peak_ratio

            if not check2_passed:
                days_failed_peak += 1

            # =====================================================================
            # FINAL DECISION: Either check fails = reversed
            # =====================================================================
            is_reversed = (not check1_passed) or (not check2_passed)

            if is_reversed:
                reversed_mask.loc[group.index] = True
                days_removed += 1

        # --- Summary ---
        print("    [Reversed Detection]")
        print(f"      Check 1 - Day avg * 0.8 < Night avg - 0.5: {days_failed_avg}/{total_days} days failed")
        print(
            f"      Check 2 - >{max_night_peak_ratio:.0%} of top {(1 - peak_quantile) * 100:.0f}% at solar night: {days_failed_peak}/{total_days} days failed"
        )
        print(
            f"      REMOVED (either failed): {days_removed}/{total_days} days ({days_removed / max(1, total_days) * 100:.1f}%)"
        )

        return reversed_mask

    def _detect_variability_issues(
        self,
        series: pd.Series,
        day_mask: pd.Series,
        window_points: int,
        day_cv_low: float,
        day_cv_high: float,
        day_std_low: float,
        day_std_high: float,
        night_cv_high: float,
        night_std_high: float,
        col_name: str = None,
    ) -> pd.Series:
        """
        Detect time periods with abnormal variability (too high or too low).
        Returns a boolean mask indicating which time points to remove.

        Args:
            series: Time series data
            day_mask: Boolean mask for daytime
            window_points: Rolling window size
            day_cv_low, day_cv_high: CV thresholds for daytime
            day_std_low, day_std_high: STD thresholds for daytime
            night_cv_high: CV threshold for nighttime
            night_std_high: STD threshold for nighttime
            col_name: Column name for logging

        Returns:
            Boolean Series indicating time points to remove
        """
        variability_mask = pd.Series(False, index=series.index)
        night_mask = ~day_mask

        # Separate day and night
        day_series = series.copy()
        night_series = series.copy()
        day_series.loc[night_mask] = np.nan
        night_series.loc[day_mask] = np.nan

        # ====================================================================
        # DAYTIME VARIABILITY CHECKS
        # ====================================================================
        if not day_series.dropna().empty:
            print("  Checking daytime variability...")

            # Calculate rolling statistics for daytime
            day_rolling_mean = self.adaptive_centered_rolling(day_series, window_points, np.mean)
            day_rolling_std = self.adaptive_centered_rolling(day_series, window_points, np.std)

            # Coefficient of variation
            day_cv = day_rolling_std / (day_rolling_mean.abs() + 1e-10)

            # Identify problematic periods
            day_cv_too_low = (day_cv < day_cv_low) & day_mask
            day_cv_too_high = (day_cv > day_cv_high) & day_mask
            day_std_too_low = (day_rolling_std < day_std_low) & day_mask
            day_std_too_high = (day_rolling_std > day_std_high) & day_mask

            # Combine issues
            day_issues = day_cv_too_low | day_cv_too_high | day_std_too_low | day_std_too_high

            # Only flag daytime points
            day_issues = day_issues & day_mask

            variability_mask = variability_mask | day_issues

            # Report statistics
            if day_issues.sum() > 0:
                print(f"    CV too low: {day_cv_too_low.sum()} points")
                print(f"    CV too high: {day_cv_too_high.sum()} points")
                print(f"    STD too low: {day_std_too_low.sum()} points")
                print(f"    STD too high: {day_std_too_high.sum()} points")
                print(f"    Total day issues: {day_issues.sum()} points")

        # ====================================================================
        # NIGHTTIME VARIABILITY CHECKS
        # ====================================================================
        if not night_series.dropna().empty:
            print("  Checking nighttime variability...")

            # Calculate rolling statistics for nighttime
            night_rolling_mean = self.adaptive_centered_rolling(night_series, window_points, np.mean)
            night_rolling_std = self.adaptive_centered_rolling(night_series, window_points, np.std)

            # Coefficient of variation
            night_cv = night_rolling_std / (night_rolling_mean.abs() + 1e-10)

            # Identify problematic periods (nighttime should be stable)
            night_cv_too_high = (night_cv > night_cv_high) & night_mask
            night_std_too_high = (night_rolling_std > night_std_high) & night_mask

            # Combine issues
            night_issues = night_cv_too_high | night_std_too_high

            # Only flag nighttime points
            night_issues = night_issues & night_mask

            variability_mask = variability_mask | night_issues

            # Report statistics
            if night_issues.sum() > 0:
                print(f"    CV too high: {night_cv_too_high.sum()} points")
                print(f"    STD too high: {night_std_too_high.sum()} points")
                print(f"    Total night issues: {night_issues.sum()} points")

        return variability_mask

    def _remove_incomplete_days(
        self, df: pd.DataFrame, day_mask: pd.Series, completeness_threshold: float = 0.5, data_columns: list = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Remove entire days where more than a threshold of daytime data is missing.
        Checks each column INDEPENDENTLY - a day can be removed for one column but kept for another.

        Args:
            df: DataFrame with datetime index
            day_mask: Boolean Series (True = daytime)
            completeness_threshold: Minimum fraction of daytime data required (default 0.5 = 50%)
            data_columns: List of data columns to check. If None, uses all numeric columns.

        Returns:
            Tuple of (filtered DataFrame, summary DataFrame with daily completeness stats per column)
        """
        df_result = df.copy()

        if data_columns is None:
            data_columns = [
                col
                for col in df.columns
                if col not in ["TIMESTAMP", "solar_TIMESTAMP", "TIMESTAMP_LOCAL", "lat", "long", "ta"]
                and "Unnamed:" not in str(col)
            ]

        # Ensure day_mask is aligned with df index
        day_mask = day_mask.reindex(df.index).fillna(False)

        # Handle case where there's no daytime data at all
        if day_mask.sum() == 0:
            print("  WARNING: No daytime points found in day_mask. Skipping incomplete day removal.")
            return df_result, pd.DataFrame()

        # === FIX: Use SOLAR dates for date grouping (matches day_mask calculation) ===
        if "solar_TIMESTAMP" in df.columns:
            solar_ts = pd.to_datetime(df["solar_TIMESTAMP"])
            dates = pd.Series(solar_ts.dt.date.values, index=df.index)
            print("  Using solar_TIMESTAMP for date grouping")
        elif "TIMESTAMP_LOCAL" in df.columns:
            local_ts = pd.to_datetime(df["TIMESTAMP_LOCAL"])
            dates = pd.Series(local_ts.dt.date.values, index=df.index)
            print("  Using TIMESTAMP_LOCAL for date grouping (solar_TIMESTAMP not found)")
        else:
            dates = pd.Series(df.index.date, index=df.index)
            print("  WARNING: Using UTC dates for grouping")
        # === END FIX ===

        # Track statistics per column
        all_stats = []
        total_removed_per_column = {}

        print("\n[INCOMPLETE DAY REMOVAL]")
        print(f"  Completeness threshold: {completeness_threshold * 100:.0f}% of daytime data required")
        print(f"  Checking {len(data_columns)} columns independently...")
        print("-" * 60)

        for col in data_columns:
            if col not in df_result.columns:
                continue

            days_removed = 0
            points_removed = 0

            # Group by date for this column
            for date in dates.unique():
                date_mask = dates == date
                day_data = df_result.loc[date_mask]
                day_daylight = day_mask.loc[date_mask]

                # Count expected daytime points
                expected_daytime_points = day_daylight.sum()

                if expected_daytime_points == 0:
                    # No daytime for this date - record but don't remove
                    all_stats.append(
                        {
                            "date": date,
                            "column": col,
                            "expected_daytime_points": 0,
                            "valid_daytime_points": 0,
                            "completeness": np.nan,
                            "removed": False,
                        }
                    )
                    continue

                # Count actual valid daytime data for THIS column
                daytime_indices = day_daylight[day_daylight].index
                valid_daytime_count = day_data.loc[daytime_indices, col].notna().sum()

                # Calculate completeness for this column on this day
                completeness = valid_daytime_count / expected_daytime_points

                # Store stats
                all_stats.append(
                    {
                        "date": date,
                        "column": col,
                        "expected_daytime_points": expected_daytime_points,
                        "valid_daytime_points": valid_daytime_count,
                        "completeness": completeness,
                        "removed": completeness < completeness_threshold,
                    }
                )

                # Remove this day's data for this column if incomplete
                if completeness < completeness_threshold:
                    df_result.loc[date_mask, col] = np.nan
                    days_removed += 1
                    points_removed += date_mask.sum()

            total_removed_per_column[col] = {"days_removed": days_removed, "points_removed": points_removed}

            if days_removed > 0:
                total_days = len(dates.unique())
                print(f"  {col}: Removed {days_removed}/{total_days} days ({days_removed / total_days * 100:.1f}%)")

        # Create summary DataFrame
        summary_df = pd.DataFrame(all_stats)

        # Print overall summary
        print("-" * 60)
        if not summary_df.empty and "removed" in summary_df.columns:
            removed_df = summary_df[summary_df["removed"] == True]
            if not removed_df.empty:
                total_days_any_removed = removed_df.groupby("date").size().shape[0]
            else:
                total_days_any_removed = 0
            total_days = len(dates.unique())
            print(f"  Total days with at least one column removed: {total_days_any_removed}/{total_days}")
        else:
            print("  No completeness data to summarize")

        return df_result, summary_df

    def _get_timewindow(self, time_window: int, timesteps_data: pd.Index | pd.Series | pd.DataFrame) -> int:
        """
        Calculate the number of points in a time window based on the determined timestep.
        """
        # Determine the timestamp Index from the input data
        if isinstance(timesteps_data, pd.DataFrame):
            if "TIMESTAMP" in timesteps_data.columns:
                timesteps = timesteps_data["TIMESTAMP"]
            elif timesteps_data.index.name == "TIMESTAMP":
                timesteps = timesteps_data.index
            else:
                raise ValueError(
                    "DataFrame must have a 'TIMESTAMP' column or its index named 'TIMESTAMP' to determine the timestep."
                )
        elif isinstance(timesteps_data, pd.Series):
            if timesteps_data.index.name == "TIMESTAMP":
                timesteps = timesteps_data.index
            else:
                if pd.api.types.is_datetime64_any_dtype(timesteps_data):
                    timesteps = timesteps_data
                else:
                    raise ValueError(
                        "Series must have a 'TIMESTAMP' index or contain datetime values directly to determine the timestep."
                    )
        elif isinstance(timesteps_data, pd.Index):
            timesteps = timesteps_data
        else:
            raise TypeError(
                "Input 'timesteps_data' must be a pandas Index, Series, or DataFrame with a 'TIMESTAMP' column or index."
            )

        # Ensure the extracted timesteps are datetime objects
        if not pd.api.types.is_datetime64_any_dtype(timesteps):
            try:
                timesteps = pd.to_datetime(timesteps)
            except Exception as e:
                raise ValueError(f"Could not convert 'timesteps_data' to datetime objects: {e}")

        if len(timesteps) < 2:
            raise ValueError(
                "Timestamp data must contain at least two points to calculate the difference and determine the timestep."
            )

        # Calculate the median timestep in seconds
        timestep_seconds = timesteps.to_series().diff().median().total_seconds()

        if timestep_seconds <= 0:
            raise ValueError(
                f"Calculated timestep is non-positive ({timestep_seconds} seconds). "
                "This can happen if timestamps are identical or not in increasing order."
            )

        return int(time_window / timestep_seconds)

    def _calculate_daytime_mask(
        self, timestamps: pd.DatetimeIndex, lat: float, lon: float, elevation: float, elevation_threshold: float = 0
    ) -> pd.Series:
        """
        Calculates a boolean mask indicating daytime based on solar elevation.
        """
        if not isinstance(timestamps, pd.DatetimeIndex):
            raise ValueError("Timestamps must be a pandas DatetimeIndex.")
        if timestamps.tzinfo is None:
            print("Warning: Timestamps are timezone-naive. Assuming UTC for solar position calculation.")
            timestamps = timestamps.tz_localize("UTC")

        solar_position = pvlib.solarposition.get_solarposition(timestamps, lat, lon, altitude=elevation)
        return solar_position["elevation"] > elevation_threshold

    def adaptive_centered_rolling(self, series, window_size, func=np.mean):
        """
        Apply rolling function with adaptive centered windows.
        """
        if func == np.mean:
            func = np.nanmean
        elif func == np.median:
            func = np.nanmedian
        if window_size >= len(series):
            return pd.Series([func(series)] * len(series), index=series.index)

        result = pd.Series(index=series.index, dtype=float)
        half_window = window_size // 2

        for i in range(len(series)):
            ideal_start = i - half_window
            ideal_end = i + half_window + 1

            actual_start = max(0, ideal_start)
            actual_end = min(len(series), ideal_end)

            current_window_size = actual_end - actual_start
            if current_window_size < window_size:
                if actual_start > 0:
                    extend_left = min(actual_start, window_size - current_window_size)
                    actual_start -= extend_left
                    current_window_size = actual_end - actual_start

                if current_window_size < window_size and actual_end < len(series):
                    extend_right = min(len(series) - actual_end, window_size - current_window_size)
                    actual_end += extend_right

            window_data = series.iloc[actual_start:actual_end]
            result.iloc[i] = func(window_data)

        return result

    def _detect_outliers(
        self, series: pd.Series, n_std: float = 3, time_window: int = 1440, method: str = "C"
    ) -> pd.Series:
        """
        Improved outlier detection with better edge handling
        """
        outliers = pd.Series(False, index=series.index)

        if method == "A":
            # Monthly grouping method
            grouped = series.groupby([series.index.year, series.index.month])

            for (year, month), group in grouped:
                if len(group) > 0:
                    mean = group.mean()
                    std = group.std()

                    month_mask = (series.index.year == year) & (series.index.month == month)
                    monthly_data = series[month_mask]

                    monthly_outliers = np.abs(monthly_data - mean) > n_std * std
                    outliers[monthly_data.index] = monthly_outliers

                    n_outliers = monthly_outliers.sum()
                    print(
                        f"Month {month}/{year}: "
                        f"{n_outliers} outliers out of {len(monthly_data)} points "
                        f"({(n_outliers / len(monthly_data) * 100):.1f}%)"
                    )

        elif method == "B":
            # Improved rolling with adaptive centering
            print(f"Using adaptive centered rolling (window={time_window})")

            rolling_mean = self.adaptive_centered_rolling(series, time_window, np.mean)
            rolling_std = self.adaptive_centered_rolling(series, time_window, np.std)

            rolling_std = rolling_std.replace(0, rolling_std.mean())

            outliers = np.abs(series - rolling_mean) > (n_std * rolling_std)

            print(f"Adaptive rolling outliers detected: {outliers.sum()} out of {len(series)} points")

        elif method == "C":
            # Improved MAD with adaptive centering
            print(f"Using adaptive centered MAD (window={time_window})")

            def rolling_mad(window_data):
                if len(window_data) <= 1:
                    return 0
                median_val = np.nanmedian(window_data)
                return np.nanmedian(np.abs(window_data - median_val))

            rolling_median = self.adaptive_centered_rolling(series, time_window, np.median)
            rolling_mad_values = self.adaptive_centered_rolling(series, time_window, rolling_mad)

            scaled_mad = rolling_mad_values / 0.6745
            epsilon = 1e-10
            scaled_mad = scaled_mad + epsilon

            deviations = np.abs(series - rolling_median)
            outliers = deviations > (n_std * scaled_mad)

            print(f"Adaptive MAD outliers detected: {outliers.sum()} out of {len(series)} points")

        elif method == "D":
            # Standard pandas rolling
            print(f"Using standard pandas rolling (window={time_window})")

            rolling_mean = series.rolling(window=time_window, center=True, min_periods=max(1, time_window // 2)).mean()

            rolling_std = series.rolling(window=time_window, center=True, min_periods=max(1, time_window // 2)).std()

            rolling_mean = rolling_mean.fillna(series.mean())
            rolling_std = rolling_std.fillna(series.std())

            outliers = np.abs(series - rolling_mean) > (n_std * rolling_std)

            nan_count = series.rolling(time_window, center=True).mean().isna().sum()
            print(f"Standard rolling outliers detected: {outliers.sum()} out of {len(series)} points")
            print(f"Original NaN values in rolling: {nan_count}")

        else:
            raise ValueError(f"Unknown method: {method}")

        return outliers

    def plot_all_plants(
        self,
        files_to_load: list[Path] | None = None,
        figsize=(12, 6),
        save_dir: str = None,
        skip_empty: bool = True,
        plot_limit: int = None,
        progress_update: bool = True,
    ):
        """
        Plot individual trees for all sites with enhanced error handling and progress tracking
        """
        # empty no use dictionary to free memory
        self.sapf_raw_data = {}
        self.sapflow_data = {}

        # if self.outlier_removed_data is empty, load outlier_removed_data from saved files
        if not self.outlier_removed_data:
            print("Loading outlier removed data from saved files...")
            for file in files_to_load:
                parts = file.stem.split("_")
                location = "_".join(parts[:2])
                plant_type = "_".join(parts[2:-2])
                df = pd.read_csv(file, parse_dates=["TIMESTAMP"])
                df = df.set_index("TIMESTAMP")
                if location not in self.outlier_removed_data:
                    self.outlier_removed_data[location] = {}
                self.outlier_removed_data[location][plant_type] = df

        # Create summary dictionary to store processing results
        summary = {
            "total_locations": len(self.outlier_removed_data),
            "processed_locations": 0,
            "successful_plots": 0,
            "failed_plots": 0,
            "skipped_empty": 0,
            "errors": [],
        }

        # Create save directory if specified
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

        # Process each location
        total_locations = len(self.outlier_removed_data)
        for loc_idx, location in enumerate(self.outlier_removed_data, 1):
            if progress_update:
                print(f"\nProcessing location {loc_idx}/{total_locations}: {location}")

            for plant_type in self.outlier_removed_data[location]:
                try:
                    # Get data for current location/plant type
                    data = self.outlier_removed_data[location][plant_type]
                    plot_columns = [
                        col for col in data.columns if col != "solar_TIMESTAMP" and col != "TIMESTAMP_LOCAL"
                    ]

                    # Check if data is empty
                    if skip_empty and all(data[col].isna().all() for col in plot_columns):
                        if progress_update:
                            print(f"Skipping {location}_{plant_type} - no valid data")
                        summary["skipped_empty"] += 1
                        continue

                    # Limit number of plots if specified
                    if plot_limit:
                        plot_columns = plot_columns[:plot_limit]

                    if progress_update:
                        print(f"\nProcessing {location}_{plant_type}")
                        print(f"Plotting {len(plot_columns)} trees")

                    # Create location-specific save directory
                    if save_dir:
                        location_save_dir = save_path / f"{location}_{plant_type}"
                        location_save_dir.mkdir(parents=True, exist_ok=True)
                    else:
                        location_save_dir = None

                    # Plot individual plants
                    self.plot_individual_plants(
                        location=location, plant_type=plant_type, figsize=figsize, save_dir=location_save_dir
                    )

                    summary["successful_plots"] += len(plot_columns)

                except Exception as e:
                    error_msg = f"Error processing {location}_{plant_type}: {str(e)}"
                    print(f"Error: {error_msg}")
                    summary["errors"].append(error_msg)
                    summary["failed_plots"] += 1

            summary["processed_locations"] += 1

        # Print final summary
        if progress_update:
            print("\nProcessing Summary:")
            print(f"Total locations processed: {summary['processed_locations']}/{summary['total_locations']}")
            print(f"Successful plots: {summary['successful_plots']}")
            print(f"Failed plots: {summary['failed_plots']}")
            print(f"Skipped empty locations: {summary['skipped_empty']}")

            if summary["errors"]:
                print("\nErrors encountered:")
                for error in summary["errors"]:
                    print(f"- {error}")

        return summary

    def plot_individual_plants(self, location: str, plant_type: str, figsize=(12, 6), save_dir: str = None):
        """
        Create individual plots for each plant/tree with outlier and variability detection,
        using local time for the x-axis.
        """
        unit = "cm3 * cm-2 * h-1"
        if location not in self.outlier_removed_data or plant_type not in self.outlier_removed_data[location]:
            raise ValueError(f"No data found for {location}_{plant_type}")

        data = self.outlier_removed_data[location][plant_type].copy()

        if "TIMESTAMP_LOCAL" not in data.columns:
            raise KeyError(f"TIMESTAMP_LOCAL column not found for {location}_{plant_type}.")

        plot_columns = [
            col
            for col in data.columns
            if col not in ["solar_TIMESTAMP", "TIMESTAMP", "TIMESTAMP_LOCAL", "lat", "long"]
            and "Unnamed:" not in str(col)
            and len(col) > 0
        ]

        print(f"\nPlotting {location}_{plant_type}")
        print(f"Data shape: {data.shape}")
        print(f"Columns to plot: {plot_columns}")

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            print(f"Saving plots to {save_path}")

        for column in plot_columns:
            if data[column].isna().all():
                print(f"Skipping {column} - all values are NA")
                continue

            # Load flagged data
            flag_path = self.paths.sap_filtered_dir / f"{column}_flagged.csv"
            if flag_path.exists():
                flagged_data = pd.read_csv(flag_path, parse_dates=["TIMESTAMP"]).set_index("TIMESTAMP")
            else:
                flagged_data = None

            # Load outliers and variability data
            outlier_path = self.paths.sap_outliers_dir / f"{column}_outliers.csv"
            if outlier_path.exists():
                outliers_df = pd.read_csv(outlier_path, parse_dates=["timestamp"]).set_index("timestamp")
            else:
                outliers_df = None

            print(f"\nColumn: {column}")
            print(f"Non-null values: {data[column].count()}")
            print(f"Value range: {data[column].min()} to {data[column].max()}")

            fig = plt.figure(figsize=(figsize[0], figsize[1] * 1.5))
            gs = plt.GridSpec(2, 1, height_ratios=[3, 1])
            ax1 = fig.add_subplot(gs[0])

            valid_data_series = data[column].dropna()

            if not valid_data_series.empty:
                local_timestamps = data.loc[valid_data_series.index, "TIMESTAMP_LOCAL"]

                ax1.plot(
                    local_timestamps,
                    valid_data_series.values,
                    "-b.",
                    alpha=0.5,
                    label="Normal data",
                    linewidth=1,
                    markersize=3,
                )

                # Plot flagged data
                if flagged_data is not None and not flagged_data[column].dropna().empty:
                    flagged_points = flagged_data[column].dropna()
                    local_flagged_ts = data.loc[flagged_points.index, "TIMESTAMP_LOCAL"]
                    ax1.scatter(
                        local_flagged_ts,
                        flagged_points.values,
                        color="red",
                        alpha=0.7,
                        label="Flagged data",
                        marker="x",
                        s=50,
                        zorder=3,
                    )

                # Plot outliers and variability issues separately
                if outliers_df is not None:
                    if "is_outlier" in outliers_df.columns:
                        outlier_mask = outliers_df["is_outlier"]
                        outlier_points = outliers_df.loc[outlier_mask, "value"]
                        if not outlier_points.empty:
                            local_outlier_ts = data.loc[outlier_points.index, "TIMESTAMP_LOCAL"]
                            ax1.scatter(
                                local_outlier_ts,
                                outlier_points.values,
                                color="orange",
                                alpha=0.7,
                                label="Outliers",
                                marker="o",
                                s=50,
                                zorder=3,
                            )

                    if "is_variability_issue" in outliers_df.columns:
                        variability_mask = outliers_df["is_variability_issue"]
                        variability_points = outliers_df.loc[variability_mask, "value"]
                        if not variability_points.empty:
                            local_var_ts = data.loc[variability_points.index, "TIMESTAMP_LOCAL"]
                            ax1.scatter(
                                local_var_ts,
                                variability_points.values,
                                color="purple",
                                alpha=0.7,
                                label="Variability issues",
                                marker="s",
                                s=30,
                                zorder=3,
                            )

                ax1.set_title(f"Sap Flow Time Series - {location} {plant_type}\nTree {column}", pad=20)
                ax1.set_xlabel("Date (Local Time)")
                ax1.set_ylabel(f"Sap Flow ({unit})")
                ax1.grid(True, alpha=0.3)
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
                ax1.legend()

                # Data quality info
                valid_percent = (len(valid_data_series) / len(data)) * 100 if len(data) > 0 else 0
                flagged_count = len(flagged_data[column].dropna()) if flagged_data is not None else 0
                outlier_count = (
                    outliers_df["is_outlier"].sum()
                    if outliers_df is not None and "is_outlier" in outliers_df.columns
                    else 0
                )
                variability_count = (
                    outliers_df["is_variability_issue"].sum()
                    if outliers_df is not None and "is_variability_issue" in outliers_df.columns
                    else 0
                )

                info_text = (
                    f"Valid data: {valid_percent:.1f}%\n"
                    f"Flagged: {flagged_count}\n"
                    f"Outliers: {outlier_count}\n"
                    f"Variability: {variability_count}\n"
                    f"Range: {valid_data_series.min():.3f} to {valid_data_series.max():.3f}"
                )
                ax1.text(
                    0.02,
                    0.98,
                    info_text,
                    transform=ax1.transAxes,
                    va="top",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                )

                plt.tight_layout()

                if save_dir:
                    filename = f"tree_{column}.png"
                    fig.savefig(save_path / filename, bbox_inches="tight", dpi=300)
                    plt.close(fig)
                    print(f"Saved {filename}")
                else:
                    plt.show()
                    plt.close(fig)
            else:
                plt.close(fig)
                print(f"No valid data to plot for {column}")

    def plot_histogram(self, save_dir: str = None):
        """Plot histogram of sapflow data"""
        for location in self.sapf_raw_data:
            for plant_type in self.sapf_raw_data[location]:
                df = self.sapf_raw_data[location][plant_type]
                plot_columns = [
                    col
                    for col in df.columns
                    if col != "solar_TIMESTAMP"
                    and col != "TIMESTAMP"
                    and col != "TIMESTAMP_LOCAL"
                    and len(col) > 0
                    and col not in ["lat", "long"]
                    and "Unnamed:" not in str(col)
                ]
                df = df.replace("NA", np.nan)
                for column in plot_columns:
                    try:
                        series = pd.to_numeric(df[column], errors="coerce")
                        series = series.dropna()

                        if len(series) > 0:
                            fig = plt.figure(figsize=(12, 6))
                            ax = fig.add_subplot(111)

                            sns.histplot(series, bins=50, kde=True, ax=ax)
                            ax.set_title(
                                f"Histogram of {column}\n"
                                f"Mean: {series.mean():.2f}, Std: {series.std():.2f}\n"
                                f"Valid points: {len(series)}"
                            )
                            ax.set_xlabel("Sap Flow")
                            ax.set_ylabel("Frequency")

                            if save_dir:
                                save_path = Path(save_dir)
                                save_path.mkdir(parents=True, exist_ok=True)
                                plt.savefig(save_path / f"{column}_histogram.png")
                                print(f"Saved histogram to {save_path / f'{column}_histogram.png'}")
                            plt.close()
                        else:
                            print(f"No valid data for {column}")

                    except Exception as e:
                        print(f"Error plotting histogram for {column}: {str(e)}")
                        continue

    def get_summary(self, location: str, plant_type: str) -> dict:
        """Get summary statistics for a specific site"""
        if location not in self.sapflow_data or plant_type not in self.sapflow_data[location]:
            raise ValueError(f"No data found for {location}_{plant_type}")

        data = self.sapflow_data[location][plant_type]
        plot_columns = [col for col in data.columns if col != "solar_TIMESTAMP" and col != "TIMESTAMP_LOCAL"]

        summary = {
            "location": location,
            "plant_type": plant_type,
            "time_range": {
                "start": data.index.min().strftime("%Y-%m-%d"),
                "end": data.index.max().strftime("%Y-%m-%d"),
                "duration_days": (data.index.max() - data.index.min()).days,
            },
            "trees": len(plot_columns),
            "measurements": len(data),
            "missing_data": (data[plot_columns].isna().sum() / len(data) * 100).mean(),
        }

        return summary

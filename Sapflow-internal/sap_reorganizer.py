import os
import re
from pathlib import Path

import pandas as pd
import pyreadr

# --- CONFIGURATION ---

INPUT_DIR = "Sapflow-internal"

OUTPUT_DIR = "Sapflow_SAPFLUXNET_format"

# Supported file extensions

SUPPORTED_EXTENSIONS = {".xlsx", ".xls", ".csv", ".txt", ".tsv", ".dat", ".rds"}

# Common delimiters to try for text files

DELIMITERS = [",", "\t", ";", "|", " "]

# Date format patterns to try

DATE_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y",
    "%Y%m%d%H%M%S",
    "%Y%m%d",
    "%d.%m.%Y %H:%M:%S",  # Added European format with dots
    "%d.%m.%Y",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d",
    "%d-%b-%Y %H:%M:%S",  # ADD THIS LINE (e.g., 09-Oct-2018 17:30:00)
    "%d-%b-%Y",  # ADD THIS TOO (date only version)
    "ISO8601",
]

# Column name mappings for flexible matching

TIMESTAMP_NAMES = [
    "timestamp",
    "Hour",
    "month",
    "date",
    "datetime",
    "date.time",
    "date_utc+1",
    "time",
    "datum",
    "fecha",
    "data",
    "timestamp_utc",
    "doy",
    "date_time",
    "measurement_time",
    "observation_time",
]

TREE_ID_NAMES = [
    "tree_id",
    "treeid",
    "treenumber",
    "tree_number",
    "treenumb",
    "treenumber/treen",
    "treen",  # CH-Dav uses 'TreeNumber/TreeN'
    "tid",
    "tree",
    "plant_id",
    "plantid",
    "plot",
    "tree_code",
    "sample_id",
    "series",
]

# Expanded list to catch more column patterns

VALUE_NAMES = [
    "value",
    "sapflow",
    "sap",
    "sfd",
    "sfdm",
    "spd",
    "flow",
    "flux",
    "tdsv",
    "tree_sapflow_kg/h",
    "Js_outer",
    "Js_inner",
    "velocity",
    "rate",
    "js",
    "sap_flow",
    "sap_velocity",
    "transpiration",
    "water_flow",
    "flow_rate",
    "flux_density",
    "sap_flux",
    "sf",
    "vfd",
    "measurement",
]

ENV_NAMES = [
    "temperature",
    "Ta",
    "Tair",
    "air_temperature",
    "temp",
    "RH",
    "rH",
    "relative_humidity",
    "precipitation",
    "precipitations(mm)",
    "precip",
    "evapotranspiration",
    "Et",
    "Evapotransp. (mmol m-2s-1)",
    "soil_moisture",
    "sm",
    "SoilUR10cm",
    "SoilUR50cm",
    "SoilUR100cm",
    "humidity",
    "wind_speed",
    "ws",
    "vapor_pressure_deficit",
    "vpd",
    "shortwave_incoming_radiation",
    "sw_in",
    "solar_radiation",
    "ppfd",
    "photosynthetic_photon_flux_density",
    "ppfd_in",
    "gpp (umol m-2s-1)",
    "gpp",
]

# --- CONFIGURATION --- (add to existing)

SENSOR_ID_NAMES = ["sensor_id", "sensorid", "orientation", "sensor", "probe_id", "probeid", "probe", "probe_number"]

# --- END CONFIGURATION ---


def extract_tree_id_from_column(col_name: str) -> str | None:
    """

    Extract tree_id from column name patterns like:

    - NO-Hur_T3_SFM1E21V_sapflow -> T3

    - NO-Hur_H1_SFM1E30A_sapflow -> H1

    - ARG_MAZ_T1_sapflow -> T1

    - tree1_sapflow -> 1

    """

    # First try to match tree IDs with underscores (like H3_1, H3_2)

    match = re.search(r"[_\-]([TH]\d+_\d+)[_\-]", col_name, re.IGNORECASE)

    if match:
        return match.group(1)

    # Then try simple tree IDs (T3, H1, T124, etc.)

    match = re.search(r"[_\-]([TH]\d+)[_\-]", col_name, re.IGNORECASE)

    if match:
        return match.group(1)

    # Try to match just numbers with common prefixes

    match = re.search(r"[_\-]tree(\d+)[_\-]", col_name, re.IGNORECASE)

    if match:
        return match.group(1)

    return None


def extract_sensor_id_from_column(col_name: str) -> str | None:
    """

    Extract sensor_id from column name patterns like:

    - NO-Hur_T3_SFM1E21V_sapflow -> 1E21V (sensor is after SFM prefix)

    - NO-Hur_H1_SFM1E30A_sapflow -> 1E30A

    """

    # Pattern: SFM followed by sensor ID

    match = re.search(r"SFM([A-Z0-9]+)", col_name, re.IGNORECASE)

    if match:
        return match.group(1)

    # Try direct sensor ID pattern (alphanumeric like 1E30A)

    match = re.search(r"[_\-](\d+[A-Z]+\d*)[_\-]", col_name, re.IGNORECASE)

    if match:
        return match.group(1)

    return None


def build_id_based_column_mapping(value_cols: list[str], meta_df: pd.DataFrame) -> dict[str, str]:
    """

    Build column mapping based on extracting tree_id/sensor_id from column names

    instead of relying on column order.



    Returns:

        dict mapping original column name -> pl_code

    """

    # Create lookup dictionaries from metadata

    tree_id_to_pl_code = {}

    sensor_id_to_pl_code = {}

    for _, row in meta_df.iterrows():
        if pd.notna(row.get("tree_id")) and pd.notna(row.get("pl_code")):
            tree_id_to_pl_code[str(row["tree_id"]).upper()] = row["pl_code"]

        if pd.notna(row.get("sensor_id")) and pd.notna(row.get("pl_code")):
            sensor_id_to_pl_code[str(row["sensor_id"]).upper()] = row["pl_code"]

    mapping = {}
    unmatched = []

    for col in value_cols:
        # Try to extract tree_id from column name
        tree_id = extract_tree_id_from_column(col)
        sensor_id = extract_sensor_id_from_column(col)

        pl_code = None

        # Strategy 1: Try exact column name matching with metadata tree_id
        # This handles cases like ZOE_AT where column name IS the tree_id (e.g., 'ZOE_5106B00')
        col_upper = col.upper()
        if col_upper in tree_id_to_pl_code:
            pl_code = tree_id_to_pl_code[col_upper]

        # Strategy 2: Try matching by extracted tree_id
        if pl_code is None and tree_id:
            tree_id_upper = tree_id.upper()
            if tree_id_upper in tree_id_to_pl_code:
                pl_code = tree_id_to_pl_code[tree_id_upper]

        # Strategy 3: Try matching by sensor_id
        if pl_code is None and sensor_id:
            sensor_id_upper = sensor_id.upper()
            if sensor_id_upper in sensor_id_to_pl_code:
                pl_code = sensor_id_to_pl_code[sensor_id_upper]

        # Strategy 4: Try substring matching - check if column name contains any tree_id
        if pl_code is None:
            for tid, pcode in tree_id_to_pl_code.items():
                if tid in col_upper or col_upper in tid:
                    pl_code = pcode
                    break

        if pl_code:
            mapping[col] = pl_code
        else:
            unmatched.append(col)

    if unmatched:
        print(
            f"        WARNING: {len(unmatched)} columns could not be matched by ID: {unmatched[:3]}{'...' if len(unmatched) > 3 else ''}"
        )

    return mapping


def create_sfn_directories(output_path, site_code):
    """Create output directories for SAPFLUXNET format."""

    Path(output_path, "plant").mkdir(parents=True, exist_ok=True)

    Path(output_path, "sapwood").mkdir(parents=True, exist_ok=True)

    Path(output_path, "leaf").mkdir(parents=True, exist_ok=True)

    print(f"    Created output directories for {site_code}")


def get_site_files(site_path):
    """Find metadata and data files in site directory."""

    all_files = os.listdir(site_path)

    metadata_file = None

    data_files = []

    for f in all_files:
        f_lower = f.lower()

        file_ext = Path(f).suffix.lower()

        # Check if it's a metadata file

        if "meta" in f_lower and file_ext in [".xlsx", ".xls"]:
            metadata_file = f

        # Check if it's a data file

        elif file_ext in SUPPORTED_EXTENSIONS and not f_lower.startswith("metadata"):
            # Skip dendro and meta files

            if "meta" not in f_lower:
                data_files.append(f)

    if not metadata_file:
        print(f"    WARNING: No metadata Excel file found in {site_path}")

    return metadata_file, data_files


def detect_delimiter(file_path: Path, sample_lines: int = 5) -> str:
    """Detect delimiter in text file by analyzing first few lines."""

    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            lines = [f.readline() for _ in range(sample_lines)]

        # Count occurrences of each delimiter

        delimiter_counts = {delim: sum(line.count(delim) for line in lines) for delim in DELIMITERS}

        # Return delimiter with most consistent occurrence

        best_delim = max(delimiter_counts, key=delimiter_counts.get)

        if delimiter_counts[best_delim] > 0:
            return best_delim

    except Exception as e:
        print(f"    WARNING: Error detecting delimiter: {e}")

    return ","  # Default to comma


def parse_dates_flexible(date_series: pd.Series) -> pd.Series:
    """Try multiple date formats to parse dates."""

    # If already datetime, return as is

    if pd.api.types.is_datetime64_any_dtype(date_series):
        return date_series

    # Convert any datetime objects to strings first

    if date_series.dtype == object:
        date_series = date_series.astype(str)

    # CRITICAL FIX: Use format='mixed' FIRST to handle files with mixed datetime/date-only strings

    # This is essential for files like DE-Har where most rows have '2023-04-28 00:20:00'

    # but midnight rows have just '2023-04-28' (no time component)

    try:
        parsed = pd.to_datetime(date_series, format="mixed", errors="coerce")

        if parsed.notna().sum() > len(date_series) * 0.5:
            return parsed

    except:
        pass

    for date_format in DATE_FORMATS:
        try:
            if date_format == "ISO8601":
                return pd.to_datetime(date_series, format="ISO8601", errors="coerce")

            else:
                parsed = pd.to_datetime(date_series, format=date_format, errors="coerce")

                # Check if at least 50% parsed successfully (lowered threshold)

                if parsed.notna().sum() > len(date_series) * 0.5:
                    return parsed

        except:
            continue

    # Fallback to pandas automatic parsing without infer_datetime_format

    try:
        return pd.to_datetime(date_series, format="mixed", errors="coerce")

    except:
        return pd.Series([pd.NaT] * len(date_series))


def generate_timestamps_from_metadata(meta_row, n_rows):
    """

    Generate timestamp series from metadata when data file has no timestamp column.

    Uses sensor_start, sensor_stop, and data_frequency from metadata.

    Handles both original column names and SAPFLUXNET renamed versions:
    - sensor_start / pl_sensor_start
    - sensor_stop / pl_sensor_end
    - data frequency / data_frequency / env_timestep

    """

    try:
        # Get start and end times - try both original and renamed column names
        start_time = None
        for col in ["sensor_start", "pl_sensor_start"]:
            if col in meta_row.index:
                start_time = pd.to_datetime(meta_row.get(col, None), errors="coerce")
                if pd.notna(start_time):
                    break

        end_time = None
        for col in ["sensor_stop", "pl_sensor_end"]:
            if col in meta_row.index:
                end_time = pd.to_datetime(meta_row.get(col, None), errors="coerce")
                if pd.notna(end_time):
                    break

        # Get frequency (e.g., "30 min", "1 hour", "15min") - try multiple column names
        freq_str = ""
        for col in ["data frequency", "data_frequency", "env_timestep"]:
            if col in meta_row.index:
                freq_str = str(meta_row.get(col, "")).strip()
                if freq_str:
                    break

        if pd.isna(start_time) or not freq_str:
            return None

        # Parse frequency string

        freq_value = re.search(r"(\d+)", freq_str)

        if not freq_value:
            return None

        freq_num = int(freq_value.group(1))

        # Determine frequency unit

        if "min" in freq_str.lower():
            freq = f"{freq_num}min"  # Use 'min' instead of 'T'

        elif "hour" in freq_str.lower() or "hr" in freq_str.lower() or "h" in freq_str.lower():
            freq = f"{freq_num}H"  # hours

        elif "day" in freq_str.lower():
            freq = f"{freq_num}D"  # days

        else:
            freq = f"{freq_num}min"  # default to minutes

        # Generate timestamps

        if pd.notna(end_time):
            # Use both start and end

            timestamps = pd.date_range(start=start_time, end=end_time, freq=freq)

        else:
            # Use start and number of rows

            timestamps = pd.date_range(start=start_time, periods=n_rows, freq=freq)

        return timestamps[:n_rows]  # Ensure we don't exceed n_rows

    except Exception as e:
        print(f"        WARNING: Could not generate timestamps from metadata: {e}")

        return None


def read_data_file(file_path: Path, header="infer") -> pd.DataFrame | None:
    """

    Read data file with flexible format detection.

    Supports Excel, CSV, TSV, TXT, DAT, and RDS files.

    Handles files with no headers or only numeric headers.

    """

    file_path = Path(file_path)

    file_ext = file_path.suffix.lower()

    try:
        # Excel files

        if file_ext in [".xlsx", ".xls"]:
            # Special handling for ES-Abr files (they have NO headers)

            if any(x in str(file_path) for x in ["ES-Abr", "ES_Abr", "ES-LM1", "ES_LM1", "ES-LM2", "ES_LM2"]):
                df = pd.read_excel(file_path, header=None, engine="openpyxl" if file_ext == ".xlsx" else None)

                print("dataframe columns before assigning:", df.columns)

                # Assign column names - first is timestamp, rest are values

                if len(df.columns) == 2:
                    df.columns = ["timestamp", "value"]

                else:
                    df.columns = ["timestamp"] + [f"value_{i}" for i in range(1, len(df.columns))]

                print(f"        INFO: ES-Abr file read without headers, columns assigned: {list(df.columns)}")

            else:
                df = pd.read_excel(file_path, engine="openpyxl" if file_ext == ".xlsx" else None, header=header)

        # RDS files (R data format)

        elif file_ext == ".rds":
            df = pyreadr.read_r(str(file_path))[None]

        # Text-based files (CSV, TXT, TSV, DAT)

        elif file_ext in [".csv", ".txt", ".tsv", ".dat"]:
            # Detect delimiter

            delimiter = detect_delimiter(file_path)

            # Try reading with detected delimiter

            df = pd.read_csv(
                file_path, sep=delimiter, header=header, on_bad_lines="skip", encoding="utf-8", low_memory=False
            )

            print(df.columns)

            # If delimiter detection failed, try with python engine (auto-detect)

            if df.shape[1] == 1 and delimiter != "\t":
                df = pd.read_csv(
                    file_path, sep=None, engine="python", header=header, on_bad_lines="skip", encoding="utf-8"
                )

        else:
            print(f"    WARNING: Unsupported file format: {file_ext}")

            return None

        print(
            f"        INFO: Successfully read file '{file_path.name}' with shape {df.shape}, data columns: {list(df.columns)}"
        )

        # Check if headers are meaningful or just numbers/default names

        has_meaningful_headers = False

        if header == "infer" or header == 0:
            # Check if column names look like real headers

            col_names_str = [str(col).lower() for col in df.columns]

            # Count how many columns have meaningful names (not just numbers or "unnamed" or 'nan')

            meaningful_count = sum(
                1
                for col in col_names_str
                if not col.replace(".", "").replace("-", "").isdigit()
                and "unnamed: 0" not in col
                and "nan" not in col
                and "na" not in col
                and "NA" not in col
            )

            # CRITICAL FIX: Before removing "meaningless" columns, check if they contain actual data

            # This handles cases where data is in a column named "Unnamed: 0" or similar

            cols_to_check = [col for col in df.columns if str(col).lower() in ["unnamed: 0", "nan", "na", "NA"]]

            cols_to_keep = []

            for col in cols_to_check:
                # Try to convert to numeric and see if we have valid data

                numeric_values = pd.to_numeric(df[col], errors="coerce")

                n_valid = numeric_values.notna().sum()

                if n_valid > len(df) * 0.1:  # If more than 10% are valid numeric values
                    print(
                        f"        INFO: Column '{col}' appears meaningless but contains {n_valid} numeric values - KEEPING IT!"
                    )

                    cols_to_keep.append(col)

            # Only drop columns that are truly meaningless (no data)

            cols_to_drop = [col for col in cols_to_check if col not in cols_to_keep]

            if cols_to_drop:
                print(f"        INFO: Dropping meaningless columns: {cols_to_drop}")

                df = df.drop(columns=cols_to_drop)

            has_meaningful_headers = meaningful_count > 1

        else:
            has_meaningful_headers = True

        # drop meaningless columns if any

        print(f"        INFO: Has meaningful headers: {has_meaningful_headers}")

        # Handle files with no meaningful headers

        if not has_meaningful_headers:
            print(f"        INFO: Detected no meaningful headers in '{file_path.name}'")

            # SPECIAL CASE: If we have 'Unnamed: 0' with numeric data and 'A' with text

            # This happens with SE-Nor files where data is in 'Unnamed: 0' and 'A' is just letters

            if "Unnamed: 0" in df.columns and "A" in df.columns:
                # Check if 'A' column is all text

                a_col_numeric = pd.to_numeric(df["A"], errors="coerce")

                if a_col_numeric.notna().sum() == 0:  # All non-numeric
                    print("        INFO: Column 'A' contains no numeric data - dropping it")

                    df = df.drop(columns=["A"])

                    # Rename 'Unnamed: 0' to 'value'

                    df.columns = ["value"]

                    print("        INFO: Renamed 'Unnamed: 0' to 'value' (contains the actual data)")

            else:
                print("        INFO: No special case detected")

                n_cols = len(df.columns)

                if n_cols == 1:
                    # Single column: must be value, timestamp will be generated from metadata

                    df.columns = ["value"]

                    print("        INFO: Single column detected, assigned as 'value' (timestamp to be generated)")

                elif n_cols == 2:
                    # Two columns: first is timestamp, second is value

                    df.columns = ["timestamp", "value"]

                    print("        INFO: Two columns detected, assigned as ['timestamp', 'value']")

                else:
                    # Multiple columns: first is timestamp, rest are values or additional columns

                    df.columns = ["timestamp"] + [f"value_{i}" for i in range(1, n_cols)]

                    print(f"        INFO: {n_cols} columns detected, assigned: {list(df.columns)}")

        else:
            # Clean column names (handle both string and numeric columns)

            if hasattr(df.columns, "str"):
                df.columns = df.columns.str.strip()

            else:
                # Convert numeric columns to strings

                df.columns = [str(col).strip() for col in df.columns]

        # Parse date columns if any

        if not df.empty and len(df.columns) > 0 and "timestamp" in df.columns:
            # We already have a timestamp column assigned

            print("        DEBUG READ: Found time column: 'timestamp'")

            print(f"        DEBUG READ: Raw values before parsing: {df['timestamp'].head(3).tolist()}")

            print(f"        DEBUG READ: Dtype before parsing: {df['timestamp'].dtype}")

            df["timestamp"] = parse_dates_flexible(df["timestamp"])

            print(f"        DEBUG READ: After parsing: {df['timestamp'].head(3).tolist()}")

            print(f"        DEBUG READ: Valid dates: {df['timestamp'].notna().sum()}/{len(df)}")

        elif not df.empty and len(df.columns) > 0:
            time_col = find_col_by_names(df.columns.tolist(), TIMESTAMP_NAMES)

            if time_col is not None:
                if isinstance(time_col, list):
                    # skip parsing if multiple found

                    print(f"        WARNING: Multiple timestamp columns found: {time_col}, skipping date parsing")

                    return df

                print(f"        DEBUG READ: Found time column: '{time_col}'")

                print(f"        DEBUG READ: Raw values before parsing: {df[time_col].head(3).tolist()}")

                print(f"        DEBUG READ: Dtype before parsing: {df[time_col].dtype}")

                df[time_col] = parse_dates_flexible(df[time_col])

                print(f"        DEBUG READ: After parsing: {df[time_col].head(3).tolist()}")

                print(f"        DEBUG READ: Valid dates: {df[time_col].notna().sum()}/{len(df)}")

        return df

    except Exception as e:
        print(f"    ERROR reading data file {file_path.name}: {e}")

        return None


def tidy_metadata(
    site_path: Path, metadata_file: str, site_code: str, data_files: list[str] = []
) -> pd.DataFrame | None:
    """

    Parse and tidy metadata from Excel file.

    Handles both two-sheet and single-sheet formats.

    """

    file_path = Path(site_path, metadata_file)

    meta_df_tidy = None

    try:
        # Try reading all sheets

        all_sheets = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")

        # Look for site and sapflow sheets

        site_sheet_name = next((s for s in all_sheets if "site" in s.lower()), None)

        sapflow_sheet_name = next((s for s in all_sheets if "sapflow" in s.lower() or "tree" in s.lower()), None)

        if site_sheet_name and sapflow_sheet_name:
            # Two-sheet format

            site_df_wide = all_sheets[site_sheet_name].set_index(all_sheets[site_sheet_name].columns[0])

            site_df_wide = site_df_wide.loc[~site_df_wide.index.duplicated(keep="first")]

            plant_info = all_sheets[sapflow_sheet_name].set_index(all_sheets[sapflow_sheet_name].columns[0])

            plant_info = plant_info.loc[~plant_info.index.duplicated(keep="first")].transpose().reset_index(drop=True)

            # Merge site info into plant info

            for col in site_df_wide.transpose().columns:
                plant_info[col] = site_df_wide.transpose()[col].iloc[0]

            meta_df_tidy = plant_info

        elif site_sheet_name or sapflow_sheet_name:
            # One of the sheets is missing, try to use the available one

            sheet_name = site_sheet_name if site_sheet_name else sapflow_sheet_name

            print(f"    INFO: Only one metadata sheet found: {sheet_name}, attempting to read it...")

            meta_df_wide = all_sheets[sheet_name]

            # meta_df_wide = meta_df_wide.loc[~meta_df_wide.index.duplicated(keep='first')]

            meta_df_tidy = meta_df_wide

    except Exception as e:
        print(f"    INFO: Two-sheet format not detected: {e}")

    # Fallback to single-sheet format

    if meta_df_tidy is None:
        print("    INFO: Trying single-sheet metadata format...")

        try:
            meta_df_wide = pd.read_excel(file_path, sheet_name=0, header=None, index_col=0, engine="openpyxl")

            meta_df_tidy = (
                meta_df_wide.loc[~meta_df_wide.index.duplicated(keep="first")].transpose().reset_index(drop=True)
            )

            print(meta_df_tidy.columns)

        except Exception as e:
            print(f"    ERROR: Failed to read metadata: {e}")

            return None

    # Clean column names (remove units in parentheses)

    def clean_col(c):

        return str(c).split("(")[0].strip() if isinstance(c, str) else c

    meta_df_tidy.rename(columns=clean_col, inplace=True)

    # Find site_id column - prefer exact matches first, then partial matches

    # Priority: 'site_id' > columns starting with 'site_id' > columns containing 'site_id'

    site_id = None

    for col in meta_df_tidy.columns:
        col_lower = str(col).lower().strip()

        if col_lower == "site_id":
            site_id = col

            break

        elif col_lower.startswith("site_id") and site_id is None:
            site_id = col

    # Fallback to any column containing 'site' followed by 'id'

    if site_id is None:
        site_id = next(
            (col for col in meta_df_tidy.columns if "site" in str(col).lower() and "id" in str(col).lower()), None
        )

    # Last resort - any column just containing 'site' (but not 'site_' prefixed columns like 'site_stems')

    if site_id is None:
        site_id = next(
            (
                col
                for col in meta_df_tidy.columns
                if str(col).lower().strip() == "site" or str(col).lower().startswith("site_id")
            ),
            None,
        )

    # Find tree_id column - use TREE_ID_NAMES list for matching
    tree_id = None
    # First try exact match with TREE_ID_NAMES
    for col in meta_df_tidy.columns:
        col_clean = str(col).lower().strip().replace(" ", "_").replace("/", "_")
        # Also try without special characters
        col_normalized = re.sub(r"[^a-z0-9]", "", str(col).lower())
        for name in TREE_ID_NAMES:
            name_normalized = re.sub(r"[^a-z0-9]", "", name.lower())
            if col_clean == name or col_normalized == name_normalized:
                tree_id = col
                break
        if tree_id:
            break
    # Fallback: check for 'tree' and 'id' or 'number' in column name
    if tree_id is None:
        tree_id = next(
            (
                col
                for col in meta_df_tidy.columns
                if (
                    "tree" in str(col).lower()
                    and ("id" in str(col).lower() or "number" in str(col).lower() or "numb" in str(col).lower())
                )
                or "plant" in str(col).lower()
            ),
            None,
        )

    print(meta_df_tidy.columns)

    print(f"    INFO: Detected site_id column: {site_id}, tree_id column: {tree_id}")

    # Validate required columns

    if site_id not in meta_df_tidy.columns or tree_id not in meta_df_tidy.columns:
        print("    ERROR: Required columns 'site_id' or 'tree_id' not found in metadata")

        return None

    # CRITICAL FIX: Remove rows where tree_id is NaN (these create bogus plant entries like "ES-Abr-nan-nan")
    before_count = len(meta_df_tidy)
    meta_df_tidy = meta_df_tidy[meta_df_tidy[tree_id].notna()]
    after_count = len(meta_df_tidy)
    if before_count != after_count:
        print(f"    INFO: Removed {before_count - after_count} rows with NaN tree_id (bogus plant entries)")

    # Create plant codes - handle NaN sensor_id by using empty string or omitting
    if "sensor_id" in meta_df_tidy.columns:

        def make_pl_code(r):
            sensor = r.get("sensor_id")
            if pd.notna(sensor):
                return f"{r[site_id]}-{r[tree_id]}-{sensor}"
            else:
                return f"{r[site_id]}-{r[tree_id]}"

        meta_df_tidy["pl_code"] = meta_df_tidy.apply(make_pl_code, axis=1)
    else:
        meta_df_tidy["pl_code"] = meta_df_tidy.apply(lambda r: f"{r[site_id]}-{r[tree_id]}", axis=1)

    RENAME_MAP = {
        site_id: "site_id",
        tree_id: "tree_id",
        # Plant metadata mappings
        "tree_sapwood_area_cm2": "pl_sapw_area",
        "tree_sapwood_area": "pl_sapw_area",
        "sapwood_area": "pl_sapw_area",
        "tree_sapwood_thickness_cm": "pl_sapw_depth",
        "sapwood_depth": "pl_sapw_depth",
        "tree_totalbark_thickness_mm": "pl_bark_thick",
        "bark_thickness": "pl_bark_thick",
        "tree_phloem_thickness_mm": "pl_phloem_thick",
        "tree_dbh": "pl_dbh",
        "dbh": "pl_dbh",
        "tree_height": "pl_height",
        "height": "pl_height",
        "tree_age": "pl_age",
        "age": "pl_age",
        "tree_genus": "pl_genus",
        "genus": "pl_genus",
        "tree_species": "pl_species",
        "species": "pl_species",
        "tree_latitude": "pl_lat",
        "tree latitude": "pl_lat",
        "latitude": "pl_lat",
        "tree_longitude": "pl_long",
        "tree longitude": "pl_long",
        "longitude": "pl_long",
        "tree_altitude": "pl_elev",
        "tree altitude": "pl_elev",
        "elevation": "pl_elev",
        "tree_status": "pl_social",
        "status": "pl_social",
        "sensor_type": "pl_sens_meth",
        "sensor_height": "pl_sens_hgt",
        "sensor_id": "pl_sensor_id",
        "series_id": "pl_series_id",
        "sensor_start": "pl_sensor_start",
        "sensor_stop": "pl_sensor_end",
        "sensor_cutout": "pl_sensor_cutout",
        "sensor_exposition": "pl_sensor_exposition",
        "units": "pl_sap_units_orig",
        "variable_name": "pl_variable_name",
        "notes": "pl_remarks",
        "sensor_timezone": "env_time_zone",
        "data frequency": "env_timestep",
        "data_frequency": "env_timestep",
    }

    meta_df_tidy.rename(columns=RENAME_MAP, inplace=True)

    return meta_df_tidy


def normalize_name(name: str) -> str:
    """Normalize file name by removing double underscores and getting stem."""

    return re.sub(r"__+", "_", Path(name).stem)


def find_col_by_names(df_columns: list[str], potential_names: list[str]) -> str | None:
    """

    Find matching column names using flexible keyword matching.

    Returns single column name, list of column names, or None.

    """

    # Convert all columns to strings first to handle numeric column names

    df_cols_lower = {str(col).lower().strip(): str(col) for col in df_columns}

    matches = {}  # Dictionary to store column: score

    # Identify which type of column we're looking for

    is_timestamp = potential_names == TIMESTAMP_NAMES

    is_tree_id = potential_names == TREE_ID_NAMES

    is_sensor_id = potential_names == SENSOR_ID_NAMES

    is_value = potential_names == VALUE_NAMES

    is_env = potential_names == ENV_NAMES

    # For each column in the dataframe

    for col_lower, col_orig in df_cols_lower.items():
        score = 0

        # Skip unnamed columns

        if "unnamed" in col_lower:
            continue

        # Apply exclusion rules based on column type

        if is_env:
            # CRITICAL: Skip timestamp columns - be very strict here

            if any(kw in col_lower for kw in ["timestamp", "date", "datetime", "datum", "fecha", "data", "time"]):
                continue

            # Skip tree_id and sensor_id columns

            if any(
                kw in col_lower
                for kw in [
                    "treeid",
                    "tree_id",
                    "treenumber",
                    "tree_number",
                    "tree number",
                    "sensorid",
                    "sensor_id",
                    "sensor",
                    "probe",
                    "orientation",
                ]
            ):
                continue

            if any(
                kw in col_lower
                for kw in ["sapflow", "sap flow", "sap_flow", "sap flux", "sap_flux", "sap", "velocity", "sfd", "js"]
            ):
                continue

        if is_value:
            # CRITICAL: Exclude specific column types for sap flow data

            # Skip TDSV (sap velocity), TD (temperature difference), K (correction factors)

            if any(
                pattern in col_lower
                for pattern in [
                    "tdsv",
                    "td1",
                    "td2",
                    "td3",
                    "td4",
                    "td5",
                    "td6",
                    "td7",
                    "td8",
                    "td9",
                    "td10",
                    "k1",
                    "k2",
                    "k3",
                    "k4",
                    "k5",
                    "k6",
                    "k7",
                    "k8",
                    "k9",
                    "k10",
                ]
            ):
                # Only exclude if it's exactly these patterns (not part of a longer name)

                col_parts = col_lower.replace("_", " ").replace("-", " ").split()

                if any(
                    part
                    in [
                        "tdsv1",
                        "tdsv2",
                        "tdsv3",
                        "tdsv4",
                        "tdsv5",
                        "tdsv6",
                        "tdsv7",
                        "tdsv8",
                        "tdsv9",
                        "tdsv10",
                        "td1",
                        "td2",
                        "td3",
                        "td4",
                        "td5",
                        "td6",
                        "td7",
                        "td8",
                        "td9",
                        "td10",
                        "k1",
                        "k2",
                        "k3",
                        "k4",
                        "k5",
                        "k6",
                        "k7",
                        "k8",
                        "k9",
                        "k10",
                    ]
                    for part in col_parts
                ):
                    continue

            # Skip columns with "velocity" in the name unless they also clearly indicate mass flow

            if "velocity" in col_lower and "g h-1" not in col_lower and "kg" not in col_lower:
                continue

            # Skip timestamp columns

            if any(
                kw in col_lower
                for kw in ["timestamp", "date", "datetime", "datum", "fecha", "doy", "toy", "hour", "month"]
            ):
                continue

            # Skip tree_id and sensor_id columns unless they contain value-related terms

            if any(
                kw in col_lower
                for kw in [
                    "treeid",
                    "tree_id",
                    "treenumber",
                    "tree_number",
                    "sensorid",
                    "sensor_id",
                    "sensor number",
                    "probe",
                ]
            ):
                if not any(
                    vk in col_lower
                    for vk in ["sapflow", "kg/h", "g h-1", "flux", "sfd", "flow", "js_", "js_outer", "js_inner"]
                ):
                    continue

            # BOOST score for columns that match pattern "Sap" + number (e.g., Sap1, Sap2)

            if col_lower.startswith("sap") and any(char.isdigit() for char in col_lower[:5]):
                score += 150  # High priority for Sap# columns

        if is_tree_id or is_sensor_id:
            # Skip value columns

            if any(
                term in col_lower
                for term in ["sapflow", "kg/h", "flux", "velocity", "sfd", "flow", "js_", "js_outer", "js_inner"]
            ):
                continue

        # Check each potential keyword

        for keyword in potential_names:
            keyword_lower = keyword.lower()

            keyword_clean = keyword_lower.replace("_", "").replace("-", "").replace(" ", "")

            col_clean = col_lower.replace("_", "").replace("-", "").replace(" ", "").replace(".", "")

            # Exact match (highest score)

            if col_lower == keyword_lower:
                score += 100

            # Clean match (no delimiters)

            elif col_clean == keyword_clean:
                score += 90

            # Keyword appears as whole word in column

            elif (
                keyword_lower in col_lower.split("_")
                or keyword_lower in col_lower.split("-")
                or keyword_lower in col_lower.split(".")
            ):
                score += 70

            # Keyword contained in column

            elif keyword_lower in col_lower:
                score += 50

            # Partial match (keyword parts in column) - only for keywords > 3 chars

            elif len(keyword_lower) > 3:
                for kw_part in keyword_lower.split("_"):
                    if len(kw_part) > 2 and kw_part in col_lower:
                        score += 20

                        break

        # Store match if score > 0

        if score > 0:
            matches[col_orig] = score

        print(f"        DEBUG: Column '{col_orig}' scored {score} for potential names {potential_names}")

    # If no matches found

    if not matches:
        # Special case for value columns: return all remaining numeric-looking columns

        if is_value:
            potential_value_cols = []

            for col_lower, col_orig in df_cols_lower.items():
                if "unnamed" in col_lower:
                    continue

                # Exclude timestamp, tree_id, sensor_id, and metadata columns

                if any(
                    kw in col_lower
                    for kw in [
                        "timestamp",
                        "date",
                        "datetime",
                        "treeid",
                        "tree_id",
                        "sensorid",
                        "sensor_id",
                        "sensor",
                        "probe",
                        "doy",
                        "toy",
                        "hour",
                        "month",
                        "tdsv",
                        "td1",
                        "td2",
                        "td3",
                        "td4",
                        "td5",
                        "td6",
                        "td7",
                        "td8",
                        "td9",
                        "td10",
                        "k1",
                        "k2",
                        "k3",
                        "k4",
                        "k5",
                        "k6",
                        "k7",
                        "k8",
                        "k9",
                        "k10",
                    ]
                ):
                    continue

                potential_value_cols.append(col_orig)

            if potential_value_cols:
                if is_value:
                    print(
                        f"        DEBUG: Found value columns (fallback): {potential_value_cols[:3]}{'...' if len(potential_value_cols) > 3 else ''}"
                    )

                if len(potential_value_cols) == 1:
                    return potential_value_cols[0]

                return potential_value_cols

        return None

    # Sort by score (highest first)

    sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)

    # Debug output for value columns

    if is_value and sorted_matches:
        col_names = [col for col, score in sorted_matches]

        print(f"        DEBUG: Found value columns: {col_names}{'...' if len(sorted_matches) > 3 else ''}")

    # Return results

    if len(sorted_matches) == 1:
        return sorted_matches[0][0]

    # If multiple columns with similar high scores, return them all

    # for env columns, return all with score >= 50% of best score

    if is_env:
        best_score = sorted_matches[0][1]

        high_score_matches = [col for col, score in sorted_matches if score >= best_score * 0.3]

        return high_score_matches

    else:
        best_score = sorted_matches[0][1]

    # Get all columns with score >= 80% of best score

    high_score_matches = [col for col, score in sorted_matches if score >= best_score * 0.8]

    if len(high_score_matches) == 1:
        return high_score_matches[0]

    return high_score_matches


def process_pattern_A(site_path, site_code, metadata_file, data_files, output_path):
    """

    Processes sites using a flexible 'Pattern A' that handles both single files per plant

    and multiple yearly files per plant.

    """

    print(f"--> Processing '{site_code}' using Pattern A (intelligent file inspection)...")

    meta_df_tidy = tidy_metadata(site_path, metadata_file, site_code)

    if meta_df_tidy is None:
        print("    ERROR: Metadata could not be tidied or is missing key columns.")

        return

    meta_df_tidy["file_stem_norm"] = meta_df_tidy["file_name"].astype(str).apply(normalize_name)

    # CRITICAL: Also remove year patterns from metadata file names to match data file names

    year_pattern = re.compile(r"(_(19|20)\d{2}[._-](19|20)\d{2}|_(19|20)\d{2})")

    meta_df_tidy["file_stem_norm"] = meta_df_tidy["file_stem_norm"].apply(
        lambda x: normalize_name(year_pattern.sub("", x))
    )

    print(f"    DEBUG: After year removal, meta_df_tidy['file_stem_norm']:\n{meta_df_tidy['file_stem_norm']}")

    # Create lookup dictionaries for both tree_id and sensor_id

    tree_id_to_pl_code = pd.Series(meta_df_tidy.pl_code.values, index=meta_df_tidy.tree_id.astype(str)).to_dict()

    # Also create sensor_id mapping if sensor_id exists in metadata

    sensor_id_to_pl_code = {}

    if "sensor_id" in meta_df_tidy.columns:
        sensor_id_to_pl_code = pd.Series(
            meta_df_tidy.pl_code.values, index=meta_df_tidy.sensor_id.astype(str)
        ).to_dict()

    # --- Group ALL files by base name ---

    file_groups = {}

    year_pattern = re.compile(r"(_(19|20)\d{2}[._-](19|20)\d{2}|_(19|20)\d{2})")

    for file in data_files:
        # Create a base name by removing year patterns

        base_name = normalize_name(year_pattern.sub("", file))

        file_groups.setdefault(base_name, []).append(file)

    all_sapflow_dfs = []

    all_env_dfs = []

    print(f"    INFO: Processing {len(file_groups)} file groups from {len(data_files)} total files")

    # Loop through each group of files

    for base_name, file_list in file_groups.items():
        matched_meta = meta_df_tidy[meta_df_tidy["file_stem_norm"] == base_name]

        if matched_meta.empty:
            print(f"    WARNING: No metadata entry found for file group with base name: '{base_name}'. Skipping.")

            continue

        # --- Concatenate all files within the group ---

        plant_data_frames = []

        for f in file_list:
            df = read_data_file(Path(site_path, f), header=0)

            print(
                f"        DEBUG: File '{f}' has columns: {list(df.columns)[:5]}{'...' if len(df.columns) > 5 else ''}"
            )

            if df is not None and not df.empty:
                print(
                    f"        DEBUG: File '{f}' has columns: {list(df.columns)[:5]}{'...' if len(df.columns) > 5 else ''}"
                )

                plant_data_frames.append(df)

        if not plant_data_frames:
            continue

        full_plant_df = pd.concat(plant_data_frames, ignore_index=True)

        pl_code = matched_meta["pl_code"].iloc[0]

        print(f"    INFO: For plant '{pl_code}', combined {len(file_list)} file(s) from group '{base_name}'.")

        # --- Check if we need to generate timestamps ---

        needs_timestamp_generation = "timestamp" not in full_plant_df.columns

        ts_col = None
        print(
            f"        DEBUG: needs_timestamp_generation={needs_timestamp_generation}, columns={full_plant_df.columns.tolist()}"
        )

        if not needs_timestamp_generation:
            # Check if timestamp column exists but was not parsed

            ts_col = find_col_by_names(full_plant_df.columns.tolist(), TIMESTAMP_NAMES)

            if ts_col is None and "timestamp" in full_plant_df.columns:
                ts_col = "timestamp"

        else:
            # No timestamp column - check if we have only value column
            # Strip column names first to handle any whitespace issues
            full_plant_df.columns = [str(c).strip() for c in full_plant_df.columns]

            if "value" in full_plant_df.columns and len(full_plant_df.columns) == 1:
                # Generate timestamps from metadata
                # Check for both original and renamed column names
                has_start = any(c in matched_meta.columns for c in ["sensor_start", "pl_sensor_start"])
                has_freq = any(c in matched_meta.columns for c in ["data frequency", "data_frequency", "env_timestep"])
                print(f"        DEBUG: Has sensor_start (orig or renamed): {has_start}")
                print(f"        DEBUG: Has data frequency (orig or renamed): {has_freq}")

                timestamps = generate_timestamps_from_metadata(matched_meta.iloc[0], len(full_plant_df))

                if timestamps is not None and len(timestamps) == len(full_plant_df):
                    full_plant_df.insert(0, "timestamp", timestamps)

                    ts_col = "timestamp"

                    print(f"        INFO: Generated {len(timestamps)} timestamps from metadata")

                else:
                    print(f"        WARNING: Could not generate timestamps for file group '{base_name}'. Skipping.")

                    continue

        # --- Intelligent column detection on the combined DataFrame ---

        if ts_col is None:
            ts_col = find_col_by_names(full_plant_df.columns.tolist(), TIMESTAMP_NAMES)

        tree_col = find_col_by_names(full_plant_df.columns.tolist(), TREE_ID_NAMES)

        sensor_col = find_col_by_names(full_plant_df.columns.tolist(), SENSOR_ID_NAMES)

        value_col = find_col_by_names(full_plant_df.columns.tolist(), VALUE_NAMES)

        env_cols = find_col_by_names(full_plant_df.columns.tolist(), ENV_NAMES)

        print(
            f"        DEBUG DETECT: ts_col={ts_col}, tree_col={tree_col}, sensor_col={sensor_col}, value_col={value_col}"
        )

        # Check if sensor_col has uninformative values (e.g., single unique value like "Sensor")

        # If so, don't use it as an ID column - the tree name should come from filename

        if sensor_col:
            sensor_col_name = sensor_col[0] if isinstance(sensor_col, list) else sensor_col

            unique_sensor_values = full_plant_df[sensor_col_name].dropna().unique()

            if len(unique_sensor_values) <= 1:
                uninformative_val = unique_sensor_values[0] if len(unique_sensor_values) == 1 else "empty"

                print(
                    f"    INFO: Ignoring sensor_col '{sensor_col_name}' - uninformative (all values = '{uninformative_val}')"
                )

                sensor_col = None  # Don't use this as ID column

        # Handle multiple ID columns - combine tree and sensor if both exist

        id_col = None

        use_combined_id = False

        id_to_pl_code = None

        tree_col_for_lookup = None  # Store for later use in pivoting

        sensor_col_for_lookup = None  # Store for later use in pivoting

        if tree_col and sensor_col:
            # Handle lists and store for later lookup

            if isinstance(tree_col, list):
                tree_col_for_lookup = tree_col[0]

            else:
                tree_col_for_lookup = tree_col

            if isinstance(sensor_col, list):
                sensor_col_for_lookup = sensor_col[0]

            else:
                sensor_col_for_lookup = sensor_col

            # Check if sensor_col already embeds tree_id (e.g., sensor_id="SF022_20_ENE_1"
            # contains tree number 22).  Use sensor_col alone so each physical sensor
            # installation becomes its own column instead of averaging sub-sensors.
            sensor_vals = full_plant_df[sensor_col_for_lookup].dropna().unique()
            tree_vals = full_plant_df[tree_col_for_lookup].dropna().unique()
            sensor_embeds_tree = False
            if len(sensor_vals) > len(tree_vals):
                # sensor_col is more granular — check if tree_id is a substring of sensor values
                sample = full_plant_df[[tree_col_for_lookup, sensor_col_for_lookup]].dropna().head(20)
                matches = sum(
                    str(row[tree_col_for_lookup]) in str(row[sensor_col_for_lookup]) for _, row in sample.iterrows()
                )
                sensor_embeds_tree = matches >= len(sample) * 0.8

            if sensor_embeds_tree:
                # sensor_id already contains tree info — use it directly
                id_col = sensor_col_for_lookup
                id_to_pl_code = sensor_id_to_pl_code
                use_combined_id = False
                print(f"    INFO: sensor_col '{sensor_col_for_lookup}' embeds tree_id — using as sole ID column")
            else:
                # Combine tree + sensor for unique identifiers
                full_plant_df["_combined_id"] = (
                    full_plant_df[tree_col_for_lookup].astype(str)
                    + "-"
                    + full_plant_df[sensor_col_for_lookup].astype(str)
                )

                id_col = "_combined_id"

                use_combined_id = True

                print(f"    INFO: Created combined ID from '{tree_col_for_lookup}' and '{sensor_col_for_lookup}'")

        elif tree_col:
            # Only tree_col exists

            id_col = tree_col

            id_to_pl_code = tree_id_to_pl_code

            print(f"    INFO: Using tree_col as ID: {tree_col}")

        elif sensor_col:
            # Only sensor_col exists

            id_col = sensor_col

            id_to_pl_code = sensor_id_to_pl_code

            print(f"    INFO: Using sensor_col as ID: {sensor_col}")

        print(f"        DEBUG DETECT: id_col={id_col}")

        if not ts_col:
            print(f"    WARNING: No timestamp column found in group '{base_name}'. Skipping.")

            continue

        print(f"        DEBUG: Passed ts_col check, ts_col type is {type(ts_col)}")

        # CRITICAL FIX: Handle timestamp column that might be a list BEFORE any operations

        if isinstance(ts_col, list):
            print(f"        DEBUG: ts_col is a list: {ts_col}")

            if len(ts_col) > 1:
                print(f"    WARNING: Multiple timestamp columns found: {ts_col}. Selecting best match...")

                # Try to find the best timestamp column by checking data frequency

                frequency_col_name = next((c for c in matched_meta.columns if "frequency" in str(c).lower()), None)

                freq_str = matched_meta[frequency_col_name].iloc[0] if frequency_col_name else None

                best_ts_col = None

                expected_freq = None

                if freq_str:
                    # Parse frequency string

                    freq_value = re.search(r"(\d+)", str(freq_str))

                    if freq_value:
                        freq_num = int(freq_value.group(1))

                        # Determine frequency unit

                        freq_str_lower = str(freq_str).lower()

                        if "min" in freq_str_lower:
                            expected_freq = pd.Timedelta(minutes=freq_num)

                        elif "hour" in freq_str_lower or "hr" in freq_str.lower() or "h" in freq_str.lower():
                            expected_freq = pd.Timedelta(hours=freq_num)

                        elif "day" in freq_str.lower():
                            expected_freq = pd.Timedelta(days=freq_num)

                        else:
                            expected_freq = pd.Timedelta(minutes=freq_num)

                        # Check each timestamp column individually

                        for col in ts_col:
                            checked_timestamps = pd.to_datetime(full_plant_df[col], format="mixed", errors="coerce")

                            valid_timestamps = checked_timestamps.dropna()

                            if len(valid_timestamps) > 1:
                                diffs = valid_timestamps.sort_values().diff().dropna()

                                mode_diff = diffs.mode()[0] if not diffs.mode().empty else None

                                if mode_diff and abs(mode_diff - expected_freq) < pd.Timedelta(minutes=1):
                                    print(
                                        f"    INFO: Selected timestamp column '{col}' matching data frequency '{freq_str}'"
                                    )

                                    best_ts_col = col

                                    break

                                else:
                                    print(
                                        f"        DEBUG: Column '{col}' mode diff {mode_diff} does not match expected {expected_freq}"
                                    )

                # If no single column worked, try combining Date + Time

                if not best_ts_col and "Date" in ts_col and "Time" in ts_col:
                    print("    INFO: Combining Date and Time columns...")

                    print(f"        DEBUG: ts_col = {ts_col}")

                    print("        DEBUG: Checking if 'Date' and 'Time' columns exist in dataframe...")

                    print(f"        DEBUG: DataFrame columns: {list(full_plant_df.columns)}")

                    print(f"        DEBUG: 'Date' in columns: {'Date' in full_plant_df.columns}")

                    print(f"        DEBUG: 'Time' in columns: {'Time' in full_plant_df.columns}")

                    try:
                        date_col = "Date"

                        time_col = "Time"

                        print(f"        DEBUG: Date column first 3 values: {full_plant_df[date_col].head(3).tolist()}")

                        print(f"        DEBUG: Time column first 3 values: {full_plant_df[time_col].head(3).tolist()}")

                        print(f"        DEBUG: Time column dtype: {full_plant_df[time_col].dtype}")

                        print(f"        DEBUG: Type of first time value: {type(full_plant_df[time_col].iloc[0])}")

                        # Convert dates

                        dates = pd.to_datetime(full_plant_df[date_col], errors="coerce")

                        print(f"        DEBUG: Converted dates first 3: {dates.head(3).tolist()}")

                        print(f"        DEBUG: Valid dates count: {dates.notna().sum()}/{len(dates)}")

                        # Handle time column - check if it's time objects

                        time_values = full_plant_df[time_col]

                        has_hour_attr = hasattr(time_values.iloc[0], "hour")

                        print(f"        DEBUG: First time value has 'hour' attribute: {has_hour_attr}")

                        if has_hour_attr:  # datetime.time objects
                            print("        DEBUG: Converting datetime.time objects to timedelta...")

                            # Convert time objects to timedelta and add to date

                            time_deltas = time_values.apply(
                                lambda t: (
                                    pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
                                    if pd.notna(t) and hasattr(t, "hour")
                                    else None
                                )
                            )

                            print(f"        DEBUG: time_deltas first 3: {time_deltas.head(3).tolist()}")

                            # Ensure it's timedelta dtype

                            time_deltas = pd.to_timedelta(time_deltas)

                            print(f"        DEBUG: After pd.to_timedelta, first 3: {time_deltas.head(3).tolist()}")

                            full_plant_df["combined_timestamp"] = dates + time_deltas

                            print(
                                f"        DEBUG: Combined timestamp first 3: {full_plant_df['combined_timestamp'].head(3).tolist()}"
                            )

                        else:
                            print("        DEBUG: Parsing time as strings...")

                            # Parse as time strings

                            times = pd.to_datetime(time_values.astype(str), format="%H:%M:%S", errors="coerce")

                            full_plant_df["combined_timestamp"] = pd.to_datetime(
                                dates.dt.date.astype(str) + " " + times.dt.time.astype(str), errors="coerce"
                            )

                            print(
                                f"        DEBUG: Combined timestamp first 3: {full_plant_df['combined_timestamp'].head(3).tolist()}"
                            )

                        # Check if combination worked

                        valid_timestamps = full_plant_df["combined_timestamp"].dropna()

                        print(f"        DEBUG: Valid combined timestamps: {len(valid_timestamps)}/{len(full_plant_df)}")

                        print(f"        DEBUG: First 3 valid timestamps: {valid_timestamps.head(3).tolist()}")

                        if len(valid_timestamps) > 10:
                            print(
                                f"    INFO: âœ“ Successfully combined Date+Time! ({len(valid_timestamps)} valid timestamps)"
                            )

                            best_ts_col = "combined_timestamp"

                            # Also check frequency if available

                            if expected_freq:
                                diffs = valid_timestamps.sort_values().diff().dropna()

                                mode_diff = diffs.mode()[0] if not diffs.mode().empty else None

                                if mode_diff:
                                    print(
                                        f"        DEBUG: Combined timestamp frequency: {mode_diff} (expected: {expected_freq})"
                                    )

                        else:
                            print(
                                f"        WARNING: Only {len(valid_timestamps)} valid timestamps - need more than 10!"
                            )

                    except Exception as e:
                        import traceback

                        print(f"        ERROR: Exception caught: {e}")

                        print(f"        ERROR: Full traceback:\n{traceback.format_exc()}")

                # If we found a good timestamp column, use it

                if best_ts_col:
                    ts_col = best_ts_col

                    print(f"    INFO: Using timestamp column: '{ts_col}'")

                else:
                    # Last resort: generate timestamps from metadata

                    print("    INFO: No valid timestamp column found, generating from metadata...")

                    timestamps = generate_timestamps_from_metadata(matched_meta.iloc[0], len(full_plant_df))

                    if timestamps is not None and len(timestamps) == len(full_plant_df):
                        full_plant_df.insert(0, "timestamp", timestamps)

                        ts_col = "timestamp"

                        print(f"        INFO: Generated {len(timestamps)} timestamps from metadata")

                    else:
                        print(f"        WARNING: Could not generate timestamps for file group '{base_name}'. Skipping.")

                        continue

            else:
                # Only one column in the list

                ts_col = ts_col[0]

                print(f"    INFO: Single timestamp column from list: '{ts_col}'")

        print(f"        DEBUG: After list handling, ts_col='{ts_col}' (type: {type(ts_col)})")

        # Verify ts_col is now a string

        if not isinstance(ts_col, str):
            print(f"    ERROR: ts_col is not a string after processing: {ts_col} (type: {type(ts_col)})")

            continue

        # --- Process environmental data if present ---

        if env_cols and ts_col:
            if not isinstance(env_cols, list):
                env_cols = [env_cols]

            print(
                f"    INFO: Found {len(env_cols)} environmental column(s): {env_cols[:3]}{'...' if len(env_cols) > 3 else ''}"
            )

            env_df = full_plant_df[[ts_col] + env_cols].copy()

            env_df = env_df.rename(columns={ts_col: "timestamp"})

            env_df["timestamp"] = pd.to_datetime(env_df["timestamp"], format="mixed", errors="coerce")

            env_df.dropna(subset=["timestamp"], inplace=True)

            # Convert all env columns to numeric

            for col in env_cols:
                env_df[col] = pd.to_numeric(env_df[col], errors="coerce")

            if not env_df.empty:
                env_df = env_df.set_index("timestamp")

                env_df = env_df.loc[~env_df.index.duplicated(keep="first")]

                all_env_dfs.append(env_df)

        processed_df = None

        print(f"        DEBUG BRANCH: Checking id_col... id_col={id_col}, value_col={value_col}")

        # This is a simple file group (no internal tree_id or sensor_id column)

        if not id_col:
            print("        DEBUG BRANCH: Entered 'no id_col' branch (simple file group)")

            renamed_vc = []

            if not value_col:
                print(f"    WARNING: No value column found in group '{base_name}'. Skipping.")

                continue

            print(f"        DEBUG BRANCH: value_col check passed, value_col={value_col}, type={type(value_col)}")

            try:
                if isinstance(value_col, list):
                    print("        DEBUG BRANCH: value_col is a list, entering list processing branch")

                    print(f"    INFO: Multiple value columns found: {value_col}")

                    rename_map = {ts_col: "timestamp"}

                    for i, vc in enumerate(value_col):
                        new_name = f"{pl_code}_{i}" if len(value_col) > 1 else pl_code

                        rename_map[vc] = new_name

                        renamed_vc.append(new_name)

                    print(
                        f"        DEBUG: Before rename - columns: {list(full_plant_df.columns)}, shape: {full_plant_df.shape}"
                    )

                    print(f"        DEBUG: Rename map: {rename_map}")

                    print(f"        DEBUG: First 3 rows of data:\n{full_plant_df.head(3)}")

                    df = full_plant_df.rename(columns=rename_map)

                    print(f"        DEBUG: After rename - columns: {list(df.columns)}, shape: {df.shape}")

                    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")

                    print(
                        f"        DEBUG: After timestamp parse - valid timestamps: {df['timestamp'].notna().sum()}/{len(df)}"
                    )

                    df.dropna(subset=["timestamp"], inplace=True)

                    print(f"        DEBUG: After dropping NA timestamps - shape: {df.shape}")

                    # Convert value columns to numeric

                    for col in renamed_vc:
                        print(
                            f"        DEBUG: Converting column '{col}' to numeric. Sample values before: {df[col].head(3).tolist()}"
                        )

                        df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")

                        print(f"        DEBUG: After conversion - valid values: {df[col].notna().sum()}/{len(df)}")

                    df.dropna(subset=renamed_vc, how="all", inplace=True)

                    print(f"        DEBUG: After dropping all-NA rows - shape: {df.shape}")

                    if not df.empty:
                        processed_df = df.set_index("timestamp")[renamed_vc]

                        print(f"        DEBUG: Final processed_df shape: {processed_df.shape}")

                    else:
                        processed_df = None

                        print("        DEBUG: DataFrame is empty after processing!")

                else:  # Single value column
                    print("        DEBUG BRANCH: Entered ELSE block for single value column")

                    print(f"        DEBUG: ts_col={ts_col}, value_col={value_col}, pl_code={pl_code}")

                    print(f"        DEBUG: full_plant_df.columns={list(full_plant_df.columns)}")

                    print(f"        DEBUG: full_plant_df.shape={full_plant_df.shape}")

                    rename_map = {ts_col: "timestamp", value_col: pl_code}

                    renamed_vc.append(pl_code)

                    print(f"        DEBUG: Attempting to select columns [{ts_col}, {value_col}] from dataframe")

                    df = full_plant_df[[ts_col, value_col]].rename(columns=rename_map)

                    print(
                        f"        DEBUG: After selecting and renaming - df.shape={df.shape}, df.columns={list(df.columns)}"
                    )

                    print(f"        DEBUG: First 3 rows:\n{df.head(3)}")

                    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")

                    print(
                        f"        DEBUG: After timestamp parse - valid timestamps: {df['timestamp'].notna().sum()}/{len(df)}"
                    )

                    df.dropna(subset=["timestamp"], inplace=True)

                    print(f"        DEBUG: After dropping NA timestamps - shape: {df.shape}")

                    print(
                        f"        DEBUG: Converting column '{pl_code}' to numeric. Sample values before: {df[pl_code].head(3).tolist()}"
                    )

                    df[pl_code] = pd.to_numeric(df[pl_code], errors="coerce")

                    print(
                        f"        DEBUG: After conversion - valid values: {df[pl_code].notna().sum()}/{len(df)}, NA values: {df[pl_code].isna().sum()}"
                    )

                    df.dropna(subset=[pl_code], inplace=True)

                    print(f"        DEBUG: After dropping NA values - shape: {df.shape}")

                    if not df.empty:
                        processed_df = df.set_index("timestamp")[[pl_code]]

                        print(f"        DEBUG: Final processed_df shape: {processed_df.shape}")

                    else:
                        processed_df = None

                        print("        DEBUG: DataFrame is empty after processing!")

            except Exception as e:
                print(f"        DEBUG ERROR: Exception caught during processing: {type(e).__name__}: {e}")

                import traceback

                traceback.print_exc()

                processed_df = None

        else:  # This file group contains an internal tree_id or sensor_id column (long format)
            id_col_display = (
                id_col if not use_combined_id else f"combined({tree_col_for_lookup}+{sensor_col_for_lookup})"
            )

            print(f"    INFO: ID column ('{id_col_display}') found in group '{base_name}'. Processing as long format.")

            if isinstance(id_col, list):
                id_col = id_col[0]

            if not value_col:
                print(f"    WARNING: No value column could be identified in group '{base_name}'. Skipping.")

                continue

            # Ensure value_col is a list for uniform processing

            if not isinstance(value_col, list):
                value_col = [value_col]

            else:
                print(f"    INFO: Processing {len(value_col)} value columns: {value_col}")

            # Process each value column separately

            pivoted_dfs = []

            for val_col in value_col:
                long_df = full_plant_df[[ts_col, id_col, val_col]].copy()

                long_df.columns = ["timestamp", "id", "value"]

                long_df["timestamp"] = pd.to_datetime(long_df["timestamp"], format="mixed", errors="coerce")

                long_df["id"] = long_df["id"].astype(str)

                long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")

                long_df.dropna(subset=["timestamp", "id", "value"], inplace=True)

                if not long_df.empty:
                    pivoted_df = long_df.pivot_table(index="timestamp", columns="id", values="value", aggfunc="mean")

                    if use_combined_id:
                        # For combined IDs, the column names are already complete (e.g., "282-N")

                        # We need to map these to plant codes from metadata

                        new_cols = []

                        for col in pivoted_df.columns:
                            # Try to find matching plant code in metadata

                            # Split the combined ID

                            parts = str(col).split("-")

                            if len(parts) >= 2:
                                tree_part = parts[0]

                                sensor_part = "-".join(parts[1:])  # In case sensor has dashes

                                # Look for matching row in metadata using METADATA's column names

                                # The metadata uses 'tree_id' and 'sensor_id' as standardized names

                                try:
                                    # Try with 'tree_id' and 'sensor_id' (metadata standard names)

                                    if "tree_id" in meta_df_tidy.columns and "sensor_id" in meta_df_tidy.columns:
                                        matched = meta_df_tidy[
                                            (meta_df_tidy["tree_id"].astype(str) == tree_part)
                                            & (meta_df_tidy.get("sensor_id", pd.Series()).astype(str) == sensor_part)
                                        ]

                                    # Fallback: try with actual data column names (less likely to work)

                                    elif (
                                        tree_col_for_lookup in meta_df_tidy.columns
                                        and sensor_col_for_lookup in meta_df_tidy.columns
                                    ):
                                        matched = meta_df_tidy[
                                            (meta_df_tidy[tree_col_for_lookup].astype(str) == tree_part)
                                            & (meta_df_tidy[sensor_col_for_lookup].astype(str) == sensor_part)
                                        ]

                                    else:
                                        matched = pd.DataFrame()  # Empty dataframe

                                    if not matched.empty:
                                        new_cols.append(matched["pl_code"].iloc[0])

                                    else:
                                        # Fallback: use the combined ID as-is with site code

                                        new_cols.append(f"{site_code}-{col}")

                                        print(f"        WARNING: No metadata match for {col}, using default name")

                                except KeyError as e:
                                    print(f"        WARNING: Column lookup failed for {col}: {e}, using default name")

                                    new_cols.append(f"{site_code}-{col}")

                            else:
                                new_cols.append(f"{site_code}-{col}")

                        pivoted_df.columns = new_cols

                    else:
                        # Single ID column - use existing mapping

                        # Create column names with sensor/measurement type suffix

                        sensor_suffix = val_col.replace("Js_", "").replace("_", "-")

                        if len(value_col) > 1:
                            # Add suffix only if multiple value columns exist

                            pivoted_df.columns = [
                                f"{id_to_pl_code.get(str(tid), str(tid))}-{sensor_suffix}" for tid in pivoted_df.columns
                            ]

                        else:
                            pivoted_df.columns = [id_to_pl_code.get(str(tid), str(tid)) for tid in pivoted_df.columns]

                    pivoted_dfs.append(pivoted_df)

            # Combine all value columns

            if pivoted_dfs:
                processed_df = pd.concat(pivoted_dfs, axis=1)

        if processed_df is not None and not processed_df.empty:
            print(
                f"        DEBUG FINAL: processed_df is valid, shape={processed_df.shape}, appending to all_sapflow_dfs"
            )

            processed_df = processed_df.loc[~processed_df.index.duplicated(keep="first")]

            all_sapflow_dfs.append(processed_df)

        else:
            print(f"        DEBUG FINAL: processed_df is None or empty! processed_df={processed_df}")

    print(f"    DEBUG END: Finished processing all file groups. all_sapflow_dfs has {len(all_sapflow_dfs)} dataframes")

    if not all_sapflow_dfs:
        print(f"    WARNING: No valid sapflow data was processed for site {site_code}.")

        return

    site_sapflow_df = pd.concat(all_sapflow_dfs, axis=1)

    site_sapflow_df = site_sapflow_df.groupby(site_sapflow_df.index).mean().sort_index()

    # Process environmental data if any was collected

    site_env_df = None

    if all_env_dfs:
        site_env_df = pd.concat(all_env_dfs, axis=1)

        site_env_df = site_env_df.groupby(site_env_df.index).mean().sort_index()

        print(f"    INFO: Collected environmental data with {len(site_env_df.columns)} variable(s)")

    write_sfn_files(meta_df_tidy, site_sapflow_df, site_code, output_path, site_env_df)


def process_pattern_B(site_path, site_code, metadata_file, data_files, output_path):
    """

    Processes sites where ONE file contains data for ALL trees as separate columns.

    Uses ID-based column mapping by extracting tree_id/sensor_id from column names.

    Falls back to order-based mapping if ID extraction fails.

    Handles multiple yearly files by concatenating them.



    Pattern B characteristics:

    - One file contains all trees (each tree is a column)

    - No tree_id or sensor_id columns within files

    - Column names often contain embedded tree_id/sensor_id (e.g., NO-Hur_T3_SFM1E21V_sapflow)

    - Uses ID-based mapping for reliability; order-based as fallback

    """

    print(f"--> Processing '{site_code}' using Pattern B (all trees in one file, no file name matching)...")

    # Read and tidy metadata

    meta_df_tidy = tidy_metadata(site_path, metadata_file, site_code)

    if meta_df_tidy is None:
        print("    ERROR: Metadata could not be tidied or is missing key columns.")

        return

    # Sort metadata by tree_id to match expected column order

    meta_df_tidy = meta_df_tidy.sort_values("tree_id").reset_index(drop=True)

    print(f"    INFO: Metadata contains {len(meta_df_tidy)} tree(s): {meta_df_tidy['pl_code'].tolist()}")

    # Group data files by year pattern (in case there are yearly splits)

    year_pattern = re.compile(r"(_(19|20)\d{2}[._-](19|20)\d{2}|_(19|20)\d{2})")

    file_groups = {}

    for file in data_files:
        base_name = normalize_name(year_pattern.sub("", file))

        file_groups.setdefault(base_name, []).append(file)

    print(f"    INFO: Found {len(file_groups)} file group(s) from {len(data_files)} total file(s)")

    all_sapflow_dfs = []

    all_env_dfs = []

    # Process each file group (usually just one group unless split by year)

    for base_name, file_list in file_groups.items():
        print(f"\n    INFO: Processing file group '{base_name}' ({len(file_list)} file(s))")

        # Read and concatenate all files in this group

        yearly_dfs = []

        for f in sorted(file_list):  # Sort to ensure chronological order
            df = read_data_file(Path(site_path, f), header=0)

            if df is not None and not df.empty:
                print(f"        INFO: Read file '{f}' - shape: {df.shape}, columns: {list(df.columns)}")

                # Try to extract year from filename

                year_match = re.search(r"(19|20)\d{2}", f)

                if year_match:
                    df["_file_year"] = int(year_match.group(0))

                    print(f"        INFO: Extracted year {df['_file_year'].iloc[0]} from filename")

                yearly_dfs.append(df)

            else:
                print(f"        WARNING: Failed to read or empty file: '{f}'")

        if not yearly_dfs:
            print(f"    WARNING: No valid data files in group '{base_name}'. Skipping.")

            continue

        # Concatenate all yearly files

        combined_df = pd.concat(yearly_dfs, ignore_index=True)

        print(f"    INFO: Combined data shape: {combined_df.shape}")

        # Detect columns

        ts_col = find_col_by_names(combined_df.columns.tolist(), TIMESTAMP_NAMES)

        value_cols = find_col_by_names(combined_df.columns.tolist(), VALUE_NAMES)

        env_cols = find_col_by_names(combined_df.columns.tolist(), ENV_NAMES)

        # SPECIAL: IT-CP2_sapwood uses TDSV (velocity) columns which are normally excluded

        if site_code == "IT-CP2_sapwood":
            tdsv_cols = [col for col in combined_df.columns if "tdsv" in col.lower() or "velocity" in col.lower()]

            if tdsv_cols:
                value_cols = tdsv_cols

                print(f"        INFO: IT-CP2_sapwood special handling - using TDSV velocity columns: {tdsv_cols}")

                # Also exclude these from env_cols if they were accidentally included

                if env_cols and isinstance(env_cols, list):
                    env_cols = [c for c in env_cols if c not in tdsv_cols]

        print(f"        DEBUG: ts_col={ts_col}, value_cols={value_cols}, env_cols={env_cols}")

        # Handle timestamp column

        if ts_col is None:
            # Try to generate from metadata

            print("        INFO: No timestamp column found. Attempting to generate from metadata...")

            timestamps = generate_timestamps_from_metadata(meta_df_tidy.iloc[0], len(combined_df))

            if timestamps is not None and len(timestamps) == len(combined_df):
                combined_df.insert(0, "timestamp", timestamps)

                ts_col = "timestamp"

                print(f"        INFO: Generated {len(timestamps)} timestamps from metadata")

            else:
                print(f"    WARNING: Could not find or generate timestamp for '{base_name}'. Skipping.")

                continue

        # Handle list of timestamp columns (take first one)

        if isinstance(ts_col, list):
            if len(ts_col) > 1:
                print(f"        WARNING: Multiple timestamp columns: {ts_col}. Using '{ts_col[0]}'")

            ts_col = ts_col[0]

        # Special handling for DOY (Day of Year) format

        if ts_col.upper() == "DOY" or "doy" in ts_col.lower():
            print("        INFO: Detected DOY (Day of Year) format. Constructing timestamps...")

            # Check if we have year information

            if "_file_year" not in combined_df.columns:
                print("    ERROR: DOY format detected but no year information available. Skipping.")

                continue

            # Check for additional time components

            hour_col = None

            for col in ["Hour", "hour", "TOY", "toy", "time"]:
                if col in combined_df.columns:
                    hour_col = col

                    break

            # Construct timestamps from year + DOY + hour

            try:
                # Create base date (January 1st of the year)

                base_date = pd.to_datetime(combined_df["_file_year"].astype(str) + "-01-01")

                # Convert DOY to timedelta (DOY=1 is Jan 1, so subtract 1)

                doy_offset = pd.to_timedelta(combined_df[ts_col].astype(float) - 1, unit="D")

                # Add DOY offset to base date

                combined_df["timestamp"] = base_date + doy_offset

                # Add hour component if available

                if hour_col:
                    hour_offset = pd.to_timedelta(combined_df[hour_col].astype(float), unit="h")

                    combined_df["timestamp"] = combined_df["timestamp"] + hour_offset

                    print(f"        INFO: Constructed timestamps from Year + DOY + {hour_col}")

                else:
                    print("        INFO: Constructed timestamps from Year + DOY")

                # Verify timestamps

                valid_ts = combined_df["timestamp"].notna().sum()

                print(f"        INFO: Valid timestamps: {valid_ts}/{len(combined_df)}")

            except Exception as e:
                print(f"    ERROR: Failed to construct timestamps from DOY: {e}")

                import traceback

                traceback.print_exc()

                continue

        else:
            # Parse timestamp normally

            combined_df[ts_col] = pd.to_datetime(combined_df[ts_col], format="mixed", errors="coerce")

            combined_df.rename(columns={ts_col: "timestamp"}, inplace=True)

        # Drop rows with invalid timestamps

        combined_df.dropna(subset=["timestamp"], inplace=True)

        if combined_df.empty:
            print(f"    WARNING: No valid timestamps in '{base_name}'. Skipping.")

            continue

        print(f"        INFO: Valid timestamps: {len(combined_df)}")

        # Process environmental data if present

        if env_cols:
            if not isinstance(env_cols, list):
                env_cols = [env_cols]

            print(f"        INFO: Found {len(env_cols)} environmental column(s)")

            env_df = combined_df[["timestamp"] + env_cols].copy()

            env_df = env_df.set_index("timestamp")

            # Convert to numeric

            for col in env_cols:
                env_df[col] = pd.to_numeric(env_df[col], errors="coerce")

            env_df = env_df.loc[~env_df.index.duplicated(keep="first")]

            if not env_df.empty:
                all_env_dfs.append(env_df)

        # Get value columns

        if value_cols is None:
            # If no value columns detected, use all remaining numeric columns

            exclude_cols = ["timestamp", "_file_year", ts_col] + (
                env_cols if env_cols and isinstance(env_cols, list) else []
            )

            if "hour_col" in locals() and hour_col:
                exclude_cols.append(hour_col)

            # Also exclude DOY, TOY, Hour, month if they exist

            exclude_cols.extend(["DOY", "TOY", "Hour", "month", "doy", "toy", "hour"])

            value_cols = [col for col in combined_df.columns if col not in exclude_cols]

            if not value_cols:
                print(f"    WARNING: No value columns found in '{base_name}'. Skipping.")

                continue

            print(
                f"        INFO: Using all remaining columns as values: {value_cols[:5]}{'...' if len(value_cols) > 5 else ''}"
            )

        # Ensure value_cols is a list

        if not isinstance(value_cols, list):
            value_cols = [value_cols]

        print(f"        INFO: Found {len(value_cols)} value column(s) in data")

        print(f"        INFO: Metadata has {len(meta_df_tidy)} plant(s)")

        # Create sapflow dataframe

        sapflow_df = combined_df[["timestamp"] + value_cols].copy()

        sapflow_df = sapflow_df.set_index("timestamp")

        # Convert to numeric

        for col in value_cols:
            sapflow_df[col] = pd.to_numeric(sapflow_df[col], errors="coerce")

        # Map columns to plant codes based on extracted IDs (tree_id/sensor_id from column names)

        # This is more reliable than assuming column order matches metadata order

        print("        INFO: Attempting ID-based column mapping...")

        column_mapping = build_id_based_column_mapping(value_cols, meta_df_tidy)

        if column_mapping:
            # ID-based mapping succeeded

            print(f"        INFO: Successfully mapped {len(column_mapping)} column(s) using ID extraction")

            print(f"        INFO: Sample mapping: {dict(list(column_mapping.items())[:3])}")

            sapflow_df.rename(columns=column_mapping, inplace=True)

        else:
            # Fallback to order-based mapping (legacy behavior)

            print("        WARNING: ID-based mapping failed, falling back to order-based mapping")

            plant_codes = meta_df_tidy["pl_code"].tolist()

            if len(value_cols) != len(plant_codes):
                print(
                    f"    WARNING: Number of value columns ({len(value_cols)}) doesn't match number of plants in metadata ({len(plant_codes)})"
                )

                print("    INFO: Will map available columns to plant codes in order")

                # Use minimum of both lengths

                n_cols = min(len(value_cols), len(plant_codes))

                sapflow_df = sapflow_df.iloc[:, :n_cols]

                plant_codes = plant_codes[:n_cols]

            # Rename columns to plant codes

            column_mapping = dict(zip(value_cols[: len(plant_codes)], plant_codes))

            sapflow_df.rename(columns=column_mapping, inplace=True)

            print(f"        INFO: Mapped {len(column_mapping)} column(s) to plant codes (order-based)")

            print(f"        INFO: Sample mapping: {dict(list(column_mapping.items())[:3])}")

        # Remove duplicates and sort

        sapflow_df = sapflow_df.loc[~sapflow_df.index.duplicated(keep="first")]

        sapflow_df = sapflow_df.sort_index()

        # Drop all-NA rows

        sapflow_df = sapflow_df.dropna(how="all")

        if not sapflow_df.empty:
            all_sapflow_dfs.append(sapflow_df)

            print(
                f"        INFO: âœ“ Processed sapflow data - shape: {sapflow_df.shape}, columns: {list(sapflow_df.columns)[:5]}{'...' if len(sapflow_df.columns) > 5 else ''}"
            )

        else:
            print("        WARNING: Sapflow dataframe is empty after processing")

    # Combine all sapflow data

    if not all_sapflow_dfs:
        print(f"    ERROR: No valid sapflow data was processed for site {site_code}.")

        return

    print(f"\n    INFO: Combining {len(all_sapflow_dfs)} sapflow dataframe(s)...")

    site_sapflow_df = pd.concat(all_sapflow_dfs, axis=0)  # Concatenate along rows (time) for yearly files

    site_sapflow_df = site_sapflow_df.groupby(site_sapflow_df.index).mean().sort_index()

    print(
        f"    INFO: Final sapflow dataframe - shape: {site_sapflow_df.shape}, columns: {list(site_sapflow_df.columns)[:5]}{'...' if len(site_sapflow_df.columns) > 5 else ''}"
    )

    # Combine environmental data

    site_env_df = None

    if all_env_dfs:
        site_env_df = pd.concat(all_env_dfs, axis=0)  # Concatenate along rows (time)

        site_env_df = site_env_df.groupby(site_env_df.index).mean().sort_index()

        print(f"    INFO: Final environmental dataframe - shape: {site_env_df.shape}")

    # Write output files

    write_sfn_files(meta_df_tidy, site_sapflow_df, site_code, output_path, site_env_df)

    print(f"    âœ“ Successfully processed site {site_code} using Pattern B")


def process_pattern_C(site_path, site_code, metadata_file, data_files, output_path):
    """

    Processes sites using Pattern C - handles both sapflow and dendrometer data.



    Pattern C characteristics:

    - Site folders contain both sapflow (SFD) and dendrometer (Dendro/DT) data

    - Multiple frequencies (daily, half-hourly indicated by HH)

    - Long format with tree numbers in columns

    - Both .csv and .rds files

    - Separate output for sapflow and dendro data



    Parameters:

    -----------

    site_path : Path

        Path to site folder

    site_code : str

        Site code (e.g., 'CH-Dav')

    metadata_file : str

        Metadata filename

    data_files : List[str]

        List of data files in site folder

    output_path : Path

        Output directory path

    """

    print(f"--> Processing '{site_code}' using Pattern C (sapflow + dendrometer, multiple frequencies)...")

    # Read and tidy metadata

    meta_df_tidy = tidy_metadata(site_path, metadata_file, site_code)

    if meta_df_tidy is None:
        print("    ERROR: Metadata could not be tidied or is missing key columns.")

        return

    print(f"    INFO: Metadata contains {len(meta_df_tidy)} plant(s)")

    # Create tree_id to plant code mapping

    tree_id_to_pl_code = pd.Series(meta_df_tidy.pl_code.values, index=meta_df_tidy.tree_id.astype(str)).to_dict()

    # Group files by data type and frequency

    file_groups = {"sapflow_daily": [], "sapflow_halfhourly": [], "dendro_daily": [], "dendro_halfhourly": []}

    for file in data_files:
        print(f"    DEBUG: Analyzing file: {file}")

        file_lower = file.lower()

        # Determine data type

        if "sfd" in file_lower or "sapflux" in file_lower or "sapflow" in file_lower or "_dt_" in file_lower:
            data_type = "sapflow"

        elif "dendro" in file_lower or "dendrometer" in file_lower:
            data_type = "dendro"

        else:
            print(f"    WARNING: Could not determine data type for file: {file}")

            continue

        # Determine frequency

        if "_hh" in file_lower or "halfhour" in file_lower or "half_hour" in file_lower:
            frequency = "halfhourly"

        elif "daily" in file_lower:
            frequency = "daily"

        else:
            print(f"    WARNING: Could not determine frequency for file: {file}")

            continue

        # Add to appropriate group

        group_key = f"{data_type}_{frequency}"

        file_groups[group_key].append(file)

        print(f"    INFO: Classified '{file}' as {group_key}")

    # Storage for processed dataframes

    all_sapflow_dfs = {"daily": [], "halfhourly": []}

    all_dendro_dfs = {"daily": [], "halfhourly": []}

    all_env_dfs = []

    # Process each file group

    for group_key, file_list in file_groups.items():
        if not file_list:
            continue

        data_type, frequency = group_key.split("_")

        print(f"\n    INFO: Processing {group_key} - {len(file_list)} file(s)")

        for file in file_list:
            file_path = Path(site_path, file)

            print(f"        Processing: {file}")

            # Read data file

            df = read_data_file(file_path, header=0)

            if df is None or df.empty:
                print(f"        WARNING: Could not read or empty file: {file}")

                continue

            print(f"        INFO: Loaded {len(df)} rows, {len(df.columns)} columns")

            print(f"        DEBUG: Columns: {list(df.columns)}")

            # Detect columns

            ts_col = find_col_by_names(df.columns.tolist(), TIMESTAMP_NAMES)

            tree_col = find_col_by_names(df.columns.tolist(), TREE_ID_NAMES)

            env_cols = find_col_by_names(df.columns.tolist(), ENV_NAMES)

            # Determine value column based on data type

            if data_type == "sapflow":
                value_col_names = ["sfd", "sfdm", "sapflux", "sapflow", "value"]

            else:  # dendro
                value_col_names = ["twdm", "twd", "dendrometer", "dendro", "value"]

            value_col = find_col_by_names(df.columns.tolist(), value_col_names)

            print(f"        DEBUG: ts_col={ts_col}, tree_col={tree_col}, value_col={value_col}")

            # Validate required columns

            if ts_col is None:
                print("        WARNING: No timestamp column found. Skipping file.")

                continue

            # Handle list of timestamp columns

            if isinstance(ts_col, list):
                if len(ts_col) > 1:
                    print(f"        WARNING: Multiple timestamp columns: {ts_col}. Using first.")

                ts_col = ts_col[0]

            # Parse timestamps

            df[ts_col] = pd.to_datetime(df[ts_col], format="mixed", errors="coerce")

            df.dropna(subset=[ts_col], inplace=True)

            if df.empty:
                print("        WARNING: No valid timestamps. Skipping file.")

                continue

            print(f"        INFO: Valid timestamps: {len(df)}")

            # Process environmental data if present

            if env_cols and data_type == "sapflow":  # Only extract env from sapflow files
                if not isinstance(env_cols, list):
                    env_cols = [env_cols]

                print(f"        INFO: Found {len(env_cols)} environmental column(s)")

                env_df = df[[ts_col] + env_cols].copy()

                env_df = env_df.rename(columns={ts_col: "timestamp"})

                for col in env_cols:
                    env_df[col] = pd.to_numeric(env_df[col], errors="coerce")

                env_df = env_df.set_index("timestamp")

                env_df = env_df.loc[~env_df.index.duplicated(keep="first")]

                if not env_df.empty:
                    all_env_dfs.append(env_df)

            # Determine ID column (tree_id and/or sensor_id)

            id_col = None

            use_combined_id = False

            tree_col_for_lookup = None

            sensor_col_for_lookup = None

            if tree_col:
                id_col = tree_col if not isinstance(tree_col, list) else tree_col[0]

                print(f"        INFO: Using tree_col as ID: {id_col}")

            if not id_col:
                print("        WARNING: No ID column found. Skipping file.")

                continue

            if not value_col:
                print("        WARNING: No value column found. Skipping file.")

                continue

            # Handle multiple value columns

            if isinstance(value_col, list):
                if len(value_col) > 1:
                    print(f"        INFO: Multiple value columns: {value_col}. Using first.")

                value_col = value_col[0]

            # Create long format dataframe

            long_df = df[[ts_col, id_col, value_col]].copy()

            long_df.columns = ["timestamp", "id", "value"]

            # Convert to numeric

            long_df["id"] = long_df["id"].astype(str)

            long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")

            # Drop NaN values

            long_df.dropna(subset=["timestamp", "id", "value"], inplace=True)

            if long_df.empty:
                print("        WARNING: No valid data after cleaning. Skipping file.")

                continue

            print(f"        INFO: Valid data rows: {len(long_df)}")

            # Pivot to wide format

            try:
                pivoted_df = long_df.pivot_table(index="timestamp", columns="id", values="value", aggfunc="mean")

                print(f"        INFO: Pivoted to {len(pivoted_df.columns)} columns")

                # Map column names to plant codes

                if use_combined_id:
                    print("        INFO: Using combined ID mapping for columns")

                    # Combined ID: split and map

                    new_cols = []

                    for col in pivoted_df.columns:
                        parts = str(col).split("-")

                        if len(parts) >= 2:
                            tree_part = parts[0]

                            sensor_part = "-".join(parts[1:])

                            # Find matching metadata row

                            matched = meta_df_tidy[
                                (meta_df_tidy["tree_id"].astype(str) == tree_part)
                                & (meta_df_tidy.get("sensor_id", pd.Series()).astype(str) == sensor_part)
                            ]

                            if not matched.empty:
                                new_cols.append(matched["pl_code"].iloc[0])

                            else:
                                new_cols.append(f"{site_code}-{col}")

                                print(f"        WARNING: No metadata match for {col}")

                        else:
                            new_cols.append(f"{site_code}-{col}")

                    pivoted_df.columns = new_cols

                else:
                    print("        INFO: Using simple ID mapping for columns")

                    # Simple ID column - map directly

                    if tree_col:
                        pivoted_df.columns = [
                            tree_id_to_pl_code.get(str(tid), f"{site_code}-{tid}") for tid in pivoted_df.columns
                        ]

                # Remove duplicate indices and sort

                pivoted_df = pivoted_df.loc[~pivoted_df.index.duplicated(keep="first")]

                pivoted_df = pivoted_df.sort_index()

                # Store in appropriate list

                if data_type == "sapflow":
                    all_sapflow_dfs[frequency].append(pivoted_df)

                    print(f"        âœ“ Processed sapflow data: {pivoted_df.shape}")

                else:  # dendro
                    all_dendro_dfs[frequency].append(pivoted_df)

                    print(f"        âœ“ Processed dendro data: {pivoted_df.shape}")

            except Exception as e:
                print(f"        ERROR: Failed to pivot data: {e}")

                import traceback

                traceback.print_exc()

                continue

    # Combine and write sapflow data

    for frequency in ["daily", "halfhourly"]:
        if all_sapflow_dfs[frequency]:
            print(f"\n    INFO: Combining {len(all_sapflow_dfs[frequency])} sapflow {frequency} dataframe(s)")

            site_sapflow_df = pd.concat(all_sapflow_dfs[frequency], axis=1)

            # Handle duplicate columns by taking mean (use transpose to avoid deprecated axis=1)

            site_sapflow_df = site_sapflow_df.T.groupby(level=0).mean().T

            site_sapflow_df = site_sapflow_df.groupby(site_sapflow_df.index).mean().sort_index()

            print(f"    INFO: Final sapflow {frequency} data: {site_sapflow_df.shape}")

            # Combine environmental data (only for sapflow)

            site_env_df = None

            if frequency == "halfhourly" and all_env_dfs:  # Prioritize higher frequency env data
                site_env_df = pd.concat(all_env_dfs, axis=1)

                # Handle duplicate columns by taking mean (use transpose to avoid deprecated axis=1)

                site_env_df = site_env_df.T.groupby(level=0).mean().T

                site_env_df = site_env_df.groupby(site_env_df.index).mean().sort_index()

                print(f"    INFO: Environmental data: {site_env_df.shape}")

            # Write sapflow output

            write_sfn_files(meta_df_tidy, site_sapflow_df, f"{site_code}_{frequency}", output_path, site_env_df)

    # Write dendrometer data separately

    for frequency in ["daily", "halfhourly"]:
        if all_dendro_dfs[frequency]:
            print(f"\n    INFO: Combining {len(all_dendro_dfs[frequency])} dendro {frequency} dataframe(s)")

            site_dendro_df = pd.concat(all_dendro_dfs[frequency], axis=1)

            # Handle duplicate columns by taking mean (use transpose to avoid deprecated axis=1)

            site_dendro_df = site_dendro_df.T.groupby(level=0).mean().T

            site_dendro_df = site_dendro_df.groupby(site_dendro_df.index).mean().sort_index()

            print(f"    INFO: Final dendro {frequency} data: {site_dendro_df.shape}")

            # Write dendro output to plant folder

            data_folder_path = Path(output_path, "plant")

            data_folder_path.mkdir(exist_ok=True)

            dendro_filename = f"{site_code}_dendro_{frequency}.csv"

            site_dendro_df.to_csv(data_folder_path / dendro_filename)

            print(f"    âœ“ Wrote dendrometer data to {dendro_filename}")

    print(f"\n    âœ“ Successfully processed site {site_code} using Pattern C")


def write_sfn_files(
    meta_df: pd.DataFrame,
    sapflow_df: pd.DataFrame,
    site_code: str,
    output_path: Path,
    env_df: pd.DataFrame | None = None,
):
    """

    Write metadata and data to SAPFLUXNET CSV format.

    Dynamically determines output folder based on units.

    """

    # Comprehensive rename map

    RENAME_MAP = {
        # Site Metadata
        "site_id": "si_code",
        "site": "si_code",
        "country": "si_country",
        "site_latitude": "si_lat",
        "site_longitude": "si_long",
        "site_altitude": "si_elev",
        "site_affliation": "si_contact_institution",
        "site_affiliation": "si_contact_institution",
        "contacting_person": "si_contact_name",
        "contact_person": "si_contact_name",
        "corresponding_flux_tower": "si_flux_tower",
        "flux_tower": "si_flux_tower",
        "notes": "si_remarks",
        "other_notes": "si_other_remarks",
        # Stand Metadata
        "site_forest_type": "st_forest_type",
        "forest_type": "st_forest_type",
        "site_soil_type": "st_soil_type",
        "soil_type": "st_soil_type",
        "site_stems_per_hectare": "st_density",
        "stems_per_hectare": "st_density",
        "site_basal_area": "st_basal_area",
        "basal_area": "st_basal_area",
        "site_dominant_tree_height": "st_height",
        "dominant_tree_height": "st_height",
        "site_leaf_area_index": "st_lai",
        "leaf_area_index": "st_lai",
        "lai": "st_lai",
        # Plant Metadata
        "tree_id": "pl_name",
        "tree": "pl_name",
        "TreeNumber/TreeNumber": "pl_name",
        "tree_species": "pl_species",
        "species": "pl_species",
        "species": "pl_species",
        "tree_dbh": "pl_dbh",
        "dbh": "pl_dbh",
        "DBH (cm)": "pl_dbh",
        "tree_height": "pl_height",
        "height": "pl_height",
        "Height (m)": "pl_height",
        "tree_age": "pl_age",
        "age": "pl_age",
        "tree_status": "pl_social",
        "status": "pl_social",
        "tree_totalbark_thickness_mm": "pl_bark_thick",
        "bark_thickness": "pl_bark_thick",
        "tree_sapwood_thickness_cm": "pl_sapw_depth",
        "sapwood_depth": "pl_sapw_depth",
        "tree_sapwood_area": "pl_sapw_area",
        "sapwood_area": "pl_sapw_area",
        "tree_sapwood_area_cm2": "pl_sapw_area",
        "sensor_type": "pl_sens_meth",
        "sensor": "sensor_type",
        "method": "pl_sens_meth",
        "sensor_height": "pl_sens_hgt",
        "units": "pl_sap_units_orig",
        "tree_latitude": "pl_lat",
        "latitude": "pl_lat",
        "tree_longitude": "pl_long",
        "longitude": "pl_long",
        "sensor_id": "pl_sensor_id",
        "series_id": "pl_series_id",
        "sensor_start": "pl_sensor_start",
        "sensor_stop": "pl_sensor_end",
        "sensor_cutout": "pl_sensor_cutout",
        "sensor_exposition": "pl_sensor_exposition",
        "tree_altitude": "pl_elev",
        "elevation": "pl_elev",
        "tree_genus": "pl_genus",
        "genus": "pl_genus",
        "tree_phloem_thickness_mm": "pl_phloem_thick",
        # Environmental Metadata
        "sensor_timezone": "env_time_zone",
        "timezone": "env_time_zone",
        "data_frequency": "env_timestep",
        "frequency": "env_timestep",
        "variable_name": "pl_variable_name",
    }

    # Rename columns

    meta_df_sfn = meta_df.rename(columns=lambda c: str(c).lower().replace(" ", "_")).rename(columns=RENAME_MAP)

    meta_df_sfn["st_site_id"] = meta_df_sfn.get("si_code")

    # Data transformations

    if "data_frequency" in meta_df.columns or "env_timestep" in meta_df_sfn.columns:
        freq_col = "env_timestep" if "env_timestep" in meta_df_sfn.columns else None

        if not freq_col and "data_frequency" in meta_df.columns:
            meta_df_sfn["pl_sens_timestep"] = meta_df["data_frequency"].astype(str).str.extract(r"(\d+)").astype(float)

    # Copy site coordinates from plant if missing

    if "si_lat" in meta_df_sfn.columns and "si_long" in meta_df_sfn.columns:
        if meta_df_sfn["si_lat"].isna().any() and meta_df_sfn["si_long"].isna().any():
            if "pl_lat" in meta_df_sfn.columns and "pl_long" in meta_df_sfn.columns:
                meta_df_sfn["si_lat"] = meta_df_sfn["pl_lat"].iloc[0]

                meta_df_sfn["si_long"] = meta_df_sfn["pl_long"].iloc[0]

    # Parse soil composition

    if "site % clay, silt, sand" in meta_df.columns and meta_df["site % clay, silt, sand"].notna().any():
        soil_str = meta_df["site % clay, silt, sand"].dropna().iloc[0]

        if isinstance(soil_str, str):
            sand_match = re.search(r"(\d+)\s*%\s*sand", soil_str, re.IGNORECASE)

            silt_match = re.search(r"(\d+)\s*%\s*(silt|dust)", soil_str, re.IGNORECASE)

            clay_match = re.search(r"(\d+)\s*%\s*(clay|loam)", soil_str, re.IGNORECASE)

            meta_df_sfn["st_sand_perc"] = float(sand_match.group(1)) if sand_match else None

            meta_df_sfn["st_silt_perc"] = float(silt_match.group(1)) if silt_match else None

            meta_df_sfn["st_clay_perc"] = float(clay_match.group(1)) if clay_match else None

    # Handle remarks

    if "other_notes" in meta_df.columns and meta_df["other_notes"].notna().any():
        meta_df_sfn["si_remarks"] = meta_df["other_notes"].dropna().iloc[0]

    if "notes" in meta_df.columns:
        meta_df_sfn["pl_remarks"] = meta_df["notes"]

    # Determine output folder based on units

    output_folder = "plant"

    try:
        # Find units column - check multiple possible names (original or tidied)
        # Priority: pl_sap_units_orig (tidied), then 'units' (original)
        units_col = None
        unit_string = ""

        # First check in tidied meta_df_sfn for pl_sap_units_orig
        if "pl_sap_units_orig" in meta_df_sfn.columns and meta_df_sfn["pl_sap_units_orig"].notna().any():
            unit_string = str(meta_df_sfn["pl_sap_units_orig"].iloc[0])
        # Fallback to original meta_df 'units' column
        elif "units" in [str(col).lower().strip() for col in meta_df.columns]:
            units_col = next((col for col in meta_df.columns if str(col).lower().strip() == "units"), None)
            if units_col and meta_df[units_col].notna().any():
                unit_string = str(meta_df[units_col].iloc[0])

        # Normalize unicode characters for comparison
        unit_lower = unit_string.lower()
        unit_normalized = unit_lower.replace("²", "2").replace("³", "3").replace("⁻", "-").replace("¹", "1")
        unit_normalized = unit_normalized.replace("\u00b2", "2").replace("\u00b3", "3").replace("\u207b", "-")

        print(f"    INFO: Determining output folder based on units: '{unit_normalized}'")

        # First check for leaf-specific measurements

        if "leaf" in unit_normalized:
            output_folder = "leaf"

        # Check for area component in denominator (sapwood level: mass or volume per area per time)

        elif any(
            pattern in unit_normalized
            for pattern in [
                "cm-2",
                "cm2",
                "cmÂ²",
                "/cm2",
                "/cmÂ²",
                "cm^-2",
                "cm^2",
                "per cm2",
                "per cmÂ²",
                "cmâ»Â²",
                "cm â»Â²",
                "m-2",
                "m2",
                "mÂ²",
                "/m2",
                "/mÂ²",
                "m^-2",
                "m^2",
                "per m2",
                "per mÂ²",
                "mm-2",
                "mm2",
                "mmÂ²",
                "/mm2",
                "/mmÂ²",
                "mm^-2",
                "mm^2",
                "per mm2",
                "per mmÂ²",
                "sapwood area",
                "per area",
                "per cm ",
                "per m ",
                "per mm ",
                "per sapwood ",
                "cm/h",
                "m/h",
                "mm/h",
                "cm h-1",
                "m h-1",
                "mm h-1",
                "cm/s",
                "m/s",
                "mm/s",
                "cm s-1",
                "m s-1",
                "mm s-1",
                "cm/hour",
            ]
        ):
            output_folder = "sapwood"

        # Otherwise it's plant level (mass or volume per time, no area component)

        else:
            output_folder = "plant"

        print(f"    INFO: Output folder '{output_folder}' (unit: '{unit_normalized}')")

    except Exception as e:
        print(f"    WARNING: Could not determine output folder. Using 'plant'. Error: {e}")

    # Define output columns

    MAPS = {
        "site_md": [
            "si_code",
            "si_country",
            "si_lat",
            "si_long",
            "si_elev",
            "si_contact_institution",
            "si_contact_name",
            "si_flux_tower",
            "si_remarks",
            "si_other_remarks",
        ],
        "stand_md": [
            "st_site_id",
            "st_forest_type",
            "st_soil_type",
            "st_density",
            "st_basal_area",
            "st_height",
            "st_lai",
            "st_sand_perc",
            "st_silt_perc",
            "st_clay_perc",
        ],
        "plant_md": [
            "pl_code",
            "pl_name",
            "pl_species",
            "pl_genus",
            "pl_dbh",
            "pl_height",
            "pl_age",
            "pl_social",
            "pl_bark_thick",
            "pl_phloem_thick",
            "pl_sapw_depth",
            "pl_sapw_area",
            "pl_lat",
            "pl_long",
            "pl_elev",
            "pl_sens_meth",
            "pl_sens_hgt",
            "pl_sensor_id",
            "pl_series_id",
            "pl_sensor_start",
            "pl_sensor_end",
            "pl_sensor_cutout",
            "pl_sensor_exposition",
            "pl_sap_units_orig",
            "pl_variable_name",
            "pl_remarks",
        ],
        "species_md": ["sp_name", "sp_ntrees"],
        "env_md": ["env_time_zone", "env_timestep"],
    }

    # Create output folder path

    data_folder_path = Path(output_path, output_folder)

    data_folder_path.mkdir(exist_ok=True)

    # Write metadata files

    for md_type, cols in MAPS.items():
        cols_to_write = [col for col in cols if col in meta_df_sfn.columns]

        if not cols_to_write:
            continue

        md_df = meta_df_sfn[cols_to_write].drop_duplicates()

        if md_type in ["site_md", "stand_md", "env_md"]:
            md_df = md_df.iloc[0:1]

        elif md_type == "species_md":
            species_col = "pl_species"

            if species_col not in meta_df_sfn.columns:
                continue

            md_df = pd.DataFrame({"sp_name": meta_df_sfn[species_col].dropna().unique()})

            if not md_df.empty and "pl_code" in meta_df_sfn.columns:
                species_counts = meta_df_sfn.groupby(species_col)["pl_code"].nunique()

                md_df = md_df.merge(species_counts, left_on="sp_name", right_index=True).rename(
                    columns={"pl_code": "sp_ntrees"}
                )

        md_df.to_csv(data_folder_path / f"{site_code}_{md_type}.csv", index=False)

    # Write sapflow data

    sapflow_df.to_csv(data_folder_path / f"{site_code}_sapflow_{output_folder}.csv")

    # Write environmental data (empty if no env data collected)

    if env_df is not None and not env_df.empty:
        env_df.to_csv(data_folder_path / f"{site_code}_env_data.csv")

        print(f"    INFO: Wrote {len(env_df.columns)} environmental variable(s) to env_data.csv")

    else:
        pd.DataFrame(index=sapflow_df.index).to_csv(data_folder_path / f"{site_code}_env_data.csv")

    print(f"    âœ“ Successfully wrote SAPFLUXNET files for {site_code}")


def calculate_formatted_site_years(output_dir: str, max_gap_days: int = 366):
    """

    Calculates total site-years, deducting significant gaps and rows with no data.

    Follows the logic:

    1. Drop rows where all value columns are NaN.

    2. Calculate time difference between remaining consecutive rows.

    3. Sum durations, excluding gaps > max_gap_days.

    """

    from pathlib import Path

    import pandas as pd

    output_path = Path(output_dir)

    # Find all sapflow files recursively

    all_files = list(output_path.rglob("*_sapflow_*.csv"))

    # Filter out dendro files explicitly

    all_files = [f for f in all_files if "dendro" not in f.name.lower()]

    print("\n" + "=" * 80)

    print(f"CALCULATING SITE-YEARS (Gap Corrected | Threshold: {max_gap_days} days)")

    print(f"Found {len(all_files)} files.")

    print("=" * 80)

    stats = []

    for file in all_files:
        try:
            # OPTIMIZATION: Read file

            # We need all columns to check for NaNs, but we can process efficiently

            df = pd.read_csv(file)

            # Normalize column names to lowercase for checking

            df.columns = [c.lower() for c in df.columns]

            if "timestamp" not in df.columns:
                print(f"Skipping {file.name}: No timestamp column.")

                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")

            # Drop rows where TIMESTAMP is NaT

            df = df.dropna(subset=["timestamp"])

            length_raw = len(df)

            # Identify Value Columns (Exclude timestamp and potential metadata cols)

            # We only want to keep rows where at least ONE sap flow value exists

            non_value_cols = ["timestamp", "env_time_zone", "env_timestep", "doy", "timesteps", "solar_TIMESTAMP"]

            value_cols = [c for c in df.columns if c not in non_value_cols and "unnamed" not in c]

            if not value_cols:
                print(f"Skipping {file.name}: No value columns found.")

                continue

            # Drop rows when ALL value columns are NA (keep timestamp)

            # This removes periods where the logger was running but sensors were broken/NaN

            df = df.dropna(how="all", subset=value_cols)

            length_dropped = len(df)

            rows_removed = length_raw - length_dropped

            if len(df) > 1:
                # 1. Sort Data (Crucial for diff calculations)

                df = df.sort_values("timestamp")

                # 2. Calculate time difference between consecutive rows

                # diff() gives us: [Row2-Row1, Row3-Row2, ...]

                deltas = df["timestamp"].diff()

                # 3. Define the threshold for a "gap"

                gap_threshold = pd.Timedelta(days=max_gap_days)

                # 4. Filter: Keep only durations that are connected (smaller than gap threshold)

                # If the difference is > gap_threshold (e.g., missing winter), it is excluded.

                valid_durations = deltas[deltas <= gap_threshold]

                # 5. Sum valid durations

                total_duration_seconds = valid_durations.sum().total_seconds()

                # Calculate simple Min/Max for reporting raw coverage

                start_date = df["timestamp"].min()

                end_date = df["timestamp"].max()

                # Convert to years

                years = total_duration_seconds / (3600 * 24 * 365.25)

                # Extract Site Code cleanly

                filename = file.name

                if "_sapflow_" in filename:
                    base_name = filename.split("_sapflow_")[0]

                    # Remove frequency suffixes if present

                    site_code = base_name.replace("_daily", "").replace("_halfhourly", "").replace("_hourly", "")

                else:
                    site_code = file.stem

                # Calculate simple percent coverage (Effective Time / Total Calendar Time)

                total_calendar_seconds = (end_date - start_date).total_seconds()

                coverage_pct = 0

                if total_calendar_seconds > 0:
                    coverage_pct = (total_duration_seconds / total_calendar_seconds) * 100

                stats.append(
                    {
                        "site_code": site_code,
                        "filename": file.name,
                        "start_date": start_date,
                        "end_date": end_date,
                        "years": years,
                        "rows_removed": rows_removed,
                        "coverage_percent": coverage_pct,
                    }
                )

                print(f"  Processed {site_code:<15} | Dropped {rows_removed} empty rows | Valid Years: {years:.2f}")

            else:
                print(f"  Skipping {file.name}: Not enough data points after cleaning.")

        except Exception as e:
            print(f"  Error reading {file.name}: {e}")

    summary_df = pd.DataFrame(stats)

    if not summary_df.empty:
        # Aggregate by Site Code to avoid double counting

        # (e.g. if daily and half-hourly files exist for same site, take the one with MORE data)

        site_summary = summary_df.loc[summary_df.groupby("site_code")["years"].idxmax()]

        total_site_years = site_summary["years"].sum()

        print("-" * 80)

        print(f"{'SITE CODE':<20} | {'START':<12} | {'END':<12} | {'YEARS':<8} | {'COV %':<6} | {'GAP-REMOVED'}")

        print("-" * 80)

        for _, row in site_summary.sort_values("site_code").iterrows():
            s_date = row["start_date"].strftime("%Y-%m-%d")

            e_date = row["end_date"].strftime("%Y-%m-%d")

            print(
                f"{row['site_code']:<20} | {s_date:<12} | {e_date:<12} | {row['years']:.2f}     | {row['coverage_percent']:.1f}%   | {row['rows_removed']} rows"
            )

        print("-" * 80)

        print("CALCULATION COMPLETE")

        print("-" * 80)

        print(f"Total Unique Sites: {len(site_summary)}")

        print(f"TOTAL SITE-YEARS:   {total_site_years:.2f}")

        print("-" * 80)

        return site_summary

    else:
        print("No valid data found.")

        return pd.DataFrame()


def main():

    base_path = Path(__file__).resolve().parent

    input_root, output_root = base_path / INPUT_DIR, base_path / OUTPUT_DIR

    if not input_root.exists():
        print(f"ERROR: Input directory '{INPUT_DIR}' not found.")
        return

    print(f"Starting SAPFLUXNET formatting...\nInput: '{input_root}'\nOutput: '{output_root}'\n")

    output_root.mkdir(exist_ok=True)

    processor = {
        "AT_Mmg": process_pattern_A,
        "CH-Dav": process_pattern_C,
        "CH-Lae": process_pattern_C,
        "DE-Har": process_pattern_A,
        "DE-HoH": process_pattern_A,
        "ES-Abr": process_pattern_A,
        "ES-Gdn": process_pattern_A,
        "ES-LM1": process_pattern_A,
        "ES-LM2": process_pattern_A,
        "ES-LMa": process_pattern_A,
        "FI-Hyy": process_pattern_A,
        "FR-BIL": process_pattern_B,
        # IT-CP2 has transposed metadata - use dedicated scripts instead:
        # 'IT-CP2_plant': process_pattern_B,    # Plant-level: Sap columns (g h-1) -> process_ITCP2_plant.py
        # 'IT-CP2_sapwood': process_pattern_B,  # Sapwood-level: TDSV columns (cm s-1) -> process_ITCP2_sapwood.py
        "NO-Hur": process_pattern_B,
        "PL-Mez": process_pattern_A,
        "PL-Tuc": process_pattern_A,
        "SE-Nor": process_pattern_A,
        "SE-Sgr": process_pattern_A,
        "ZOE_AT": process_pattern_B,
    }

    # for site_code in sorted(os.listdir(input_root)):
    for site_code in processor:
        site_path = input_root / site_code

        if not site_path.is_dir():
            continue

        create_sfn_directories(output_root, site_code)

        metadata_file, data_files = get_site_files(site_path)

        if metadata_file and data_files:
            try:
                processor[site_code](site_path, site_code, metadata_file, data_files, output_root)

            except Exception as e:
                print(f"  !!! UNEXPECTED ERROR processing {site_code}: {e}")

        else:
            print(f"--> Skipping '{site_code}': Missing metadata or data files.")

    calculate_formatted_site_years(output_root)

    print("\nProcessing complete.")


if __name__ == "__main__":
    main()

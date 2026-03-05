from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RemovalEntry:
    """Data class representing a single removal log entry."""
    entry_id: int
    location: str
    plant_type: str
    column: Optional[str]
    start_time: Optional[pd.Timestamp]
    end_time: Optional[pd.Timestamp]
    issue_category: str
    removal_type: str  # 'site_entire', 'column_entire', 'period'
    notes: str
    decision_date: Optional[pd.Timestamp]
    reviewer: str


class RemovalLogProcessor:
    """
    Processor for handling visual inspection removal logs.
    
    Supports three levels of removal:
    - site_entire: Remove all data for a location/plant_type combination
    - column_entire: Remove an entire column (sensor/tree)
    - period: Remove specific time periods for a column
    
    Usage:
        processor = RemovalLogProcessor()
        processor.load_removal_log('path/to/removal_log.csv')
        cleaned_df = processor.apply_removals(df, location='DEU_HIN', plant_type='OAK_sapf_data')
    """
    
    def __init__(self):
        self.removal_entries: List[RemovalEntry] = []
        self.removal_log_df: Optional[pd.DataFrame] = None
        self.removal_summary: Dict = {
            'sites_removed': [],
            'columns_removed': [],
            'periods_removed': [],
            'total_points_removed': 0
        }
    
    def load_removal_log(self, log_path: Union[str, Path], 
                         validate: bool = True) -> pd.DataFrame:
        """
        Load and parse a removal log CSV file.
        
        Args:
            log_path: Path to the removal log CSV file.
            validate: If True, validate the log entries and report issues.
            
        Returns:
            DataFrame containing the parsed removal log.
            
        Expected CSV columns:
            - entry_id: Unique identifier
            - location: Site location code
            - plant_type: Plant type identifier
            - column: Column/sensor name (optional for site_entire)
            - start_time: Start of removal period (for period type)
            - end_time: End of removal period (for period type)
            - issue_category: Category of the identified issue
            - removal_type: 'site_entire', 'column_entire', or 'period'
            - notes: Description of the issue
            - decision_date: When the decision was made
            - reviewer: Person who made the decision
        """
        log_path = Path(log_path)
        
        if not log_path.exists():
            raise FileNotFoundError(f"Removal log file not found: {log_path}")
        
        # Read the CSV file
        df = pd.read_csv(log_path)
        
        # Standardize column names (handle variations)
        column_mapping = {
            'entry_id': 'entry_id',
            'id': 'entry_id',
            'location': 'location',
            'site': 'location',
            'plant_type': 'plant_type',
            'column': 'column',
            'sensor': 'column',
            'tree': 'column',
            'start_time': 'start_time',
            'start_date': 'start_time',
            'start': 'start_time',
            'end_time': 'end_time',
            'end_date': 'end_time',
            'end': 'end_time',
            'issue_category': 'issue_category',
            'category': 'issue_category',
            'issue': 'issue_category',
            'removal_type': 'removal_type',
            'type': 'removal_type',
            'notes': 'notes',
            'description': 'notes',
            'decision_date': 'decision_date',
            'date': 'decision_date',
            'reviewer': 'reviewer',
            'reviewed_by': 'reviewer'
        }
        
        # Apply column mapping
        df.columns = df.columns.str.lower().str.strip()
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Parse datetime columns (but preserve semicolon-separated strings for later parsing)
        for col in ['start_time', 'end_time']:
            if col in df.columns:
                # Check if any values contain semicolons - if so, keep as string
                has_semicolons = df[col].astype(str).str.contains(';', na=False).any()
                if not has_semicolons:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                # If has semicolons, keep as string - will be parsed in _parse_entries
        
        # Parse decision_date separately (should never have semicolons)
        if 'decision_date' in df.columns:
            df['decision_date'] = pd.to_datetime(df['decision_date'], errors='coerce')
        
        # Standardize removal_type values
        if 'removal_type' in df.columns:
            df['removal_type'] = df['removal_type'].str.lower().str.strip()
            # Map common variations
            type_mapping = {
                'site': 'site_entire',
                'site_entire': 'site_entire',
                'entire_site': 'site_entire',
                'whole_site': 'site_entire',
                'column': 'column_entire',
                'column_entire': 'column_entire',
                'entire_column': 'column_entire',
                'sensor': 'column_entire',
                'tree': 'column_entire',
                'period': 'period',
                'time_period': 'period',
                'date_range': 'period'
            }
            df['removal_type'] = df['removal_type'].map(
                lambda x: type_mapping.get(x, x) if pd.notna(x) else x
            )
        
        self.removal_log_df = df
        
        # Parse entries into RemovalEntry objects
        self._parse_entries()
        
        if validate:
            self._validate_entries()
        
        print(f"Loaded {len(self.removal_entries)} removal entries from {log_path.name}")
        self._print_summary()
        
        return df
    
    def _parse_entries(self):
        """
        Parse DataFrame rows into RemovalEntry objects.
        
        Supports multiple periods in a single row using semicolon separation:
        - start_time: "2020-01-01;2020-06-01;2020-09-01"
        - end_time: "2020-01-31;2020-06-30;2020-09-30"
        
        Each semicolon-separated pair creates a separate RemovalEntry object.
        """
        self.removal_entries = []
        
        for _, row in self.removal_log_df.iterrows():
            # Get base entry information
            entry_id = row.get('entry_id', 0)
            location = str(row.get('location', '')).strip()
            plant_type = str(row.get('plant_type', '')).strip()
            column = str(row.get('column', '')).strip() if pd.notna(row.get('column')) else None
            issue_category = str(row.get('issue_category', '')).strip()
            removal_type = str(row.get('removal_type', '')).strip()
            notes = str(row.get('notes', '')).strip() if pd.notna(row.get('notes')) else ''
            decision_date = row.get('decision_date') if pd.notna(row.get('decision_date')) else None
            reviewer = str(row.get('reviewer', '')).strip() if pd.notna(row.get('reviewer')) else ''
            
            # Get start and end time values
            start_time_raw = row.get('start_time')
            end_time_raw = row.get('end_time')
            
            # Check if this is a period removal with semicolon-separated dates
            if removal_type == 'period' and pd.notna(start_time_raw) and pd.notna(end_time_raw):
                # Convert to string to check for semicolons
                start_str = str(start_time_raw)
                end_str = str(end_time_raw)
                
                # Check if semicolons are present (multiple periods in one row)
                if ';' in start_str or ';' in end_str:
                    # Split by semicolon
                    start_times = [s.strip() for s in start_str.split(';') if s.strip()]
                    end_times = [e.strip() for e in end_str.split(';') if e.strip()]
                    
                    # Validate matching counts
                    if len(start_times) != len(end_times):
                        print(f"Warning: Entry {entry_id} has mismatched start/end time counts "
                              f"({len(start_times)} starts vs {len(end_times)} ends). "
                              f"Using minimum count.")
                    
                    # Create an entry for each period pair
                    num_periods = min(len(start_times), len(end_times))
                    for i in range(num_periods):
                        try:
                            parsed_start = pd.to_datetime(start_times[i])
                            parsed_end = pd.to_datetime(end_times[i])
                        except Exception as e:
                            print(f"Warning: Could not parse dates for entry {entry_id}, "
                                  f"period {i+1}: {start_times[i]} to {end_times[i]}. Error: {e}")
                            continue
                        
                        entry = RemovalEntry(
                            entry_id=entry_id,
                            location=location,
                            plant_type=plant_type,
                            column=column,
                            start_time=parsed_start,
                            end_time=parsed_end,
                            issue_category=issue_category,
                            removal_type=removal_type,
                            notes=f"{notes} [Period {i+1}/{num_periods}]" if num_periods > 1 else notes,
                            decision_date=decision_date,
                            reviewer=reviewer
                        )
                        self.removal_entries.append(entry)
                    
                    # Log the expansion
                    if num_periods > 1:
                        print(f"  Entry {entry_id}: Expanded {num_periods} periods for {column}")
                    continue
            
            # Standard single-period or non-period entry
            start_time = None
            end_time = None
            
            if pd.notna(start_time_raw):
                try:
                    start_time = pd.to_datetime(start_time_raw)
                except:
                    start_time = None
                    
            if pd.notna(end_time_raw):
                try:
                    end_time = pd.to_datetime(end_time_raw)
                except:
                    end_time = None
            
            entry = RemovalEntry(
                entry_id=entry_id,
                location=location,
                plant_type=plant_type,
                column=column,
                start_time=start_time,
                end_time=end_time,
                issue_category=issue_category,
                removal_type=removal_type,
                notes=notes,
                decision_date=decision_date,
                reviewer=reviewer
            )
            self.removal_entries.append(entry)
    
    def _validate_entries(self):
        """Validate removal entries and report any issues."""
        issues = []
        
        for entry in self.removal_entries:
            entry_id = f"Entry {entry.entry_id}"
            
            # Check required fields
            if not entry.location:
                issues.append(f"{entry_id}: Missing location")
            if not entry.plant_type:
                issues.append(f"{entry_id}: Missing plant_type")
            if not entry.removal_type:
                issues.append(f"{entry_id}: Missing removal_type")
            
            # Validate based on removal type
            if entry.removal_type == 'period':
                if not entry.column:
                    issues.append(f"{entry_id}: Period removal requires column specification")
                if entry.start_time is None:
                    issues.append(f"{entry_id}: Period removal requires start_time")
                if entry.end_time is None:
                    issues.append(f"{entry_id}: Period removal requires end_time")
                if entry.start_time and entry.end_time and entry.start_time > entry.end_time:
                    issues.append(f"{entry_id}: start_time is after end_time")
                    
            elif entry.removal_type == 'column_entire':
                if not entry.column:
                    issues.append(f"{entry_id}: Column removal requires column specification")
                    
            elif entry.removal_type == 'site_entire':
                # Site entire doesn't require column
                pass
            else:
                issues.append(f"{entry_id}: Unknown removal_type '{entry.removal_type}'")
        
        if issues:
            print("\nValidation warnings:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("All entries validated successfully.")
    
    def _print_summary(self):
        """Print summary of loaded removal entries."""
        if not self.removal_entries:
            return
        
        # Count by type
        type_counts = {}
        for entry in self.removal_entries:
            type_counts[entry.removal_type] = type_counts.get(entry.removal_type, 0) + 1
        
        # Unique locations/plant_types
        unique_sites = set((e.location, e.plant_type) for e in self.removal_entries)
        unique_columns = set(e.column for e in self.removal_entries if e.column)
        
        print(f"\nRemoval Log Summary:")
        print(f"  Total entries: {len(self.removal_entries)}")
        print(f"  Unique sites: {len(unique_sites)}")
        print(f"  Unique columns: {len(unique_columns)}")
        print(f"  By removal type:")
        for rtype, count in sorted(type_counts.items()):
            print(f"    - {rtype}: {count}")
        
        # Show the location/plant_type combinations in the log
        print(f"\n  Sites in removal log (location + plant_type):")
        for loc, pt in sorted(unique_sites):
            print(f"    - {loc} | {pt}")
    
    def diagnose_file_matching(self, file_path: Union[str, Path]):
        """
        Diagnose why a file might not match removal log entries.
        
        This helper method shows how a filename is parsed and compares
        it against entries in the removal log.
        
        Args:
            file_path: Path to a sapflow data file.
        """
        file_path = Path(file_path)
        parts = file_path.stem.split('_')
        
        # Standard parsing (first 2 parts = location, rest = plant_type)
        location = '_'.join(parts[:2])
        plant_type = '_'.join(parts[2:])
        
        print(f"\n{'='*60}")
        print(f"FILE MATCHING DIAGNOSIS")
        print(f"{'='*60}")
        print(f"\nFilename: {file_path.name}")
        print(f"Split parts: {parts}")
        print(f"\nParsed values (standard):")
        print(f"  location:   '{location}'")
        print(f"  plant_type: '{plant_type}'")
        
        # Check against removal log entries
        if not self.removal_entries:
            print(f"\nNo removal entries loaded.")
            return
        
        print(f"\n{'='*60}")
        print(f"MATCHING AGAINST REMOVAL LOG")
        print(f"{'='*60}")
        
        # Exact match check
        exact_matches = self.get_entries_for_site(location, plant_type)
        if exact_matches:
            print(f"\n✓ EXACT MATCH FOUND: {len(exact_matches)} entries")
            for entry in exact_matches:
                print(f"    - {entry.column} ({entry.removal_type})")
        else:
            print(f"\n✗ NO EXACT MATCH for location='{location}', plant_type='{plant_type}'")
            
            # Show what's in the removal log for comparison
            print(f"\nEntries in removal log:")
            unique_sites = set((e.location, e.plant_type) for e in self.removal_entries)
            for log_loc, log_pt in sorted(unique_sites):
                match_loc = "✓" if log_loc == location else "✗"
                match_pt = "✓" if log_pt == plant_type else "✗"
                print(f"  location: '{log_loc}' {match_loc} | plant_type: '{log_pt}' {match_pt}")
            
            # Suggest corrections
            print(f"\n{'='*60}")
            print(f"SUGGESTED FIX")
            print(f"{'='*60}")
            print(f"\nUpdate your removal log CSV to use these exact values:")
            print(f"  location:   {location}")
            print(f"  plant_type: {plant_type}")
    
    def list_expected_site_format(self, data_dir: Union[str, Path], 
                                   file_pattern: str = "*_sapf_data.csv"):
        """
        List all files in a directory and show how they would be parsed.
        
        This helps you create a removal log with correct location/plant_type values.
        
        Args:
            data_dir: Directory containing sapflow data files.
            file_pattern: Glob pattern to match files.
        """
        data_dir = Path(data_dir)
        files = list(data_dir.glob(file_pattern))
        
        print(f"\n{'='*60}")
        print(f"FILE PARSING REFERENCE")
        print(f"{'='*60}")
        print(f"Directory: {data_dir}")
        print(f"Files found: {len(files)}")
        print(f"\nUse these exact values in your removal log:\n")
        print(f"{'Filename':<50} | {'location':<20} | {'plant_type'}")
        print("-" * 100)
        
        for file in sorted(files):
            parts = file.stem.split('_')
            location = '_'.join(parts[:2])
            plant_type = '_'.join(parts[2:])
            print(f"{file.name:<50} | {location:<20} | {plant_type}")
    
    def get_entries_for_site(self, location: str, plant_type: str) -> List[RemovalEntry]:
        """
        Get all removal entries for a specific site.
        
        Args:
            location: Site location code.
            plant_type: Plant type identifier.
            
        Returns:
            List of RemovalEntry objects for the specified site.
        """
        return [
            entry for entry in self.removal_entries
            if entry.location == location and entry.plant_type == plant_type
        ]
    
    def should_skip_site(self, location: str, plant_type: str) -> bool:
        """
        Check if an entire site should be skipped (not loaded at all).
        
        Args:
            location: Site location code.
            plant_type: Plant type identifier.
            
        Returns:
            True if the site should be skipped entirely.
        """
        for entry in self.removal_entries:
            if (entry.location == location and 
                entry.plant_type == plant_type and 
                entry.removal_type == 'site_entire' and
                (entry.column is None or entry.column == '' or entry.column == 'nan')):
                return True
        return False
    
    def get_columns_to_remove(self, location: str, plant_type: str) -> List[str]:
        """
        Get list of columns that should be entirely removed for a site.
        
        Args:
            location: Site location code.
            plant_type: Plant type identifier.
            
        Returns:
            List of column names to remove entirely.
        """
        columns = []
        for entry in self.removal_entries:
            if (entry.location == location and 
                entry.plant_type == plant_type and 
                entry.removal_type == 'column_entire' and
                entry.column):
                columns.append(entry.column)
        return list(set(columns))  # Remove duplicates
    
    def get_periods_to_remove(self, location: str, plant_type: str, 
                              column: str) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Get list of time periods to remove for a specific column.
        
        Args:
            location: Site location code.
            plant_type: Plant type identifier.
            column: Column/sensor name.
            
        Returns:
            List of (start_time, end_time) tuples for periods to remove.
        """
        periods = []
        for entry in self.removal_entries:
            if (entry.location == location and 
                entry.plant_type == plant_type and 
                entry.removal_type == 'period' and
                entry.column == column and
                entry.start_time is not None and
                entry.end_time is not None):
                periods.append((entry.start_time, entry.end_time))
        return periods
    
    def apply_removals(self, df: pd.DataFrame, location: str, plant_type: str,
                       inplace: bool = False, verbose: bool = True) -> pd.DataFrame:
        """
        Apply all relevant removals to a DataFrame.
        
        This method processes the DataFrame in the following order:
        1. Check if entire site should be removed (returns empty DataFrame)
        2. Remove entire columns marked for removal
        3. Remove specific time periods for remaining columns
        
        Args:
            df: DataFrame with TIMESTAMP as index OR as a column.
            location: Site location code.
            plant_type: Plant type identifier.
            inplace: If True, modify the DataFrame in place.
            verbose: If True, print detailed removal information.
            
        Returns:
            DataFrame with removals applied.
        """
        if not inplace:
            df = df.copy()
        
        # Determine the timestamp source (index or column)
        timestamp_is_index = False
        timestamp_series = None
        
        if isinstance(df.index, pd.DatetimeIndex):
            timestamp_is_index = True
            timestamp_series = df.index
        elif df.index.name == 'TIMESTAMP':
            # Index is named TIMESTAMP but might not be DatetimeIndex
            try:
                timestamp_series = pd.to_datetime(df.index)
                timestamp_is_index = True
            except:
                pass
        
        if timestamp_series is None and 'TIMESTAMP' in df.columns:
            # TIMESTAMP is a column, not the index
            timestamp_series = pd.to_datetime(df['TIMESTAMP'])
            timestamp_is_index = False
        
        if timestamp_series is None:
            raise ValueError("DataFrame must have TIMESTAMP as index or as a column")
        
        # Check if timestamp_series is timezone-aware
        timestamp_is_tz_aware = False
        if hasattr(timestamp_series, 'tz') and timestamp_series.tz is not None:
            timestamp_is_tz_aware = True
            timestamp_tz = timestamp_series.tz
        elif hasattr(timestamp_series, 'dt') and hasattr(timestamp_series.dt, 'tz') and timestamp_series.dt.tz is not None:
            timestamp_is_tz_aware = True
            timestamp_tz = timestamp_series.dt.tz
        
        # Initialize tracking
        removal_stats = {
            'site_removed': False,
            'columns_removed': [],
            'periods_removed': {},
            'points_before': df.notna().sum().sum(),
            'points_after': 0
        }
        
        # Get entries for this site
        site_entries = self.get_entries_for_site(location, plant_type)
        
        if not site_entries:
            if verbose:
                print(f"No removal entries found for {location}_{plant_type}")
            removal_stats['points_after'] = removal_stats['points_before']
            return df
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Applying removals for {location}_{plant_type}")
            print(f"Found {len(site_entries)} removal entries")
            print(f"{'='*60}")
        
        # Step 1: Check for site-level removal
        if self.should_skip_site(location, plant_type):
            if verbose:
                print(f"SITE REMOVAL: Entire site marked for removal")
            removal_stats['site_removed'] = True
            # Set all values to NaN instead of returning empty DataFrame
            # This preserves the structure for downstream processing
            data_cols = [col for col in df.columns 
                        if col not in ['TIMESTAMP', 'solar_TIMESTAMP', 'TIMESTAMP_LOCAL', 'lat', 'long']]
            df[data_cols] = np.nan
            removal_stats['points_after'] = 0
            self._update_summary(removal_stats, location, plant_type)
            return df
        
        # Step 2: Remove entire columns
        columns_to_remove = self.get_columns_to_remove(location, plant_type)
        for col in columns_to_remove:
            if col in df.columns:
                points_removed = df[col].notna().sum()
                df[col] = np.nan
                removal_stats['columns_removed'].append(col)
                if verbose:
                    print(f"COLUMN REMOVAL: {col} ({points_removed} points)")
        
        # Step 3: Remove specific periods
        data_cols = [col for col in df.columns 
                    if col not in ['TIMESTAMP', 'solar_TIMESTAMP', 'TIMESTAMP_LOCAL', 'lat', 'long']
                    and col not in columns_to_remove]
        
        # Collect all column names from removal entries for this site
        removal_entry_columns = set()
        for entry in site_entries:
            if entry.removal_type == 'period' and entry.column:
                removal_entry_columns.add(entry.column)
        
        # Check for column mismatches and warn
        if removal_entry_columns and verbose:
            matched_cols = removal_entry_columns.intersection(set(data_cols))
            unmatched_cols = removal_entry_columns - set(data_cols)
            
            if unmatched_cols:
                print(f"\nWARNING: Column name mismatch detected!")
                print(f"  Columns in removal log: {sorted(removal_entry_columns)}")
                print(f"  Columns in DataFrame:   {sorted(data_cols)}")
                print(f"  UNMATCHED (will not be removed): {sorted(unmatched_cols)}")
                
                # Suggest possible matches
                for unmatched in unmatched_cols:
                    similar = [c for c in data_cols if unmatched.lower() in c.lower() or c.lower() in unmatched.lower()]
                    if similar:
                        print(f"  -> '{unmatched}' might match: {similar}")
        
        for col in data_cols:
            periods = self.get_periods_to_remove(location, plant_type, col)
            if periods:
                removal_stats['periods_removed'][col] = []
                for start_time, end_time in periods:
                    # Handle timezone-aware timestamps
                    compare_start = start_time
                    compare_end = end_time
                    
                    if timestamp_is_tz_aware:
                        # Localize naive timestamps to match the DataFrame's timezone
                        if start_time.tzinfo is None:
                            compare_start = start_time.tz_localize(timestamp_tz)
                        else:
                            compare_start = start_time.tz_convert(timestamp_tz)
                        
                        if end_time.tzinfo is None:
                            compare_end = end_time.tz_localize(timestamp_tz)
                        else:
                            compare_end = end_time.tz_convert(timestamp_tz)
                    
                    # Create mask for the time period using the timestamp series
                    mask = (timestamp_series >= compare_start) & (timestamp_series <= compare_end)
                    
                    # Convert mask to proper format for DataFrame indexing
                    if timestamp_is_index:
                        points_removed = df.loc[mask, col].notna().sum()
                        if points_removed > 0:
                            df.loc[mask, col] = np.nan
                    else:
                        # TIMESTAMP is a column, use boolean array indexing
                        mask_array = mask.values if hasattr(mask, 'values') else mask
                        points_removed = df.loc[mask_array, col].notna().sum()
                        if points_removed > 0:
                            df.loc[mask_array, col] = np.nan
                    
                    if points_removed > 0:
                        removal_stats['periods_removed'][col].append({
                            'start': start_time,
                            'end': end_time,
                            'points': points_removed
                        })
                        if verbose:
                            print(f"PERIOD REMOVAL: {col} [{start_time.date()} to {end_time.date()}] ({points_removed} points)")
        
        # Calculate final statistics
        removal_stats['points_after'] = df.notna().sum().sum()
        points_removed = removal_stats['points_before'] - removal_stats['points_after']
        
        if verbose:
            print(f"\n--- Summary ---")
            print(f"Points before: {removal_stats['points_before']}")
            print(f"Points after: {removal_stats['points_after']}")
            print(f"Total points removed: {points_removed}")
            if removal_stats['points_before'] > 0:
                pct = (points_removed / removal_stats['points_before']) * 100
                print(f"Removal percentage: {pct:.2f}%")
        
        self._update_summary(removal_stats, location, plant_type)
        
        return df
    
    def _update_summary(self, stats: Dict, location: str, plant_type: str):
        """Update the overall removal summary."""
        site_key = f"{location}_{plant_type}"
        
        if stats['site_removed']:
            self.removal_summary['sites_removed'].append(site_key)
        
        for col in stats['columns_removed']:
            self.removal_summary['columns_removed'].append(f"{site_key}.{col}")
        
        for col, periods in stats.get('periods_removed', {}).items():
            for period in periods:
                self.removal_summary['periods_removed'].append({
                    'site': site_key,
                    'column': col,
                    'start': period['start'],
                    'end': period['end'],
                    'points': period['points']
                })
        
        self.removal_summary['total_points_removed'] += (
            stats['points_before'] - stats['points_after']
        )
    
    def get_removal_report(self) -> pd.DataFrame:
        """
        Generate a detailed report of all removals applied.
        
        Returns:
            DataFrame with removal details.
        """
        records = []
        
        for site in self.removal_summary['sites_removed']:
            records.append({
                'site': site,
                'column': 'ALL',
                'removal_type': 'site_entire',
                'start_time': None,
                'end_time': None,
                'points_removed': 'ALL'
            })
        
        for col_full in self.removal_summary['columns_removed']:
            site, col = col_full.rsplit('.', 1)
            records.append({
                'site': site,
                'column': col,
                'removal_type': 'column_entire',
                'start_time': None,
                'end_time': None,
                'points_removed': 'ALL'
            })
        
        for period in self.removal_summary['periods_removed']:
            records.append({
                'site': period['site'],
                'column': period['column'],
                'removal_type': 'period',
                'start_time': period['start'],
                'end_time': period['end'],
                'points_removed': period['points']
            })
        
        return pd.DataFrame(records)
    
    def save_removal_report(self, output_path: Union[str, Path]):
        """Save the removal report to a CSV file."""
        report = self.get_removal_report()
        report.to_csv(output_path, index=False)
        print(f"Removal report saved to {output_path}")


# =============================================================================
# Integration methods for SapFlowAnalyzer
# =============================================================================

def add_removal_log_support(analyzer_class):
    """
    Decorator/function to add removal log support to SapFlowAnalyzer class.
    
    Usage:
        from removal_log_processor import add_removal_log_support
        add_removal_log_support(SapFlowAnalyzer)
        
        # Then use as normal
        analyzer = SapFlowAnalyzer()
        analyzer.set_removal_log('path/to/removal_log.csv')
        analyzer.run_analysis_in_batches(batch_size=10)
    """
    
    # Store original _load_sapflow_data method
    original_load = analyzer_class._load_sapflow_data
    
    def set_removal_log(self, log_path: Union[str, Path], validate: bool = True):
        """
        Set the removal log to use during data processing.
        
        Args:
            log_path: Path to the removal log CSV file.
            validate: If True, validate the log entries.
        """
        self.removal_processor = RemovalLogProcessor()
        self.removal_processor.load_removal_log(log_path, validate=validate)
        self._removal_log_path = log_path
        print(f"\nRemoval log set: {log_path}")
    
    def _load_sapflow_data_with_removal(self, files_to_load):
        """
        Modified _load_sapflow_data that applies removal log before processing.
        """
        # Check if removal processor is set
        has_removal_log = hasattr(self, 'removal_processor') and self.removal_processor is not None
        
        if has_removal_log:
            print("\n" + "="*60)
            print("REMOVAL LOG ACTIVE - Will apply removals during loading")
            print("="*60)
        
        # Filter out files for sites that should be entirely skipped
        filtered_files = []
        skipped_sites = []
        
        for file in files_to_load:
            parts = file.stem.split('_')
            location = '_'.join(parts[:2])
            plant_type = '_'.join(parts[2:])
            
            if has_removal_log and self.removal_processor.should_skip_site(location, plant_type):
                skipped_sites.append(f"{location}_{plant_type}")
                print(f"SKIPPING ENTIRE SITE: {location}_{plant_type}")
            else:
                filtered_files.append(file)
        
        if skipped_sites:
            print(f"\nSkipped {len(skipped_sites)} entire sites based on removal log")
        
        # Call original load method with filtered files
        original_load(self, filtered_files)
        
        # Apply column and period removals to loaded data
        if has_removal_log:
            print("\n" + "-"*60)
            print("Applying column and period removals to loaded data...")
            print("-"*60)
            
            for location in list(self.sapflow_data.keys()):
                for plant_type in list(self.sapflow_data[location].keys()):
                    df = self.sapflow_data[location][plant_type]
                    
                    # Apply removals
                    df = self.removal_processor.apply_removals(
                        df, location, plant_type, 
                        inplace=False, verbose=True
                    )
                    
                    self.sapflow_data[location][plant_type] = df
    
    def save_removal_report(self, output_dir: Union[str, Path] = None):
        """
        Save a report of all removals applied during processing.
        
        Args:
            output_dir: Directory to save the report. Defaults to paths.sap_outliers_removed_dir.
        """
        if not hasattr(self, 'removal_processor') or self.removal_processor is None:
            print("No removal log has been set.")
            return
        
        if output_dir is None:
            output_dir = self.paths.sap_outliers_removed_dir
        
        output_path = Path(output_dir) / 'removal_log_report.csv'
        self.removal_processor.save_removal_report(output_path)
    
    # Add methods to the class
    analyzer_class.set_removal_log = set_removal_log
    analyzer_class._load_sapflow_data = _load_sapflow_data_with_removal
    analyzer_class._original_load_sapflow_data = original_load
    analyzer_class.save_removal_report = save_removal_report
    
    # Initialize removal processor as None
    if not hasattr(analyzer_class, 'removal_processor'):
        analyzer_class.removal_processor = None
    
    return analyzer_class
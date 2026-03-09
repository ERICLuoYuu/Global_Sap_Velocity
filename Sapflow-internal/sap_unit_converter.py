"""
Sapflow Unit Converter
======================
Converts formatted sapflow data to standardized units:
- Sapwood-normalized data (sap flux density) -> cm3 cm-2 h-1
- Plant-level data (sap flow rate) -> cm3 h-1

Conversion factors based on water density = 1 g/cm3 = 1 kg/L
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import shutil
from typing import Tuple, Optional, Dict, List

# =============================================================================
# UNIT PARSING AND CONVERSION FACTORS
# =============================================================================

# Target units
TARGET_SAPWOOD_UNIT = "cm3 cm-2 h-1"  # Sap flux density
TARGET_PLANT_UNIT = "cm3 h-1"         # Sap flow rate (volume per time)

# Unit patterns and their conversion factors to target units

UNIT_PATTERNS = {
    # =========================================================================
    # SAP FLUX DENSITY UNITS (per sapwood area) -> convert to cm3 cm-2 h-1
    # =========================================================================
    
    # cm/hour (velocity) - already cm3/cm2/h since cm = cm3/cm2
    r'cm[/\s]*h(our)?': {
        'factor': 1.0,  # cm/h = cm3/cm2/h
        'time_to_hour': 1.0,
        'type': 'sapwood',
        'description': 'cm/hour = cm3 cm-2 h-1 (velocity)'
    },
    
    # cm/s or cm s-1 (velocity per second) - need to convert s -> h
    r'cm[\s/]*(s|sec)[\-]?1?$': {
        'factor': 3600.0,  # cm/s -> cm/h (x3600)
        'time_to_hour': 1.0,
        'type': 'sapwood',
        'description': 'cm s-1 -> cm h-1 (x3600)'
    },
    
    # cm3/cm2*time or cm3 cm-2 with various time units
    r'cm[3]\s*[/\s]*cm[2]\s*[\*\s]*(10\s*min|20.*min|30.*min|min|h|hour|s|sec)': {
        'factor': 1.0,  # Base factor, time adjusted separately
        'time_patterns': {
            '10': 6.0,      # 10 min -> multiply by 6 for hourly
            '20': 3.0,      # 20 min -> multiply by 3 for hourly
            '30': 2.0,      # 30 min -> multiply by 2 for hourly
            'min': 60.0,    # per minute -> multiply by 60
            'h': 1.0,       # per hour
            'hour': 1.0,
            's': 3600.0,    # per second -> multiply by 3600
            'sec': 3600.0,
        },
        'type': 'sapwood',
        'description': 'cm3 cm-2 time-1'
    },
    
    # mm3/mm2 (same as mm for flux density, need to convert mm -> cm)
    # mm3 mm-2 s-1 = mm/s -> cm/h requires x3600 (s->h) x 0.1 (mm->cm) = x360
    r'mm[3]\s*mm[\-]?2\s*(sapwood\s*)?(s|sec|min|h|hour)?[\-]?1?': {
        'factor': 0.1,  # mm -> cm conversion (mm3/mm2 = mm, need cm)
        'time_patterns': {
            's': 3600.0,    # per second -> per hour
            'sec': 3600.0,
            'min': 60.0,
            'h': 1.0,
            'hour': 1.0,
        },
        'default_time': 's',  # Default to per-second if not specified
        'type': 'sapwood',
        'description': 'mm3 mm-2 time-1 -> cm3 cm-2 h-1'
    },
    
    # kg/hour/cm2 (mass flux density) - convert mass to volume
    r'kg[/\.\s]*(hour|h)[\-]?1?[/\.\s]*cm[\^2]': {
        'factor': 1000.0,  # kg/h/cm2 -> g/h/cm2 = cm3/h/cm2 (water density 1 g/cm3)
        'time_to_hour': 1.0,
        'type': 'sapwood',
        'description': 'kg h-1 cm-2 sapwood'
    },
    
    # g/hour/cm2 (mass flux density)
    r'g[/\.\s]*(hour|h)[\-]?1?[/\.\s]*cm[\^2]': {
        'factor': 1.0,  # g/h/cm2 = cm3/h/cm2 (water density 1 g/cm3)
        'time_to_hour': 1.0,
        'type': 'sapwood',
        'description': 'g h-1 cm-2 sapwood'
    },
    
    # =========================================================================
    # PLANT-LEVEL UNITS (total flow) -> convert to cm3 h-1
    # =========================================================================
    
    # kg/h or kg/hour or kg.h-1 or kg/h/tree (mass flow per plant)
    # Must check kg BEFORE g to avoid partial match
    r'kg[\./\s]*(h|hr|hour)[\-]?1?': {
        'factor': 1000.0,  # kg/h = 1000 cm3/h
        'time_to_hour': 1.0,
        'type': 'plant',
        'description': 'kg h-1 -> cm3 h-1 (x1000)'
    },
    
    # g/h or g/hour or g h-1 (mass flow per plant) - must NOT be preceded by 'k'
    r'(?<!k)g[\./\s]*(h|hr|hour)[\-]?1?': {
        'factor': 1.0,  # g/h = cm3/h (water density 1 g/cm3)
        'time_to_hour': 1.0,
        'type': 'plant',
        'description': 'g h-1 -> cm3 h-1'
    },
    
    # cm3/h or cm3.hr-1 (volume flow per plant)
    r'cm[3][\./\s]*(h|hr|hour)[\-]?1?': {
        'factor': 1.0,  # Already cm3/h
        'time_to_hour': 1.0,
        'type': 'plant',
        'description': 'cm3 h-1 (already target)'
    },
    
    # L/h or L/hour (volume flow)
    r'[Ll][\./\s]*(h|hr|hour)[\-]?1?': {
        'factor': 1000.0,  # L/h = 1000 cm3/h
        'time_to_hour': 1.0,
        'type': 'plant',
        'description': 'L h-1 -> cm3 h-1 (x1000)'
    },
    
    # mL/h or ml/hour (volume flow)
    r'm[Ll][\./\s]*(h|hr|hour)[\-]?1?': {
        'factor': 1.0,  # mL/h = cm3/h
        'time_to_hour': 1.0,
        'type': 'plant',
        'description': 'mL h-1 -> cm3 h-1'
    },
}


def parse_unit_string(unit_str: str) -> Tuple[float, str, str]:
    """
    Parse a unit string and return conversion factor, target unit, and description.
    
    Returns:
        (conversion_factor, target_unit, description)
    """
    if pd.isna(unit_str) or not unit_str:
        return 1.0, 'unknown', 'Unknown unit'
    
    # Normalize unicode characters to ASCII equivalents
    unicode_map = {
        '\u00b3': '3',   # superscript 3 (cube)
        '\u00b2': '2',   # superscript 2 (square)
        '\u207b': '-',   # superscript minus
        '\u00b9': '1',   # superscript 1
        '\u2212': '-',   # minus sign
        '\u2070': '0',   # superscript 0
    }
    unit_normalized = str(unit_str)
    for uchar, achar in unicode_map.items():
        unit_normalized = unit_normalized.replace(uchar, achar)
    
    unit_str_clean = unit_normalized.strip().lower()
    
    # Try each pattern
    for pattern, info in UNIT_PATTERNS.items():
        match = re.search(pattern, unit_str_clean, re.IGNORECASE)
        if match:
            factor = info['factor']
            
            # Handle time-dependent conversions
            if 'time_patterns' in info:
                time_factor = 1.0
                matched_time = False
                for time_key, time_mult in info['time_patterns'].items():
                    if time_key in unit_str_clean:
                        time_factor = time_mult
                        matched_time = True
                        break
                if not matched_time and 'default_time' in info:
                    time_factor = info['time_patterns'].get(info['default_time'], 1.0)
                factor *= time_factor
            
            target = TARGET_SAPWOOD_UNIT if info['type'] == 'sapwood' else TARGET_PLANT_UNIT
            return factor, target, info['description']
    
    # Special case handling for specific known formats
    if 'cm/hour' in unit_str_clean or 'cm/h' in unit_str_clean:
        return 1.0, TARGET_SAPWOOD_UNIT, 'cm/hour velocity'
    
    # Handle unicode and ASCII variants of cm3/cm2
    if 'cm3/cm2' in unit_str_clean or 'cm3 cm' in unit_str_clean or 'cm3cm' in unit_str_clean:
        # Check time unit
        if '10min' in unit_str_clean or '10 min' in unit_str_clean:
            return 6.0, TARGET_SAPWOOD_UNIT, 'cm3 cm-2 10min-1 -> hourly'
        elif '20' in unit_str_clean and 'min' in unit_str_clean:
            return 3.0, TARGET_SAPWOOD_UNIT, 'cm3 cm-2 20min-1 -> hourly'
        elif '30' in unit_str_clean and 'min' in unit_str_clean:
            return 2.0, TARGET_SAPWOOD_UNIT, 'cm3 cm-2 30min-1 -> hourly'
        else:
            return 1.0, TARGET_SAPWOOD_UNIT, 'cm3 cm-2 h-1'
    
    if 'kg' in unit_str_clean and 'cm' in unit_str_clean and '2' in unit_str_clean:
        # kg per area per time
        return 1000.0, TARGET_SAPWOOD_UNIT, 'kg cm-2 time-1'
    
    if 'mm3' in unit_str_clean or 'mm-2' in unit_str_clean:
        # mm3/mm2 = mm, need to convert mm -> cm (factor 0.1)
        # Then convert time to hourly
        if 's-1' in unit_str_clean or unit_str_clean.endswith('s'):
            return 360.0, TARGET_SAPWOOD_UNIT, 'mm3 mm-2 s-1 -> cm3 cm-2 h-1 (x360)'
        return 0.1, TARGET_SAPWOOD_UNIT, 'mm3 mm-2 -> cm3 cm-2 (x0.1)'
    
    print(f"    WARNING: Unknown unit format: '{unit_str}' - using factor=1.0")
    return 1.0, 'unknown', f'Unknown: {unit_str}'


def convert_sapflow_data(
    sapflow_df: pd.DataFrame,
    plant_md_df: pd.DataFrame,
    target_type: str = 'sapwood',
    site_code: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Convert sapflow data to target units.
    
    Args:
        sapflow_df: DataFrame with timestamp index and plant columns
        plant_md_df: DataFrame with plant metadata including pl_sap_units_orig
        target_type: 'sapwood' for cm3 cm-2 h-1, 'plant' for cm3 h-1
        site_code: Site identifier for special handling (e.g., DE-Har)
    
    Returns:
        (converted_sapflow_df, updated_plant_md_df, conversion_log)
    """
    converted_df = sapflow_df.copy()
    updated_md = plant_md_df.copy()
    
    target_unit = TARGET_SAPWOOD_UNIT if target_type == 'sapwood' else TARGET_PLANT_UNIT
    
    # Get the unique original unit from metadata (should be same for all plants at a site)
    orig_units = plant_md_df['pl_sap_units_orig'].dropna().unique()
    
    if len(orig_units) == 0:
        print("    WARNING: No original unit information found in metadata")
        return converted_df, updated_md, []
    
    # Use the first (most common) unit
    orig_unit = orig_units[0]
    factor, detected_target, desc = parse_unit_string(orig_unit)
    
    # Safe print for Windows terminal (ASCII only)
    orig_unit_safe = str(orig_unit).encode('ascii', 'replace').decode()
    print(f"    Unit: {orig_unit_safe}")
    
    # Get all numeric columns (excluding timestamp-like columns)
    data_cols = [col for col in converted_df.columns 
                 if col not in ['timestamp', 'TIMESTAMP', 'date', 'time', 'datetime']
                 and converted_df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    conversion_log = []
    
    # Special handling for DE-Har: mixed time frequencies
    # Before 2024-03-20: 20 min (factor 3.0), From 2024-03-20: 30 min (factor 2.0)
    if site_code and 'DE-Har' in site_code:
        print(f"    Special handling for DE-Har: mixed 20min/30min frequencies")
        cutoff_date = pd.Timestamp('2024-03-20')
        
        # Ensure index is datetime
        if not isinstance(converted_df.index, pd.DatetimeIndex):
            converted_df.index = pd.to_datetime(converted_df.index)
        
        # Create mask for before/after cutoff
        mask_before = converted_df.index < cutoff_date
        mask_after = converted_df.index >= cutoff_date
        
        factor_before = 3.0  # 20 min -> hourly
        factor_after = 2.0   # 30 min -> hourly
        
        n_before = mask_before.sum()
        n_after = mask_after.sum()
        
        print(f"    Before 2024-03-20 (20min): {n_before} rows x {factor_before}")
        print(f"    From 2024-03-20 (30min): {n_after} rows x {factor_after}")
        
        for col in data_cols:
            converted_df.loc[mask_before, col] = converted_df.loc[mask_before, col] * factor_before
            converted_df.loc[mask_after, col] = converted_df.loc[mask_after, col] * factor_after
            conversion_log.append({
                'pl_code': col,
                'orig_unit': orig_unit,
                'factor': f'{factor_before}/{factor_after}',
                'target_unit': target_unit,
                'description': 'DE-Har mixed: 20min before 2024-03-20, 30min after'
            })
        
        # Update metadata with note about mixed factors
        updated_md['pl_sap_units'] = target_unit
        updated_md['conversion_factor'] = f'{factor_before}/{factor_after} (20min/30min)'
        
        return converted_df, updated_md, conversion_log
    
    # Standard conversion for other sites
    print(f"    Conversion factor: {factor} ({desc})")
    
    # Apply conversion to ALL data columns
    if factor != 1.0:
        for col in data_cols:
            converted_df[col] = converted_df[col] * factor
            conversion_log.append({
                'pl_code': col,
                'orig_unit': orig_unit,
                'factor': factor,
                'target_unit': target_unit,
                'description': desc
            })
    else:
        # Still log the columns even if factor is 1.0
        for col in data_cols:
            conversion_log.append({
                'pl_code': col,
                'orig_unit': orig_unit,
                'factor': factor,
                'target_unit': target_unit,
                'description': desc
            })
    
    # Update metadata for all plants
    updated_md['pl_sap_units'] = target_unit
    updated_md['conversion_factor'] = factor
    
    return converted_df, updated_md, conversion_log


def process_site_folder(site_folder: Path, output_folder: Path):
    """
    Process all sapflow files in a site folder and convert units.
    """
    print(f"\nProcessing: {site_folder.name}")
    
    # Find sapflow data files
    sapflow_files = list(site_folder.glob('*_sapflow_*.csv'))
    
    for sapflow_file in sapflow_files:
        site_code = sapflow_file.stem.split('_sapflow_')[0]
        data_type = sapflow_file.stem.split('_sapflow_')[-1] if '_sapflow_' in sapflow_file.stem else 'sapwood'
        
        # Find corresponding plant metadata
        plant_md_file = site_folder / f"{site_code}_plant_md.csv"
        if not plant_md_file.exists():
            # Try with data type suffix
            plant_md_file = site_folder / f"{site_code}_{data_type}_plant_md.csv"
        
        if not plant_md_file.exists():
            print(f"  WARNING: No plant metadata found for {sapflow_file.name}")
            continue
        
        # Load data
        sapflow_df = pd.read_csv(sapflow_file, index_col=0, parse_dates=True)
        plant_md_df = pd.read_csv(plant_md_file)
        
        print(f"  Converting {sapflow_file.name}...")
        print(f"    Original units: {[str(u).encode('ascii', 'replace').decode() for u in plant_md_df['pl_sap_units_orig'].unique()]}")
        
        # Determine target type based on folder name or data type
        target_type = 'sapwood'  # Default to sapwood normalization
        
        # Convert (pass site_code for special handling like DE-Har)
        converted_df, updated_md, log = convert_sapflow_data(
            sapflow_df, plant_md_df, target_type, site_code=site_code
        )
        
        # Print conversion summary
        for entry in log[:3]:  # Show first 3
            factor_str = f"{entry['factor']:.4f}" if isinstance(entry['factor'], (int, float)) else str(entry['factor'])
            print(f"    {entry['pl_code']}: {entry['orig_unit']} x {factor_str} -> {entry['target_unit']}")
        if len(log) > 3:
            print(f"    ... and {len(log)-3} more plants")
        
        # Save to output folder
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Save converted sapflow data
        out_sapflow = output_folder / f"{site_code}_sapflow_converted.csv"
        converted_df.to_csv(out_sapflow)
        
        # Save updated metadata
        out_md = output_folder / f"{site_code}_plant_md_converted.csv"
        updated_md.to_csv(out_md, index=False)
        
        print(f"    [OK] Saved: {out_sapflow.name}")


def process_folder(base_path: Path, output_path: Path, target_type: str, target_unit_str: str):
    """
    Process all sapflow files in a folder and convert units.
    
    Args:
        base_path: Input folder path
        output_path: Output folder path  
        target_type: 'sapwood' or 'plant'
        target_unit_str: Target unit string for output filenames
    
    Returns:
        List of conversion summary dicts
    """
    if not base_path.exists():
        print(f"  Folder not found: {base_path}")
        return []
    
    # Find all sapflow files
    sapflow_files = list(base_path.glob('*_sapflow_*.csv'))
    
    if not sapflow_files:
        print(f"  No sapflow files found in {base_path.name}")
        return []
    
    print(f"\nFound {len(sapflow_files)} {target_type}-level file(s)")
    
    conversion_summary = []
    
    for sapflow_file in sorted(sapflow_files):
        site_code = sapflow_file.stem.split('_sapflow_')[0]
        
        # Find corresponding plant metadata  
        plant_md_patterns = [
            base_path / f"{site_code}_plant_md.csv",
            base_path / f"{sapflow_file.stem.replace('_sapflow_', '_')}_plant_md.csv",
        ]
        
        plant_md_file = None
        for pattern in plant_md_patterns:
            if pattern.exists():
                plant_md_file = pattern
                break
        
        # Also search recursively
        if plant_md_file is None:
            matches = list(base_path.rglob(f"{site_code}*_plant_md.csv"))
            if matches:
                plant_md_file = matches[0]
        
        if plant_md_file is None:
            print(f"\n  WARNING: No plant metadata for {sapflow_file.name}, skipping")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {site_code} ({target_type}-level)")
        print(f"  Sapflow: {sapflow_file.name}")
        print(f"  Metadata: {plant_md_file.name}")
        
        # Load data
        sapflow_df = pd.read_csv(sapflow_file, index_col=0, parse_dates=True)
        plant_md_df = pd.read_csv(plant_md_file)
        
        # Replace invalid values with NA (common placeholders: -9999, -999, -99, 9999, etc.)
        invalid_values = [-9999, -9999.0, -999, -999.0, -99, -99.0, ]
        n_invalid = 0
        for col in sapflow_df.columns:
            if sapflow_df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                mask = sapflow_df[col].isin(invalid_values)
                n_invalid += mask.sum()
                sapflow_df.loc[mask, col] = np.nan
        if n_invalid > 0:
            print(f"  Replaced {n_invalid} invalid values (-9999, etc.) with NA")
        
        # Show original units
        orig_units = plant_md_df['pl_sap_units_orig'].dropna().unique()
        print(f"  Original units: {[str(u).encode('ascii', 'replace').decode() for u in orig_units]}")
        
        # Convert - pass site_code for special handling (e.g., DE-Har mixed frequencies)
        converted_df, updated_md, log = convert_sapflow_data(
            sapflow_df, plant_md_df, target_type=target_type, site_code=site_code
        )
        
        # Show conversion details
        if log:
            factors_used = set(entry['factor'] for entry in log)
            print(f"  Conversion factors used: {factors_used}")
            
            # Sample before/after
            sample_col = log[0]['pl_code']
            if sample_col in sapflow_df.columns and sample_col in converted_df.columns:
                orig_mean = sapflow_df[sample_col].dropna().mean()
                conv_mean = converted_df[sample_col].dropna().mean()
                print(f"  Sample {sample_col}: {orig_mean:.6f} -> {conv_mean:.6f}")
        
        # Save
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save converted sapflow data (keep original filename pattern)
        out_sapflow = output_path / sapflow_file.name
        converted_df.to_csv(out_sapflow)
        
        # Save updated plant metadata
        out_md = output_path / plant_md_file.name
        updated_md.to_csv(out_md, index=False)
        
        # Copy all other site files (env_md, site_md, stand_md, env_data, etc.)
        copied_files = []
        for src_file in base_path.glob(f"{site_code}*"):
            if src_file.is_file():
                # Skip files we've already processed
                if src_file.name == sapflow_file.name or src_file.name == plant_md_file.name:
                    continue
                dst_file = output_path / src_file.name
                if not dst_file.exists():
                    shutil.copy2(src_file, dst_file)
                    copied_files.append(src_file.name)
        
        if copied_files:
            print(f"  Copied {len(copied_files)} additional files")
        
        print(f"  [OK] Saved: {out_sapflow.name}")
        
        conversion_summary.append({
            'site': site_code,
            'type': target_type,
            'orig_units': ', '.join(str(u) for u in orig_units),
            'n_plants': len(log),
            'factors': ', '.join(str(f) for f in set(e['factor'] for e in log))
        })
    
    # Copy any remaining files that weren't matched to a sapflow file
    all_copied = set(f.name for f in output_path.glob('*'))
    for src_file in base_path.glob('*'):
        if src_file.is_file() and src_file.name not in all_copied:
            dst_file = output_path / src_file.name
            shutil.copy2(src_file, dst_file)
            print(f"  Copied unmatched file: {src_file.name}")
    
    return conversion_summary


def main():
    """
    Main entry point for unit conversion.
    """
    print("=" * 70)
    print("SAPFLOW UNIT CONVERTER")
    print("=" * 70)
    print(f"Target sapwood unit: {TARGET_SAPWOOD_UNIT}")
    print(f"Target plant unit: {TARGET_PLANT_UNIT}")
    print("=" * 70)
    
    base_dir = Path(__file__).parent / "Sapflow_SAPFLUXNET_format"
    output_base = Path(__file__).parent / "Sapflow_SAPFLUXNET_format_unitcon"
    
    all_summaries = []
    
    # Process sapwood-level data
    print("\n" + "=" * 70)
    print("SAPWOOD-LEVEL DATA (sap flux density)")
    print("=" * 70)
    sapwood_path = base_dir / "sapwood"
    sapwood_output = output_base / "sapwood"
    summaries = process_folder(sapwood_path, sapwood_output, 'sapwood', 'cm3_cm-2_h-1')
    all_summaries.extend(summaries)
    
    # Process plant-level data
    print("\n" + "=" * 70)
    print("PLANT-LEVEL DATA (sap flow rate)")
    print("=" * 70)
    plant_path = base_dir / "plant"
    plant_output = output_base / "plant"
    summaries = process_folder(plant_path, plant_output, 'plant', 'cm3_h-1')
    all_summaries.extend(summaries)
    
    # Print summary
    print("\n" + "=" * 70)
    print("CONVERSION SUMMARY")
    print("=" * 70)
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        # Sanitize for Windows terminal
        summary_df['orig_units'] = summary_df['orig_units'].apply(
            lambda x: str(x).encode('ascii', 'replace').decode()
        )
        print(summary_df.to_string(index=False))
    else:
        print("No files converted.")
    print("=" * 70)
    print(f"\nConverted files saved to: {output_base}")
    print(f"  Sapwood target: {TARGET_SAPWOOD_UNIT}")
    print(f"  Plant target: {TARGET_PLANT_UNIT}")


if __name__ == "__main__":
    main()

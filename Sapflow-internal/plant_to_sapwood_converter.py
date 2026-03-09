"""
Plant to Sapwood Converter
==========================
Converts plant-level sapflow data (cm³ h⁻¹) to sapwood-level (cm³ cm⁻² h⁻¹)
by dividing by sapwood area for sites where sapwood area is provided.

Formula: Sap flux density = Sap flow rate / Sapwood area
         cm³ cm⁻² h⁻¹ = cm³ h⁻¹ / cm²
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil

# Paths
PLANT_INPUT = Path(r'e:\OneDrive - Universität Münster\Sap_velocity_project\global-sap-velocity\Sapflow-internal\Sapflow_SAPFLUXNET_format_unitcon\plant')
SAPWOOD_OUTPUT = Path(r'e:\OneDrive - Universität Münster\Sap_velocity_project\global-sap-velocity\Sapflow-internal\Sapflow_SAPFLUXNET_format_unitcon\sapwood')

def convert_plant_to_sapwood(site_code: str):
    """
    Convert plant-level data to sapwood-level by dividing by sapwood area.
    
    Args:
        site_code: Site code to process
    
    Returns:
        True if conversion successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Converting: {site_code} (plant -> sapwood)")
    
    # Find files
    sapflow_file = PLANT_INPUT / f"{site_code}_sapflow_plant.csv"
    plant_md_file = PLANT_INPUT / f"{site_code}_plant_md.csv"
    
    if not sapflow_file.exists():
        print(f"  ERROR: Sapflow file not found: {sapflow_file}")
        return False
    
    if not plant_md_file.exists():
        print(f"  ERROR: Metadata file not found: {plant_md_file}")
        return False
    
    # Load data
    sapflow_df = pd.read_csv(sapflow_file, index_col=0, parse_dates=True)
    plant_md_df = pd.read_csv(plant_md_file)
    
    print(f"  Sapflow shape: {sapflow_df.shape}")
    print(f"  Plants: {len(plant_md_df)}")
    
    # Check for sapwood area
    if 'pl_sapw_area' not in plant_md_df.columns:
        print(f"  SKIP: No pl_sapw_area column in metadata")
        return False
    
    # Check how many plants have valid sapwood area
    valid_sw = plant_md_df['pl_sapw_area'].notna()
    n_valid = valid_sw.sum()
    n_total = len(plant_md_df)
    
    if n_valid == 0:
        print(f"  SKIP: No valid sapwood area values")
        return False
    
    print(f"  Sapwood area available: {n_valid}/{n_total} plants")
    
    # Convert each column
    converted_df = sapflow_df.copy()
    conversion_log = []
    mismatches = []
    missing_sapwood_area = []
    
    for _, row in plant_md_df.iterrows():
        pl_code = row['pl_code']
        sw_area = row['pl_sapw_area']
        
        if pl_code not in converted_df.columns:
            mismatches.append(pl_code)
            continue
        
        if pd.isna(sw_area) or sw_area <= 0:
            missing_sapwood_area.append((pl_code, sw_area))
            continue
    
    # Report mismatches as errors
    if mismatches:
        print(f"\n  ERROR: {len(mismatches)} pl_code(s) in metadata NOT FOUND in sapflow columns!")
        print(f"    Metadata pl_codes not matched: {mismatches}")
        print(f"    Sapflow columns available: {converted_df.columns.tolist()}")
        raise ValueError(f"Metadata pl_code mismatch: {mismatches}")
    
    # Report missing sapwood area as warnings
    if missing_sapwood_area:
        print(f"\n  WARNING: {len(missing_sapwood_area)} plant(s) have no valid sapwood area:")
        for pl_code, sw_area in missing_sapwood_area:
            print(f"    - {pl_code}: sapwood_area = {sw_area}")
    
    # Now do the actual conversion
    for _, row in plant_md_df.iterrows():
        pl_code = row['pl_code']
        sw_area = row['pl_sapw_area']
        
        if pl_code not in converted_df.columns:
            continue
        
        if pd.isna(sw_area) or sw_area <= 0:
            continue
        
        # Convert: cm³ h⁻¹ / cm² = cm³ cm⁻² h⁻¹ (= cm h⁻¹)
        orig_mean = converted_df[pl_code].dropna().mean()
        converted_df[pl_code] = converted_df[pl_code] / sw_area
        conv_mean = converted_df[pl_code].dropna().mean()
        
        conversion_log.append({
            'pl_code': pl_code,
            'sapw_area_cm2': sw_area,
            'orig_mean': orig_mean,
            'conv_mean': conv_mean
        })
    
    if not conversion_log:
        print(f"  SKIP: No columns could be converted")
        return False
    
    # Show sample conversions
    print(f"\n  Sample conversions:")
    for entry in conversion_log[:3]:
        print(f"    {entry['pl_code']}: {entry['orig_mean']:.4f} cm³/h / {entry['sapw_area_cm2']:.2f} cm² = {entry['conv_mean']:.6f} cm³ cm⁻² h⁻¹")
    
    # Update metadata
    updated_md = plant_md_df.copy()
    updated_md['pl_sap_units'] = 'cm3 cm-2 h-1'
    updated_md['pl_sap_units_orig'] = plant_md_df.get('pl_sap_units_orig', 'cm3 h-1 (plant-level)')
    updated_md['conversion_note'] = 'Converted from plant-level by dividing by pl_sapw_area'
    
    # Save to sapwood folder
    SAPWOOD_OUTPUT.mkdir(parents=True, exist_ok=True)
    
    # Save with _sapwood suffix
    out_sapflow = SAPWOOD_OUTPUT / f"{site_code}_sapflow_sapwood.csv"
    out_md = SAPWOOD_OUTPUT / f"{site_code}_plant_md.csv"
    
    converted_df.to_csv(out_sapflow)
    updated_md.to_csv(out_md, index=False)
    
    print(f"\n  [OK] Saved: {out_sapflow.name}")
    print(f"  [OK] Saved: {out_md.name}")
    
    # Copy other site files
    for src_file in PLANT_INPUT.glob(f"{site_code}_*.csv"):
        if 'sapflow' in src_file.name or 'plant_md' in src_file.name:
            continue  # Already handled
        dst_file = SAPWOOD_OUTPUT / src_file.name
        shutil.copy2(src_file, dst_file)
        print(f"  Copied: {src_file.name}")
    
    return True


def main():
    print("="*70)
    print("PLANT TO SAPWOOD CONVERTER")
    print("="*70)
    print(f"Input: {PLANT_INPUT}")
    print(f"Output: {SAPWOOD_OUTPUT}")
    print("Formula: cm³ cm⁻² h⁻¹ = cm³ h⁻¹ / sapwood_area_cm²")
    print("="*70)
    
    # Sites with sapwood area data
    sites_with_sw_area = ['AT_Mmg', 'FR-BIL', 'SE-Nor', 'SE-Sgr']
    
    converted = []
    skipped = []
    
    for site in sites_with_sw_area:
        if convert_plant_to_sapwood(site):
            converted.append(site)
        else:
            skipped.append(site)
    
    print("\n" + "="*70)
    print("CONVERSION SUMMARY")
    print("="*70)
    print(f"Converted: {converted}")
    print(f"Skipped: {skipped}")
    print("="*70)


if __name__ == '__main__':
    main()

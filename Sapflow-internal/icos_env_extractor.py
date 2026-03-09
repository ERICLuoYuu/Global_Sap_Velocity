#!/usr/bin/env python3
"""
ICOS Environmental Data Extractor for Sapflow Sites

This script extracts environmental data from the ICOS Carbon Portal API
for sites with sapflow measurements. It matches the time periods from
the sapflow data and outputs environmental data in a consistent format.

Required package: pip install icoscp

Author: Generated for Sapflow Project
Date: 2026-02-05
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Site name mapping: your site names -> ICOS station IDs
SITE_MAPPING = {
    'AT_Mmg': None,  # Not in ICOS
    'CH-Dav_daily': 'CH-Dav',
    'CH-Dav_halfhourly': 'CH-Dav',
    'CH-Lae_daily': None,  # CH-Lae (Laegeren) might not be in ICOS ecosystem list
    'CH-Lae_halfhourly': None,
    'DE-Har': 'DE-Har',
    'DE-HoH': 'DE-HoH',
    'ES-Abr': None,  # Not in ICOS
    'ES-LM1': 'ES-LMa',  # Las Majadas
    'ES-LM2': 'ES-LMa',
    'ES-LMa': 'ES-LMa',
    'FI-Hyy': 'FI-Hyy',
    'FR-BIL': 'FR-Bil',
    'IT-CP2_sapwood': 'IT-Cp2',
    'SE-Nor': 'SE-Nor',
    'SE-Sgr': None,  # Not in ICOS
}

# Environmental variables to extract (ICOS FLUXNET naming)
ENV_VARIABLES = {
    'TA_F': 'ta',           # Air temperature (°C)
    'TA_F_QC': 'ta_qc',     # Quality flag
    'VPD_F': 'vpd',         # Vapor pressure deficit (hPa)
    'VPD_F_QC': 'vpd_qc',
    'SW_IN_F': 'sw_in',     # Shortwave incoming radiation (W/m²)
    'SW_IN_F_QC': 'sw_in_qc',
    'PPFD_IN': 'ppfd',      # Photosynthetic photon flux density (µmol/m²/s)
    'PA_F': 'pa',           # Atmospheric pressure (kPa)
    'RH': 'rh',             # Relative humidity (%)
    'WS_F': 'ws',           # Wind speed (m/s)
    'P_F': 'precip',        # Precipitation (mm)
    'TS_F_MDS_1': 'ts',     # Soil temperature (°C)
    'SWC_F_MDS_1': 'swc',   # Soil water content (%)
}


class ICOSExtractor:
    """Extract environmental data from ICOS Carbon Portal"""
    
    def __init__(self, sapflow_dir: str, output_dir: str):
        """
        Initialize the extractor.
        
        Parameters
        ----------
        sapflow_dir : str
            Directory containing sapflow_sapwood.csv files
        output_dir : str
            Directory to save extracted environmental data
        """
        self.sapflow_dir = Path(sapflow_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import ICOS library
        try:
            from icoscp.station import station
            from icoscp.dobj import Dobj
            self.station = station
            self.Dobj = Dobj
            logger.info("ICOS library loaded successfully")
        except ImportError:
            logger.error("icoscp not installed. Run: pip install icoscp")
            raise
    
    def get_sapflow_time_range(self, site_name: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Get the time range from a sapflow file.
        
        Parameters
        ----------
        site_name : str
            Site name (e.g., 'CH-Dav_halfhourly')
        
        Returns
        -------
        Tuple[datetime, datetime]
            Start and end timestamps
        """
        # Find matching sapflow file
        pattern = f"{site_name}_sapflow_sapwood.csv"
        files = list(self.sapflow_dir.glob(pattern))
        
        if not files:
            logger.warning(f"No sapflow file found for {site_name}")
            return None, None
        
        df = pd.read_csv(files[0], parse_dates=['timestamp'])
        start = df['timestamp'].min()
        end = df['timestamp'].max()
        
        logger.info(f"  Sapflow time range: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
        return start, end
    
    def get_icos_data(self, icos_id: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Download environmental data from ICOS for a site.
        
        Parameters
        ----------
        icos_id : str
            ICOS station ID (e.g., 'CH-Dav')
        start_date : datetime
            Start of period
        end_date : datetime
            End of period
        
        Returns
        -------
        pd.DataFrame or None
            Environmental data with timestamp index
        """
        try:
            # Get station
            st = self.station.get(icos_id)
            if st is None or not st.valid:
                logger.error(f"Station {icos_id} not found or invalid")
                return None
            
            logger.info(f"  Station: {st.name} (lat={st.lat:.4f}, lon={st.lon:.4f})")
            
            # Get available data products
            data_catalog = st.data()
            if data_catalog is None or len(data_catalog) == 0:
                logger.error(f"No data products available for {icos_id}")
                return None
            
            # Try to find half-hourly Fluxnet product first, then fallback to others
            product_priorities = [
                'Fluxnet.*half',
                'ETC L2 Fluxnet',
                'ETC L2 Meteo',
                'Fluxnet Product',
                'ETC L2 ARCHIVE'
            ]
            
            dobj_pid = None
            product_name = None
            
            for pattern in product_priorities:
                matches = data_catalog[data_catalog['specLabel'].str.contains(pattern, case=False, regex=True)]
                if len(matches) > 0:
                    dobj_pid = matches.iloc[0]['dobj']
                    product_name = matches.iloc[0]['specLabel']
                    break
            
            if dobj_pid is None:
                logger.error(f"No suitable data product found for {icos_id}")
                logger.info(f"  Available products: {list(data_catalog['specLabel'])}")
                return None
            
            logger.info(f"  Using product: {product_name}")
            
            # Download data
            dobj = self.Dobj(dobj_pid)
            
            # Get dataframe
            df = dobj.data
            
            if df is None or len(df) == 0:
                logger.error(f"No data returned for {icos_id}")
                return None
            
            logger.info(f"  Downloaded {len(df)} rows, columns: {len(df.columns)}")
            
            # Process timestamp
            if 'TIMESTAMP' in df.columns:
                df['timestamp'] = pd.to_datetime(df['TIMESTAMP'])
            elif 'TIMESTAMP_START' in df.columns:
                df['timestamp'] = pd.to_datetime(df['TIMESTAMP_START'])
            else:
                # Try to find any timestamp column
                ts_cols = [c for c in df.columns if 'TIME' in c.upper() or 'DATE' in c.upper()]
                if ts_cols:
                    df['timestamp'] = pd.to_datetime(df[ts_cols[0]])
                else:
                    logger.error(f"No timestamp column found in {icos_id} data")
                    return None
            
            # Filter to requested time range (handle timezone-aware timestamps)
            # Make timestamps timezone-naive for comparison
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)
            
            # Convert start/end to pandas timestamps and make timezone-naive
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            if start_ts.tz is not None:
                start_ts = start_ts.tz_localize(None)
            if end_ts.tz is not None:
                end_ts = end_ts.tz_localize(None)
            
            df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)]
            logger.info(f"  Filtered to {len(df)} rows in requested time range")
            
            if len(df) == 0:
                logger.warning(f"No data in time range for {icos_id}")
                return None
            
            # Rename columns to standard names
            rename_dict = {}
            for icos_name, std_name in ENV_VARIABLES.items():
                if icos_name in df.columns:
                    rename_dict[icos_name] = std_name
                else:
                    # Try without _F suffix
                    alt_name = icos_name.replace('_F', '').replace('_MDS_1', '')
                    if alt_name in df.columns:
                        rename_dict[alt_name] = std_name
            
            df = df.rename(columns=rename_dict)
            
            # Keep only timestamp and renamed environmental columns
            keep_cols = ['timestamp'] + [c for c in df.columns if c in ENV_VARIABLES.values()]
            df = df[[c for c in keep_cols if c in df.columns]]
            
            # Set index
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading data for {icos_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_site(self, site_name: str) -> bool:
        """
        Process a single site - extract ICOS data matching sapflow time range.
        
        Parameters
        ----------
        site_name : str
            Site name from sapflow files
        
        Returns
        -------
        bool
            True if successful
        """
        logger.info(f"\nProcessing: {site_name}")
        
        # Check if ICOS mapping exists
        icos_id = SITE_MAPPING.get(site_name)
        if icos_id is None:
            logger.warning(f"  No ICOS mapping for {site_name} - skipping")
            return False
        
        logger.info(f"  ICOS station: {icos_id}")
        
        # Get sapflow time range
        start_date, end_date = self.get_sapflow_time_range(site_name)
        if start_date is None:
            return False
        
        # Download ICOS data
        env_df = self.get_icos_data(icos_id, start_date, end_date)
        if env_df is None:
            return False
        
        # Save to output
        output_file = self.output_dir / f"{site_name}_env_icos.csv"
        env_df.to_csv(output_file)
        logger.info(f"  Saved: {output_file.name}")
        logger.info(f"  Variables: {list(env_df.columns)}")
        logger.info(f"  Time range: {env_df.index.min()} to {env_df.index.max()}")
        
        return True
    
    def process_all_sites(self):
        """Process all sites with ICOS mappings."""
        # Find all sapflow files
        sapflow_files = list(self.sapflow_dir.glob("*_sapflow_sapwood.csv"))
        site_names = [f.stem.replace('_sapflow_sapwood', '') for f in sapflow_files]
        
        logger.info(f"Found {len(site_names)} sapflow sites")
        
        success_count = 0
        failed_sites = []
        skipped_sites = []
        
        for site_name in sorted(site_names):
            if SITE_MAPPING.get(site_name) is None:
                skipped_sites.append(site_name)
                logger.info(f"\nSkipping {site_name} - no ICOS mapping")
                continue
            
            if self.process_site(site_name):
                success_count += 1
            else:
                failed_sites.append(site_name)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("EXTRACTION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total sites: {len(site_names)}")
        logger.info(f"Successfully extracted: {success_count}")
        logger.info(f"Skipped (no ICOS mapping): {len(skipped_sites)}")
        if skipped_sites:
            logger.info(f"  {skipped_sites}")
        logger.info(f"Failed: {len(failed_sites)}")
        if failed_sites:
            logger.info(f"  {failed_sites}")
        logger.info(f"\nOutput directory: {self.output_dir}")


def main():
    """Main entry point."""
    # Configuration
    SAPFLOW_DIR = Path(__file__).parent / "Sapflow_SAPFLUXNET_format_unitcon" / "sapwood"
    OUTPUT_DIR = Path(__file__).parent / "Sapflow_SAPFLUXNET_format_unitcon" / "env_icos"
    
    logger.info("="*60)
    logger.info("ICOS Environmental Data Extractor")
    logger.info("="*60)
    logger.info(f"Sapflow directory: {SAPFLOW_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Run extraction
    extractor = ICOSExtractor(
        sapflow_dir=str(SAPFLOW_DIR),
        output_dir=str(OUTPUT_DIR)
    )
    
    extractor.process_all_sites()


if __name__ == "__main__":
    main()

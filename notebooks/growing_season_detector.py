"""
Growing Season Detection Module

This module implements scientifically validated methods for determining growing season
from environmental and remote sensing data.

Methods implemented:
1. Growing Season Index (GSI) - Jolly et al. (2005) Global Change Biology
2. Thermal Growing Season - Based on WMO definition and ecological literature
3. Phenology-based using LAI dynamics - Following MODIS phenology algorithms
4. Soil Moisture Constrained Growing Season - For water-limited ecosystems
5. Combined Multi-factor approach

References:
- Jolly, W. M., et al. (2005). A generalized, bioclimatic index to predict foliar 
  phenology in response to climate. Global Change Biology, 11(4), 619-632.
- Nemani, R. R., et al. (2003). Climate-driven increases in global terrestrial net 
  primary production from 1982 to 1999. Science, 300(5625), 1560-1563.
- Richardson, A. D., et al. (2010). Influence of spring and autumn phenological 
  transitions on forest ecosystem productivity. Phil. Trans. R. Soc. B, 365, 3227-3246.
- Xia, J., et al. (2015). Joint control of terrestrial gross primary productivity 
  by plant phenology and physiology. PNAS, 112(9), 2788-2793.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Union
from scipy import signal
from scipy.ndimage import uniform_filter1d


class GrowingSeasonDetector:
    """
    A class for detecting growing season using various scientifically validated methods.
    """
    
    def __init__(self, df: pd.DataFrame, latitude: float, timestamp_col: str = 'TIMESTAMP'):
        """
        Initialize the detector with data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing environmental variables
        latitude : float
            Site latitude in degrees
        timestamp_col : str
            Name of the timestamp column
        """
        self.df = df.copy()
        self.latitude = latitude
        self.timestamp_col = timestamp_col
        
        # Ensure timestamp is datetime
        if timestamp_col in self.df.columns:
            self.df[timestamp_col] = pd.to_datetime(self.df[timestamp_col])
            self.df = self.df.set_index(timestamp_col)
        
        # Calculate day of year and photoperiod if not present
        self._calculate_temporal_variables()
    
    def _calculate_temporal_variables(self):
        """Calculate day of year and photoperiod."""
        self.df['doy'] = self.df.index.dayofyear
        self.df['photoperiod'] = self._calculate_photoperiod(
            self.df['doy'].values, 
            self.latitude
        )
    
    @staticmethod
    def _calculate_photoperiod(doy: np.ndarray, latitude: float) -> np.ndarray:
        """
        Calculate photoperiod (day length) in hours.
        
        Based on the CBM model (Forsythe et al., 1995).
        Forsythe, W.C., et al. (1995). A model comparison for daylength as a 
        function of latitude and day of year. Ecological Modelling, 80(1), 87-95.
        
        Parameters:
        -----------
        doy : np.ndarray
            Day of year (1-365/366)
        latitude : float
            Latitude in degrees
            
        Returns:
        --------
        np.ndarray
            Photoperiod in hours
        """
        lat_rad = np.radians(latitude)
        
        # Solar declination angle
        # Using more accurate formula from Forsythe et al.
        P = np.arcsin(0.39795 * np.cos(0.2163108 + 2 * np.arctan(0.9671396 * np.tan(0.00860 * (doy - 186)))))
        
        # Day length calculation
        # p = 0.8333 is the angle of the sun below horizon for sunrise/sunset (accounting for refraction)
        p = 0.8333
        
        numerator = np.sin(np.radians(p)) + np.sin(lat_rad) * np.sin(P)
        denominator = np.cos(lat_rad) * np.cos(P)
        
        # Clamp values to avoid domain errors
        ratio = np.clip(numerator / denominator, -1, 1)
        
        # Calculate day length
        day_length = 24 - (24 / np.pi) * np.arccos(ratio)
        
        return day_length
    
    def _scalar_to_index(self, value: float, vmin: float, vmax: float) -> float:
        """
        Convert a scalar value to an index between 0 and 1.
        Used in GSI calculation.
        
        Parameters:
        -----------
        value : float
            Input value
        vmin : float
            Minimum threshold (index = 0 below this)
        vmax : float
            Maximum threshold (index = 1 above this)
            
        Returns:
        --------
        float
            Index value between 0 and 1
        """
        if value <= vmin:
            return 0.0
        elif value >= vmax:
            return 1.0
        else:
            return (value - vmin) / (vmax - vmin)
    
    def calculate_gsi(self, 
                      tmin_col: str = 'ta',
                      vpd_col: str = 'vpd',
                      tmin_min: float = -2.0,
                      tmin_max: float = 5.0,
                      vpd_min: float = 0.9,  # kPa
                      vpd_max: float = 4.1,  # kPa
                      photo_min: float = 10.0,  # hours
                      photo_max: float = 11.0,  # hours
                      use_daily: bool = True) -> pd.Series:
        """
        Calculate Growing Season Index (GSI) following Jolly et al. (2005).
        
        GSI = iTmin × iVPD × iPhoto
        
        Where each index (i) is scaled 0-1 based on limiting thresholds.
        
        Parameters:
        -----------
        tmin_col : str
            Column name for minimum temperature (°C)
        vpd_col : str
            Column name for VPD (kPa)
        tmin_min, tmin_max : float
            Temperature thresholds for scaling (default: -2 to 5°C)
        vpd_min, vpd_max : float
            VPD thresholds for scaling (default: 0.9 to 4.1 kPa)
            Note: VPD relationship is inverted (high VPD = low index)
        photo_min, photo_max : float
            Photoperiod thresholds in hours (default: 10 to 11 hours)
        use_daily : bool
            If True, calculate daily values first then apply GSI
            
        Returns:
        --------
        pd.Series
            GSI values (0-1), where higher values indicate favorable growing conditions
        """
        df_calc = self.df.copy()
        
        if use_daily:
            # Resample to daily - use minimum temperature for tmin
            daily_data = df_calc.resample('D').agg({
                tmin_col: 'min',  # Daily minimum temperature
                vpd_col: 'max',   # Daily maximum VPD (most limiting)
                'photoperiod': 'mean'
            })
        else:
            daily_data = df_calc[[tmin_col, vpd_col, 'photoperiod']]
        
        # Calculate individual indices
        iTmin = daily_data[tmin_col].apply(
            lambda x: self._scalar_to_index(x, tmin_min, tmin_max)
        )
        
        # VPD is inverted - high VPD limits growth
        iVPD = daily_data[vpd_col].apply(
            lambda x: 1 - self._scalar_to_index(x, vpd_min, vpd_max)
        )
        
        iPhoto = daily_data['photoperiod'].apply(
            lambda x: self._scalar_to_index(x, photo_min, photo_max)
        )
        
        # Calculate GSI as product of indices
        gsi = iTmin * iVPD * iPhoto
        
        # Apply 21-day moving average as per Jolly et al.
        gsi_smoothed = gsi.rolling(window=21, center=True, min_periods=1).mean()
        
        return gsi_smoothed
    
    def calculate_thermal_growing_season(self,
                                         temp_col: str = 'ta',
                                         threshold: float = 5.0,
                                         consecutive_days: int = 5,
                                         method: str = 'wmo') -> pd.Series:
        """
        Calculate thermal growing season based on temperature thresholds.
        
        Methods:
        - 'wmo': WMO definition - period between first and last occurrence of 
                 5 consecutive days with Tmean > 5°C
        - 'gdd': Growing Degree Days based - cumulative temperature above base
        - 'smooth': Based on smoothed temperature crossing threshold
        
        Parameters:
        -----------
        temp_col : str
            Column name for temperature
        threshold : float
            Base temperature threshold (default: 5°C)
        consecutive_days : int
            Number of consecutive days required (default: 5)
        method : str
            Method to use ('wmo', 'gdd', 'smooth')
            
        Returns:
        --------
        pd.Series
            Boolean series indicating growing season (True) or not (False)
        """
        # Get daily mean temperature
        daily_temp = self.df[temp_col].resample('D').mean()
        
        if method == 'wmo':
            # WMO thermal growing season definition
            above_threshold = daily_temp > threshold
            
            # Find runs of consecutive days above threshold
            runs = above_threshold.astype(int).groupby(
                (~above_threshold).cumsum()
            ).cumsum()
            
            # Mark periods with enough consecutive days
            growing = runs >= consecutive_days
            
            # For each year, find first and last growing period
            result = pd.Series(False, index=daily_temp.index)
            
            for year in daily_temp.index.year.unique():
                year_mask = daily_temp.index.year == year
                year_growing = growing[year_mask]
                
                if year_growing.any():
                    first_idx = year_growing.idxmax()
                    last_idx = year_growing[::-1].idxmax()
                    result.loc[first_idx:last_idx] = True
            
            return result
        
        elif method == 'gdd':
            # Growing Degree Days approach
            gdd = (daily_temp - threshold).clip(lower=0).cumsum()
            
            # Growing season when GDD is accumulating significantly
            # Reset each year
            result = pd.Series(False, index=daily_temp.index)
            
            for year in daily_temp.index.year.unique():
                year_mask = daily_temp.index.year == year
                year_gdd = gdd[year_mask] - gdd[year_mask].iloc[0]
                
                # Growing season: from when GDD starts accumulating 
                # (>10% of annual total) to when it plateaus (>90%)
                if year_gdd.max() > 0:
                    annual_gdd = year_gdd.max()
                    start_mask = year_gdd >= 0.05 * annual_gdd
                    end_mask = year_gdd <= 0.95 * annual_gdd
                    result.loc[year_mask] = start_mask & end_mask
            
            return result
        
        elif method == 'smooth':
            # Smoothed temperature approach
            # Apply 10-day moving average
            temp_smooth = daily_temp.rolling(window=10, center=True, min_periods=1).mean()
            return temp_smooth > threshold
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def calculate_phenology_growing_season(self,
                                           lai_col: str = 'lai',
                                           method: str = 'threshold',
                                           threshold_pct: float = 0.2,
                                           smooth_window: int = 15) -> pd.Series:
        """
        Calculate growing season based on LAI phenology.
        
        Based on methods used in MODIS Land Cover Dynamics and 
        phenology literature.
        
        Methods:
        - 'threshold': Growing season when LAI > X% of annual amplitude
        - 'derivative': Based on rate of change of LAI
        - 'midpoint': Based on half-maximum of seasonal LAI curve
        
        Parameters:
        -----------
        lai_col : str
            Column name for LAI
        method : str
            Method to use ('threshold', 'derivative', 'midpoint')
        threshold_pct : float
            Threshold as fraction of annual amplitude (default: 0.2 = 20%)
        smooth_window : int
            Window for smoothing LAI data (days)
            
        Returns:
        --------
        pd.Series
            Boolean series indicating growing season
        """
        if lai_col not in self.df.columns:
            raise ValueError(f"LAI column '{lai_col}' not found in dataframe")
        
        # Get daily LAI
        daily_lai = self.df[lai_col].resample('D').mean()
        
        # Smooth LAI to reduce noise
        lai_smooth = daily_lai.rolling(
            window=smooth_window, center=True, min_periods=1
        ).mean()
        
        result = pd.Series(False, index=daily_lai.index)
        
        for year in daily_lai.index.year.unique():
            year_mask = daily_lai.index.year == year
            year_lai = lai_smooth[year_mask]
            
            if year_lai.isna().all():
                continue
            
            lai_min = year_lai.min()
            lai_max = year_lai.max()
            lai_amplitude = lai_max - lai_min
            
            if lai_amplitude <= 0:
                continue
            
            if method == 'threshold':
                # Growing season when LAI > baseline + threshold% of amplitude
                threshold_value = lai_min + threshold_pct * lai_amplitude
                result.loc[year_mask] = year_lai > threshold_value
            
            elif method == 'derivative':
                # Growing season based on positive rate of change
                lai_derivative = year_lai.diff()
                # Smooth derivative
                deriv_smooth = lai_derivative.rolling(
                    window=smooth_window, center=True, min_periods=1
                ).mean()
                
                # Growing season: from max increase rate to max decrease rate
                if not deriv_smooth.isna().all():
                    max_increase_idx = deriv_smooth.idxmax()
                    max_decrease_idx = deriv_smooth.idxmin()
                    
                    if max_increase_idx < max_decrease_idx:
                        result.loc[max_increase_idx:max_decrease_idx] = True
            
            elif method == 'midpoint':
                # Half-maximum method (common in phenology studies)
                midpoint = lai_min + 0.5 * lai_amplitude
                result.loc[year_mask] = year_lai > midpoint
            
            else:
                raise ValueError(f"Unknown method: {method}")
        
        return result
    
    def calculate_moisture_constrained_season(self,
                                              swc_col: str = 'swc_shallow',
                                              precip_col: str = 'precip',
                                              pet_col: Optional[str] = None,
                                              swc_threshold: float = 0.1,
                                              aridity_threshold: float = 0.5) -> pd.Series:
        """
        Calculate growing season considering water limitations.
        
        Important for semi-arid and arid ecosystems where water, not 
        temperature, limits growth.
        
        Parameters:
        -----------
        swc_col : str
            Column name for soil water content
        precip_col : str
            Column name for precipitation
        pet_col : str, optional
            Column name for potential evapotranspiration (for aridity index)
        swc_threshold : float
            Minimum soil water content threshold (volumetric, m³/m³)
        aridity_threshold : float
            Minimum P/PET ratio for growing conditions
            
        Returns:
        --------
        pd.Series
            Boolean series indicating moisture-favorable growing conditions
        """
        # Daily aggregation
        agg_dict = {}
        if swc_col in self.df.columns:
            agg_dict[swc_col] = 'mean'
        if precip_col in self.df.columns:
            agg_dict[precip_col] = 'sum'
        if pet_col and pet_col in self.df.columns:
            agg_dict[pet_col] = 'sum'
        
        if not agg_dict:
            raise ValueError("No moisture-related columns found")
        
        daily_data = self.df.resample('D').agg(agg_dict)
        
        # Calculate moisture conditions
        conditions = pd.Series(True, index=daily_data.index)
        
        if swc_col in daily_data.columns:
            # 30-day rolling mean of soil moisture
            swc_smooth = daily_data[swc_col].rolling(
                window=30, center=True, min_periods=1
            ).mean()
            conditions &= swc_smooth > swc_threshold
        
        if pet_col and pet_col in daily_data.columns and precip_col in daily_data.columns:
            # Calculate running aridity index (P/PET over 30 days)
            precip_sum = daily_data[precip_col].rolling(window=30, min_periods=1).sum()
            pet_sum = daily_data[pet_col].rolling(window=30, min_periods=1).sum()
            
            # Avoid division by zero
            aridity_index = precip_sum / pet_sum.replace(0, np.nan)
            aridity_index = aridity_index.fillna(0)
            
            conditions &= aridity_index > aridity_threshold
        
        return conditions
    
    def calculate_combined_growing_season(self,
                                          temp_col: str = 'ta',
                                          vpd_col: str = 'vpd',
                                          lai_col: Optional[str] = 'lai',
                                          swc_col: Optional[str] = 'swc_shallow',
                                          gsi_threshold: float = 0.5,
                                          use_moisture: bool = True,
                                          use_phenology: bool = True) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Calculate growing season using a combined multi-factor approach.
        
        This integrates:
        1. GSI (temperature, VPD, photoperiod)
        2. Phenology (LAI dynamics) if available
        3. Moisture constraints if available
        
        Parameters:
        -----------
        temp_col : str
            Column name for temperature
        vpd_col : str
            Column name for VPD
        lai_col : str, optional
            Column name for LAI
        swc_col : str, optional
            Column name for soil water content
        gsi_threshold : float
            GSI threshold for growing season (default: 0.5)
        use_moisture : bool
            Whether to include moisture constraints
        use_phenology : bool
            Whether to include phenology constraints
            
        Returns:
        --------
        Tuple[pd.Series, pd.DataFrame]
            - Boolean series indicating growing season
            - DataFrame with all component indices for analysis
        """
        # Calculate GSI
        gsi = self.calculate_gsi(tmin_col=temp_col, vpd_col=vpd_col)
        
        # Start with GSI-based growing season
        growing_season = gsi > gsi_threshold
        
        # Create components dataframe
        components = pd.DataFrame({'gsi': gsi})
        
        # Track which constraints are applied
        constraints_used = ['GSI']
        
        # Add phenology constraint if available
        if use_phenology and lai_col is not None and lai_col in self.df.columns:
            try:
                phenology = self.calculate_phenology_growing_season(
                    lai_col=lai_col, method='threshold'
                )
                # Resample to match GSI index
                phenology = phenology.reindex(gsi.index, method='ffill')
                growing_season &= phenology
                components['phenology'] = phenology
                constraints_used.append('Phenology (LAI)')
            except Exception as e:
                print(f"Warning: Could not calculate phenology constraint: {e}")
        elif use_phenology:
            print(f"  Note: Phenology constraint skipped (LAI column '{lai_col}' not available)")
        
        # Add moisture constraint if available
        if use_moisture and swc_col is not None and swc_col in self.df.columns:
            try:
                moisture = self.calculate_moisture_constrained_season(swc_col=swc_col)
                moisture = moisture.reindex(gsi.index, method='ffill')
                growing_season &= moisture
                components['moisture'] = moisture
                constraints_used.append('Moisture (SWC)')
            except Exception as e:
                print(f"Warning: Could not calculate moisture constraint: {e}")
        elif use_moisture:
            print(f"  Note: Moisture constraint skipped (SWC column '{swc_col}' not available)")
        
        print(f"  Constraints applied: {' + '.join(constraints_used)}")
        
        components['growing_season'] = growing_season
        
        return growing_season, components


def filter_growing_season_scientific(df: pd.DataFrame,
                                     latitude: float,
                                     method: str = 'gsi',
                                     timestamp_col: str = 'TIMESTAMP',
                                     temp_col: str = 'ta',
                                     vpd_col: str = 'vpd',
                                     lai_col: Optional[str] = 'lai',
                                     swc_col: Optional[str] = 'swc_shallow',
                                     gsi_threshold: float = 0.5,
                                     return_components: bool = False,
                                     **kwargs) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Filter dataframe to growing season only using scientific methods.
    
    This is the main function to be called from the merge script.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with environmental data
    latitude : float
        Site latitude in degrees
    method : str
        Method to use:
        - 'gsi': Growing Season Index (Jolly et al., 2005)
        - 'thermal': Temperature-based (WMO definition)
        - 'phenology': LAI-based phenology
        - 'combined': Multi-factor approach (recommended)
    timestamp_col : str
        Name of timestamp column
    temp_col, vpd_col : str
        Column names for temperature and VPD (required)
    lai_col, swc_col : str or None
        Column names for LAI and soil water content (optional)
    gsi_threshold : float
        GSI threshold for growing season (default: 0.5)
    return_components : bool
        If True, also return component indices
    **kwargs : dict
        Additional arguments passed to specific methods
        
    Returns:
    --------
    pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]
        Filtered dataframe (and optionally component indices)
    """
    # Initialize detector
    detector = GrowingSeasonDetector(df, latitude, timestamp_col)
    
    # Validate optional columns - check if they exist in dataframe
    lai_col_valid = lai_col if (lai_col is not None and lai_col in df.columns) else None
    swc_col_valid = swc_col if (swc_col is not None and swc_col in df.columns) else None
    
    if method == 'gsi':
        gsi_values = detector.calculate_gsi(tmin_col=temp_col, vpd_col=vpd_col)
        gs_mask = gsi_values > gsi_threshold
        components = pd.DataFrame({'gsi': gsi_values})
        
    elif method == 'thermal':
        gs_mask = detector.calculate_thermal_growing_season(
            temp_col=temp_col,
            method=kwargs.get('thermal_method', 'wmo'),
            threshold=kwargs.get('temp_threshold', 5.0)
        )
        components = pd.DataFrame({'thermal': gs_mask})
        
    elif method == 'phenology':
        if lai_col_valid is None:
            raise ValueError(
                f"Phenology method requires LAI data. "
                f"Column '{lai_col}' not found in dataframe. "
                f"Available columns: {list(df.columns)}"
            )
        gs_mask = detector.calculate_phenology_growing_season(
            lai_col=lai_col_valid,
            method=kwargs.get('phenology_method', 'threshold')
        )
        components = pd.DataFrame({'phenology': gs_mask})
        
    elif method == 'combined':
        gs_mask, components = detector.calculate_combined_growing_season(
            temp_col=temp_col,
            vpd_col=vpd_col,
            lai_col=lai_col_valid,
            swc_col=swc_col_valid,
            gsi_threshold=gsi_threshold
        )
        
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'gsi', 'thermal', 'phenology', 'combined'")
    
    # Apply filter to original dataframe
    # Need to align the mask with original dataframe
    df_indexed = df.copy()
    if timestamp_col in df_indexed.columns:
        df_indexed = df_indexed.set_index(pd.to_datetime(df_indexed[timestamp_col]))
    
    # Resample mask to hourly to match potential hourly data
    gs_mask_hourly = gs_mask.reindex(df_indexed.index, method='ffill')
    
    # Filter and reset index
    filtered_df = df_indexed[gs_mask_hourly.fillna(False)].reset_index(drop=True)
    
    # Restore TIMESTAMP column if it was there
    if timestamp_col in df.columns and timestamp_col not in filtered_df.columns:
        # The timestamp should be preserved from original df
        mask_aligned = gs_mask.reindex(
            pd.to_datetime(df[timestamp_col]), method='ffill'
        ).fillna(False)
        filtered_df = df[mask_aligned.values].reset_index(drop=True)
    
    if return_components:
        return filtered_df, components
    return filtered_df


# Convenience function for common use cases
def get_growing_season_mask(df: pd.DataFrame,
                            latitude: float,
                            method: str = 'combined',
                            **kwargs) -> pd.Series:
    """
    Get a boolean mask indicating growing season periods.
    
    Convenience function that returns just the mask without filtering.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    latitude : float
        Site latitude
    method : str
        Detection method
    **kwargs : dict
        Additional arguments
        
    Returns:
    --------
    pd.Series
        Boolean mask for growing season
    """
    detector = GrowingSeasonDetector(df, latitude)
    
    if method == 'gsi':
        return detector.calculate_gsi(**kwargs) > kwargs.get('gsi_threshold', 0.5)
    elif method == 'thermal':
        return detector.calculate_thermal_growing_season(**kwargs)
    elif method == 'phenology':
        return detector.calculate_phenology_growing_season(**kwargs)
    elif method == 'combined':
        mask, _ = detector.calculate_combined_growing_season(**kwargs)
        return mask
    else:
        raise ValueError(f"Unknown method: {method}")
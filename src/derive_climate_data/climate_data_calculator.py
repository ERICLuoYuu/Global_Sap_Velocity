import numpy as np
import math
class ClimateDataCalculator:
    """
    A class to calculate various climate variables from meteorological data
    """
    @staticmethod
    def kelvin_to_celsius(temp_k):
        """Convert Kelvin to Celsius"""
        return temp_k - 273.15
    @staticmethod
    def calculate_vapor_pressure(temp_c):
        """
        Calculate saturated vapor pressure using Tetens Formula
        Args:
            temp_c (float): Temperature in Celsius
        Returns:
            float: Vapor pressure in millibars
        """
        return 6.1078 * np.exp((17.269 * temp_c) / (237.3 + temp_c))
    @staticmethod
    def calculate_rh(air_temp_k, dew_temp_k):
        """
        Calculate Relative Humidity using air temperature and dew point temperature
        Args:
            air_temp_k (float): Air temperature in Kelvin
            dew_temp_k (float): Dew point temperature in Kelvin
        Returns:
            float: Relative humidity in percentage
        """
        # Convert temperatures to Celsius
        air_temp_c = ClimateDataCalculator.kelvin_to_celsius(air_temp_k)
        dew_temp_c = ClimateDataCalculator.kelvin_to_celsius(dew_temp_k)
        
        # Ensure dew point temperature is not higher than air temperature
        if dew_temp_c > air_temp_c:
            dew_temp_c = air_temp_c
        
        # Calculate saturated vapor pressure (es) at air temperature
        es = ClimateDataCalculator.calculate_vapor_pressure(air_temp_c)
        
        # Calculate actual vapor pressure (ea) at dew point temperature
        ea = ClimateDataCalculator.calculate_vapor_pressure(dew_temp_c)
        
        # Calculate relative humidity
        rh = (ea / es) * 100
        
        # Ensure RH doesn't exceed 100%
        return min(rh, 100.0)
    @staticmethod
    def calculate_vpd(air_temp_k, dew_temp_k):
        """
        Calculate Vapor Pressure Deficit (VPD) using air temperature and dew point temperature
        Args:
            air_temp_k (float): Air temperature in Kelvin
            dew_temp_k (float): Dew point temperature in Kelvin
        Returns:
            float: Vapor Pressure Deficit in millibars
        """
        # Convert temperatures to Celsius
        air_temp_c = ClimateDataCalculator.kelvin_to_celsius(air_temp_k)
        dew_temp_c = ClimateDataCalculator.kelvin_to_celsius(dew_temp_k)
        
        # Ensure dew point temperature is not higher than air temperature
        if dew_temp_c > air_temp_c:
            dew_temp_c = air_temp_c
        
        # Calculate saturated vapor pressure (es) at air temperature
        es = ClimateDataCalculator.calculate_vapor_pressure(air_temp_c)
        
        # Calculate actual vapor pressure (ea) at dew point temperature
        ea = ClimateDataCalculator.calculate_vapor_pressure(dew_temp_c)
        
        # Calculate Vapor Pressure Deficit (VPD)
        vpd = max(es - ea, 0.0)  # Ensure VPD is never negative
        
        return vpd
    @staticmethod
    def calculate_wind_speed(u, v):
        """
        Calculate wind speed from u and v components
        u: u-component of wind (east-west direction)
        v: v-component of wind (north-south direction)
        Returns: wind speed in m/s
        """
        wind_speed = np.sqrt(u**2 + v**2)
        return wind_speed
    @staticmethod
    def calculate_net_radiation(sw, lw, albedo=0.23):
        """
        Calculate net radiation from shortwave and longwave radiation
        sw: incoming shortwave radiation in W/m^2
        lw: incoming longwave radiation in W/m^2
        albedo: surface albedo (default: 0.23)
        Returns: net radiation in W/m^2
        """
        net_radiation = (1 - albedo) * sw + lw
        return net_radiation
    @staticmethod
    def calculate_ppfd_from_radiation(Rs: float, 
                                    method: str = 'standard') -> float:
        """
        Calculate PPFD from solar radiation measurements
        
        Parameters:
        -----------
        Rs : float or np.ndarray
            Solar radiation in W/m²
        method : str
            Conversion method: 'standard', 'McCree', or 'Meek'
            
        Returns:
        --------
        float or np.ndarray : PPFD in μmol/m²/s
        
        References:
        -----------
        - McCree (1972) Agricultural Meteorology
        - Meek et al. (1984) Agronomy Journal
        - Jones (2014) Plants and Microclimate
        """
        if method == 'standard':
            # Standard conversion factor (2.02 μmol/J)
            # Assumes 45% of solar radiation is in PAR waveband
            return Rs * 2.02
        
        elif method == 'McCree':
            # McCree (1972) method
            # Assumes 48% of solar radiation is in PAR waveband
            # Conversion factor of 4.6 μmol/W for PAR
            return Rs * 0.48 * 4.6
        
        elif method == 'Meek':
            # Meek et al. (1984) method
            # Includes slight variation with solar elevation
            # This is a simplified version of their equation
            return Rs * 2.04
        
        else:
            raise ValueError("Method must be 'standard', 'McCree', or 'Meek'")
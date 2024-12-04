import numpy as np
class ClimateDataCalculator:
    """
    A class to calculate various climate variables from meteorological data
    """
    @staticmethod
    def calculate_rh(t, td):
        """
        Calculate relative humidity from temperature and dewpoint
        t: temperature in Celsius
        td: dewpoint temperature in Celsius
        Returns: RH in percentage (0-100)
        """
        # Constants for Magnus formula
        a = 17.625
        b = 243.04
        
        # Calculate vapor pressure and saturated vapor pressure
        es = np.exp((a * t) / (b + t))  # Saturated vapor pressure
        e =  np.exp((a * td) / (b + td))  # Actual vapor pressure
        
        rh = (e / es) * 100
        return rh
    @staticmethod
    def calculate_vpd(T: float, RH: float) -> float:
        """
        Calculate Vapor Pressure Deficit (VPD)
        
        Parameters:
        -----------
        T : float
            Temperature in degrees Celsius
        RH : float
            Relative humidity (0-100)
            
        Returns:
        --------
        float : VPD in kPa
        
        References:
        -----------
        1. Murray, F.W. (1967). "On the computation of saturation vapor pressure"
        2. Allen, R.G., et al. (1998). FAO Irrigation and Drainage Paper No. 56
        3. Buck, A.L. (1981). "New equations for computing vapor pressure and enhancement factor"
        """
        # Calculate saturation vapor pressure
        es = 0.61121 * np.exp((18.678 - T/234.5) * (T/(257.14 + T)))
        
        # Calculate actual vapor pressure
        ea = es * (RH/100)
        
        # Calculate VPD
        vpd = es - ea
        
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
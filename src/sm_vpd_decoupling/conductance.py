# src/sm_vpd_decoupling/conductance.py
"""Whole-tree canopy conductance from sap flux density (Flo et al. 2021, Eqn 2).

Flo, V., Martinez-Vilalta, J., et al. (2021). "Climate and functional traits
jointly mediate tree water-use strategies." New Phytologist 231(2): 617-630,
doi:10.1111/nph.17404, Eqn 2 (after Phillips & Oren 1998).

    G_Asw = (115.8 + 0.4236*T) * (SFD/VPD) * (eta*T0/(T0+T)) * exp(0.00012*h)

with SFD in kg m-2_Asw s-1, T in degC, VPD in kPa, h = altitude (m).

NOTE: Gc is proportional to 1/VPD. Binning Gc BY VPD therefore induces a
spurious negative Gc-VPD relationship (Oren et al. 1999). The SM leg of the
decoupling (computed within VPD bins) is unaffected; the VPD leg is biased.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 1 cm3 cm-2 h-1 of water -> kg m-2 s-1 (rho_water = 1 g cm-3).
SFD_CM3CM2H_TO_KGM2S = 1e-3 * 1e4 / 3600.0
ETA = 44.6  # mol m-3, molar air density at STP
T0_K = 273.0  # K
SW_TO_PPFD = 2.04  # umol J-1, shortwave -> PAR (used by loader)

Numeric = "float | pd.Series"


def sfd_to_kg_m2_s(sfd_cm3_cm2_h):
    """Convert sap flux density from cm3 cm-2 h-1 to kg m-2 s-1."""
    return sfd_cm3_cm2_h * SFD_CM3CM2H_TO_KGM2S


def canopy_conductance(sap_velocity, tair_c, vpd_kpa, altitude_m):
    """Flo 2021 Eqn 2 whole-tree canopy conductance per sapwood area (mol m-2 s-1).

    Accepts scalars or pandas Series (broadcast). VPD <= 0 -> NaN (avoids div0).
    """
    sfd = sfd_to_kg_m2_s(sap_velocity)
    vpd = vpd_kpa
    if isinstance(vpd, pd.Series):
        vpd = vpd.where(vpd > 0)
    elif vpd <= 0:
        return float("nan")
    gc = (115.8 + 0.4236 * tair_c) * (sfd / vpd) * (ETA * T0_K / (T0_K + tair_c)) * np.exp(0.00012 * altitude_m)
    return gc

"""Site-level SM-VPD decoupling for sap velocity and canopy conductance.

Ports Liu et al. (2020, Nat. Commun. 11:4892) SM/VPD percentile-binning
decoupling to site-level sap flow. Two responses: sap velocity (E, water-use)
and Flo et al. (2021, New Phytol. 231:617-630, Eqn 2) canopy conductance (Gc,
SIF-analog). See docs/superpowers/specs/2026-06-16-sm-vpd-decoupling-sapflow-design.md.
"""

from __future__ import annotations

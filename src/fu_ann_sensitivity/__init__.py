"""Fu et al. 2022 (Nat. Commun. 13:989) ANN sensitivity, ported to sap velocity (E)
and Flo-2021 canopy conductance (Gc).

Per site, a small feed-forward ANN learns the response surface over
[Tair, VPD, SWC, PPFD]; local +1 SD perturbations of the (z-scored) inputs give
the SWC and VPD partial-derivative sensitivities, binned into SWC x VPD percentile
grids (5x5 and 10x10) and aggregated across sites with per-cell t-tests.

Gc note (per project decision): a negative dGc/dVPD is genuine regulation, not an
artifact -- d ln Gc / d ln VPD = d ln E / d ln VPD - 1, so it is equivalent to E
being sub-proportional to VPD (stomatal closure); the null is *flat* Gc. The E-leg
and Gc-leg VPD sensitivities are algebraically linked (same signal, two forms), and
VPD measurement error inflates the magnitude (VPD on both axes) but not the sign.
"""

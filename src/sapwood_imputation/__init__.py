"""Allometric imputation of sapwood area (ASW) for trees lacking it.

Fits a per-functional-type log-log relationship
``ln(ASW) = a + b * ln(DBH)`` on all SAPFLUXNET trees that have measured
sapwood area, then predicts ASW (cm²) for trees where it is missing.
"""

from __future__ import annotations

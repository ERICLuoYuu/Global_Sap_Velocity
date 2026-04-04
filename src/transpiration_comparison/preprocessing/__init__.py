"""Preprocessing: regridding, temporal aggregation, normalization."""

from .normalize import compute_anomaly, compute_climatology, zscore_normalize
from .regrid import regrid_to_common
from .temporal import aggregate_to_daily

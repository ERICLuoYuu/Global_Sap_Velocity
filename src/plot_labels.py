"""Shared display-label vocabulary for figure axes, titles, and legends.

Maps internal data-column names (used as DataFrame keys throughout the pipeline)
to human-readable labels, so figures across the SM-VPD decoupling, Fu ANN
sensitivity, and afternoon-depression modules use *consistent* terms WITHOUT
renaming any data columns, model artifacts, CSV outputs, or tests.

Conventions agreed for these figures:
  * sap velocity (``E``/``E_norm``) is shown as **"sap flow"**;
  * canopy conductance keeps the conventional **"Gc"**;
  * ERA5-Land soil layers are shown by depth, and ``root_zone_sm`` — a
    depth-weighted mean — is shown as **"weighted mean sm"**.

The maps fall back to the raw name, so an unmapped column still renders rather
than raising.
"""

from __future__ import annotations

# Measured-response labels. Per-site-normalized variants (``_norm``) map to the
# same label as the raw response — the normalization is implied by the
# percentile-bin axes, so the shown term stays the plain physical quantity.
_RESPONSE_LABELS = {
    "E": "sap flow",
    "E_norm": "sap flow",
    "Gc": "Gc",
    "Gc_norm": "Gc",
}

# Soil-moisture variant labels. ERA5-Land layer depths (0-7, 7-28, 28-100,
# 100-289 cm); ``root_zone_sm`` is a depth-weighted mean of the upper layers.
_SM_LABELS = {
    "swvl1": "SM 0–7 cm",
    "swvl2": "SM 7–28 cm",
    "swvl3": "SM 28–100 cm",
    "swvl4": "SM 100–289 cm",
    "root_zone_sm": "weighted mean sm",
}


def response_label(name: str) -> str:
    """Display label for a response column (falls back to the raw name)."""
    return _RESPONSE_LABELS.get(name, name)


def sm_label(name: str) -> str:
    """Display label for a soil-moisture variant column (falls back to raw name)."""
    return _SM_LABELS.get(name, name)

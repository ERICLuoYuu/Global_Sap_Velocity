"""One corpus at sapwood scale (cm3 cm-2 h-1).

SAPFLUXNET and the internal European sites each ship two flavours of every site:
``plant`` (sap flow, cm3 h-1) and ``sapwood`` (sap flux density, cm3 cm-2 h-1).
Most sites exist at both; some only at plant scale. This package assembles the
union at sapwood scale -- copying the sites that already carry it, and dividing
the plant-only ones by an imputed sapwood area from
:mod:`src.sapwood_imputation`.

Entry point::

    python -m src.sapwood_combined.build_sapwood_combined --help
"""

from __future__ import annotations

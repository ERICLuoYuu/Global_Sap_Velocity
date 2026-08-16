"""Miniature two-corpus fixtures.

The real corpus is 227 sites and 4 GB; nothing here touches it. These builders
make the smallest tree that still exercises every route:

    EXT_BOTH    external, present at plant *and* sapwood  -> native (sapwood wins)
    EXT_PLANT   external, plant only                      -> converted
    EXT_SAP     external, sapwood only                    -> native
    INT_PLANT   internal, plant only                      -> converted
    INT_SAP     internal, sapwood only                    -> native
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest

TIMESTAMPS = (
    "2015-01-01T00:00:00Z",
    "2015-01-01T01:00:00Z",
    "2015-01-01T02:00:00Z",
)
SOLAR = (
    "2015-01-01T00:19:33Z",
    "2015-01-01T01:19:33Z",
    "2015-01-01T02:19:33Z",
)

#: The seven kinds both corpora ship.
KINDS = ("sapf_data", "plant_md", "env_data", "env_md", "site_md", "stand_md", "species_md")

#: The two the archive ships and the internal corpus never generated. Measured
#: 2026-08-15: 194/194 external plant and 165/165 external sapwood sites carry
#: both; 0/30 internal sites carry either.
FLAG_KINDS = ("sapf_flags", "env_flags")

#: What an archive site looks like on disk, and what an internal one looks like.
EXTERNAL_KINDS = KINDS + FLAG_KINDS
INTERNAL_KINDS = KINDS

#: Flag vocabulary, from the archive's own values.
FLAG_VALUES = ("", "OUT_WARN", "NA_ADDED")


def sapflow_frame(codes: Sequence[str], values: Mapping[str, Sequence[float]] | None = None) -> pd.DataFrame:
    """A sapf_data frame: two timestamp columns then one column per tree."""
    data = {"TIMESTAMP": list(TIMESTAMPS), "solar_TIMESTAMP": list(SOLAR)}
    for i, code in enumerate(codes):
        if values is not None and code in values:
            data[code] = list(values[code])
        else:
            data[code] = [100.0 * (i + 1), 200.0 * (i + 1), 300.0 * (i + 1)]
    return pd.DataFrame(data)


def sapf_flags_frame(codes: Sequence[str]) -> pd.DataFrame:
    """A sapf_flags frame: the same columns as sapf_data, holding QC labels.

    Column-for-column identity with ``sapf_data`` is not a convenience of this
    fixture, it is the archive's actual contract -- verified exact, order
    included, on all 359 external files.
    """
    data = {"TIMESTAMP": list(TIMESTAMPS), "solar_TIMESTAMP": list(SOLAR)}
    for i, code in enumerate(codes):
        data[code] = [FLAG_VALUES[(i + j) % len(FLAG_VALUES)] for j in range(len(TIMESTAMPS))]
    return pd.DataFrame(data)


def metadata_frame(
    site: str,
    codes: Sequence[str],
    areas: Mapping[str, float | None] | None = None,
    flags: Mapping[str, str] | None = None,
    units: str = "cm3 h-1",
) -> pd.DataFrame:
    """A plant_md frame in the shape `*_asw_filled/` delivers it."""
    rows = []
    for code in codes:
        area = 10.0 if areas is None else areas.get(code)
        rows.append(
            {
                "si_code": site,
                "pl_code": code,
                "pl_species": "Pinus sylvestris",
                "pl_dbh": 30.0,
                "pl_sap_units": units,
                "pl_sap_units_orig": "kg h-1",
                "pl_sapw_area": area,
                "pl_sapw_area_filled": area,
                "pl_sapw_area_source": (flags or {}).get(code, "measured"),
            }
        )
    return pd.DataFrame(rows)


def write_site(
    directory: Path,
    site: str,
    codes: Sequence[str],
    areas: Mapping[str, float | None] | None = None,
    units: str = "cm3 h-1",
    values: Mapping[str, Sequence[float]] | None = None,
    kinds: Iterable[str] = EXTERNAL_KINDS,
) -> None:
    """Write one site's full set of files into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    for kind in kinds:
        path = directory / f"{site}_{kind}.csv"
        if kind == "sapf_data":
            sapflow_frame(codes, values).to_csv(path, index=False)
        elif kind == "plant_md":
            metadata_frame(site, codes, areas, units=units).to_csv(path, index=False)
        elif kind == "sapf_flags":
            sapf_flags_frame(codes).to_csv(path, index=False)
        else:
            # The remaining kinds are copied verbatim; their content is
            # irrelevant beyond being distinguishable per site and kind.
            pd.DataFrame({"si_code": [site], "kind": [kind]}).to_csv(path, index=False)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A tiny two-corpus tree rooted at ``tmp_path``."""
    csv = tmp_path / "csv"
    internal = tmp_path / "internal"

    write_site(csv / "plant", "EXT_BOTH", ["EXT_BOTH_Psy_Jt_1", "EXT_BOTH_Psy_Jt_2"])
    write_site(csv / "sapwood", "EXT_BOTH", ["EXT_BOTH_Psy_Jt_1", "EXT_BOTH_Psy_Jt_2"], units="cm3 cm-2 h-1")
    write_site(csv / "plant", "EXT_PLANT", ["EXT_PLANT_Psy_Jt_1", "EXT_PLANT_Psy_Jt_2"])
    write_site(csv / "sapwood", "EXT_SAP", ["EXT_SAP_Psy_Jt_1"], units="cm3 cm-2 h-1")
    # The internal corpus ships no flags files. Faithful to disk, and the reason
    # the product's kind set is per-site rather than global.
    write_site(internal / "plant", "INT_PLANT", ["INT_PLANT_Psy_Jt_1"], kinds=INTERNAL_KINDS)
    write_site(
        internal / "sapwood",
        "INT_SAP",
        ["INT_SAP_Psy_Jt_1"],
        units="cm3 cm-2 h-1",
        kinds=INTERNAL_KINDS,
    )

    # The filled-ASW directories mirror each flavour's site list.
    write_site(
        csv / "plant_asw_filled",
        "EXT_BOTH",
        ["EXT_BOTH_Psy_Jt_1", "EXT_BOTH_Psy_Jt_2"],
        kinds=("plant_md",),
    )
    write_site(
        csv / "plant_asw_filled",
        "EXT_PLANT",
        ["EXT_PLANT_Psy_Jt_1", "EXT_PLANT_Psy_Jt_2"],
        areas={"EXT_PLANT_Psy_Jt_1": 50.0, "EXT_PLANT_Psy_Jt_2": 25.0},
        kinds=("plant_md",),
    )
    write_site(
        csv / "plant_asw_filled",
        "INT_PLANT",
        ["INT_PLANT_Psy_Jt_1"],
        areas={"INT_PLANT_Psy_Jt_1": 20.0},
        kinds=("plant_md",),
    )
    for site, codes in (
        ("EXT_BOTH", ["EXT_BOTH_Psy_Jt_1", "EXT_BOTH_Psy_Jt_2"]),
        ("EXT_SAP", ["EXT_SAP_Psy_Jt_1"]),
        ("INT_SAP", ["INT_SAP_Psy_Jt_1"]),
    ):
        write_site(csv / "sapwood_asw_filled", site, codes, units="cm3 cm-2 h-1", kinds=("plant_md",))

    return tmp_path

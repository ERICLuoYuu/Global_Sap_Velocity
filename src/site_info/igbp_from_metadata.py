"""Derive one IGBP class per site from the metadata the site itself ships.

``si_igbp`` is absent at exactly the 23 internal sites -- they never had
SAPFLUXNET site metadata -- so ``pft_resolution`` falls back to MODIS there, and
MODIS is least reliable at precisely the canopies those sites have: open
dehesas, thinned plantations, subalpine stands.  Measured 2026-08-20, MODIS put
ESP_ABR (a holm-oak dehesa) in *Croplands* and FRA_BIL (a maritime-pine
plantation) in *Woody Savannas*.

This module derives the class from two files the provider already filled in.

Why ``st_forest_type`` leads
----------------------------
It is the provider's own one-line description of the STAND ("Coniferous",
"Mixed beech-spruce forest", "Woody Savanna").  ``species_md`` describes only
the trees that were INSTRUMENTED: ``sp_ntrees`` is ``nunique(pl_code)``, so at
AUT_ZOE it reads 4 beech + 3 sycamore -- both broadleaf deciduous -- while the
stand the provider describes is a mixed beech-SPRUCE forest.  Deriving from
species alone would have labelled that site DBF and been wrong, because nobody
put a sensor on the spruce.

Why tree counts cannot settle "mixed"
-------------------------------------
The IGBP mixed-forest test is "neither leaf type exceeds 60% of COVER".
``sp_basal_area_perc`` is NA at all 23 sites, and ``sp_ntrees`` is measurement
effort, not composition -- the 5 hornbeam / 5 pine at DEU_HAR is a sampling
design.  Evaluating a cover threshold with it would be fabricated precision, so
the species tier reports MF for any two-leaf-type stand and marks the dominance
unverifiable rather than inventing a split.

Composition, not substitution
-----------------------------
``st_forest_type`` names the leaf TYPE; the leaf HABIT comes from
``sp_leaf_habit``, and only from the species that match the stated type.  That
is what makes FIN_HYY answerable by rule: the stand is "Coniferous", its one
needleleaf species (Pinus sylvestris) is evergreen, so it is ENF -- the birch
sharing the plot is not evidence about the habit of the conifers.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

#: Conifer genera occurring in this corpus.  Needleleaf is a property of the
#: genus and no column states it, so it has to be enumerated.  A genus absent
#: from this set is treated as broadleaf -- see UNKNOWN_GENUS_IS_BROADLEAF.
CONIFER_GENERA = frozenset(
    {
        "abies",
        "araucaria",
        "athrotaxis",
        "austrocedrus",
        "callitris",
        "calocedrus",
        "cedrus",
        "chamaecyparis",
        "cryptomeria",
        "cunninghamia",
        "cupressus",
        "dacrydium",
        "fitzroya",
        "juniperus",
        "keteleeria",
        "larix",
        "libocedrus",
        "metasequoia",
        "nothotsuga",
        "picea",
        "pinus",
        "podocarpus",
        "prumnopitys",
        "pseudotsuga",
        "sciadopitys",
        "sequoia",
        "sequoiadendron",
        "taxodium",
        "taxus",
        "thuja",
        "tsuga",
        "widdringtonia",
    }
)

#: Documented rather than implicit: an unrecognised genus is called broadleaf.
#: Every unknown in this corpus is an angiosperm, and a misfiled conifer would
#: show up as an implausible EBF/DBF rather than disappearing.
UNKNOWN_GENUS_IS_BROADLEAF = True

NEEDLELEAF = "NL"
BROADLEAF = "BL"
EVERGREEN = "E"
DECIDUOUS = "D"
UNKNOWN_HABIT = "?"

#: ``sp_leaf_habit`` across the 23 internal sites is exactly two strings --
#: "cold deciduous" (18) and "evergreen" (20) -- but the external corpus adds
#: "drought deciduous" and "brevideciduous", so match on tokens, not equality.
_DECIDUOUS_TOKENS = ("decid", "brevi")
_EVERGREEN_TOKENS = ("evergreen", "perennial")

#: Kept identical to ``pft_resolution._ABSENT``; a test asserts they agree.
_ABSENT = {"", "nan", "na", "n/a", "none", "null", "-9999", "unknown"}

#: Tiers, most authoritative first.  Reported per site so a reader can see
#: which evidence produced each value rather than trusting a bare code.
TIER_STAND = "stand"
TIER_SPECIES = "species"
TIER_NONE = "unresolved"


def is_absent(value) -> bool:
    """True when a metadata field carries no usable value.

    Deliberately kept dependency-free rather than importing the pandas-backed
    twin in ``pft_resolution``: this module is imported by a CLI that otherwise
    needs no pandas.  ``float("nan")`` stringifies to ``"nan"``, which
    ``_ABSENT`` already holds, so the text test covers the NaN case too.
    """
    if value is None:
        return True
    return str(value).strip().lower() in _ABSENT


def leaf_type(species_name: str) -> str:
    """NEEDLELEAF or BROADLEAF, from the genus (the first word of the binomial)."""
    text = str(species_name).strip()
    if not text:
        return BROADLEAF if UNKNOWN_GENUS_IS_BROADLEAF else NEEDLELEAF
    genus = text.split()[0].lower()
    return NEEDLELEAF if genus in CONIFER_GENERA else BROADLEAF


def leaf_habit(value) -> str:
    """EVERGREEN, DECIDUOUS or UNKNOWN_HABIT, from ``sp_leaf_habit``.

    Deciduous is tested first: "cold deciduous" and "drought deciduous" both
    carry a habit word, and no corpus value contains both stems.
    """
    if is_absent(value):
        return UNKNOWN_HABIT
    text = str(value).strip().lower()
    if any(token in text for token in _DECIDUOUS_TOKENS):
        return DECIDUOUS
    if any(token in text for token in _EVERGREEN_TOKENS):
        return EVERGREEN
    return UNKNOWN_HABIT


def _forest_class(kind: str, habits: Iterable[str]) -> str | None:
    """ENF/DNF or EBF/DBF, given a leaf type and the habits of THAT type."""
    present = set(h for h in habits if h != UNKNOWN_HABIT)
    if not present:
        return None
    if present == {EVERGREEN}:
        return "ENF" if kind == NEEDLELEAF else "EBF"
    if present == {DECIDUOUS}:
        return "DNF" if kind == NEEDLELEAF else "DBF"
    # Both habits inside one leaf type (a larch among spruce).  IGBP splits
    # those on cover, which is exactly what this corpus cannot measure.
    return None


def class_from_stand_type(forest_type, species: Sequence[tuple[str, str]]) -> str | None:
    """IGBP code from the provider's ``st_forest_type``, or None if it is silent.

    ``species`` is a sequence of ``(leaf_type, habit)`` pairs.  Only species
    matching the stated leaf type contribute their habit -- see the module
    docstring for why that is what makes FIN_HYY decidable.
    """
    if is_absent(forest_type):
        return None
    text = str(forest_type).strip().lower()

    # Savanna is tested before the leaf words: "Open Holm Oak woodland /
    # Savanna" contains a leaf word too, and in the IGBP scheme cover outranks
    # leaf type -- a dehesa of evergreen broadleaf oaks is SAV/WSA, not EBF.
    if "savanna" in text:
        return "WSA" if ("woody" in text or "woodland" in text) else "SAV"
    if "wetland" in text or "swamp" in text:
        return "WET"
    if "mixed" in text:
        return "MF"

    if "conifer" in text or "needle" in text:
        kind = NEEDLELEAF
    elif "broadleaf" in text or "broad-leaved" in text or "broadleaved" in text or "deciduous" in text:
        kind = BROADLEAF
    else:
        return None

    return _forest_class(kind, [habit for typ, habit in species if typ == kind])


def class_from_species(species: Sequence[tuple[str, str]]) -> str | None:
    """IGBP code from the instrumented species alone -- the fallback tier.

    Both leaf types present means MF, because the 60% cover test cannot be
    evaluated here (no basal area, and tree counts are sampling effort).  This
    tier only sees trees somebody instrumented, so it UNDER-reports mixture;
    that is why it runs second.
    """
    if not species:
        return None
    types = set(typ for typ, _ in species)
    if len(types) > 1:
        return "MF"
    kind = types.pop()
    return _forest_class(kind, [habit for _, habit in species])


def derive_igbp(forest_type, species_rows: Sequence[tuple[str, str]]) -> tuple[str | None, str, str]:
    """``(code, tier, evidence)`` for one site.

    ``species_rows`` is a sequence of ``(sp_name, sp_leaf_habit)`` straight from
    ``species_md``.  Never raises: a site this cannot classify comes back as
    ``(None, TIER_NONE, reason)`` so the caller reports it, rather than one bad
    row killing a corpus-wide run.
    """
    species = [(leaf_type(name), leaf_habit(habit)) for name, habit in species_rows]

    code = class_from_stand_type(forest_type, species)
    if code is not None:
        return code, TIER_STAND, f"st_forest_type={str(forest_type).strip()!r}"

    code = class_from_species(species)
    if code is not None:
        shape = "+".join(sorted({f"{typ}{habit}" for typ, habit in species}))
        return code, TIER_SPECIES, f"species={shape} (no usable st_forest_type)"

    if not species:
        return None, TIER_NONE, "no species_md rows"
    return None, TIER_NONE, "leaf habit unknown, or ambiguous within one leaf type"


def summarise(rows: dict[str, tuple[str | None, str, str]]) -> list[str]:
    """One line per site plus a tier tally, for printing at the end of a run."""
    lines = []
    tiers: dict[str, int] = {}
    for site in sorted(rows):
        code, tier, evidence = rows[site]
        tiers[tier] = tiers.get(tier, 0) + 1
        lines.append(f"  {site:<10} {code or '--':<4}  {tier:<9} {evidence}")
    tally = ", ".join(f"{count} {tier}" for tier, count in sorted(tiers.items()))
    lines.append(f"  ---- {len(rows)} site(s): {tally}")
    return lines

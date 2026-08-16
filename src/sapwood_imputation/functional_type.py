"""Classify a tree as angiosperm or gymnosperm from its species name.

The SAPFLUXNET plant metadata stores ``pl_species`` ("Genus species") but no
explicit functional-type field. The angiosperm/gymnosperm split is a clean
botanical dichotomy among seed plants: gymnosperms are the conifers plus
Ginkgo, the cycads, and the Gnetales; everything else with a resolvable genus
is an angiosperm.

We therefore enumerate the (finite) set of gymnosperm genera and treat any
other resolvable genus as angiosperm. Blank names, or family-level names
(``*aceae``) that are not genera, return ``UNKNOWN`` so the caller can decide
not to impute rather than guess.
"""

from __future__ import annotations

ANGIOSPERM = "angiosperm"
GYMNOSPERM = "gymnosperm"
UNKNOWN = "unknown"

# ---------------------------------------------------------------------------
# All extant gymnosperm genera (conifers + Ginkgo + cycads + Gnetales).
# Capitalised genus names. Source: standard botanical classification; covers
# every gymnosperm genus that appears in SAPFLUXNET and well beyond, so the
# classifier is robust to new sites.
# ---------------------------------------------------------------------------
GYMNOSPERM_GENERA: frozenset[str] = frozenset(
    {
        # Pinaceae
        "Abies",
        "Cathaya",
        "Cedrus",
        "Keteleeria",
        "Larix",
        "Nothotsuga",
        "Picea",
        "Pinus",
        "Pseudolarix",
        "Pseudotsuga",
        "Tsuga",
        # Araucariaceae
        "Agathis",
        "Araucaria",
        "Wollemia",
        # Cupressaceae (incl. former Taxodiaceae)
        "Actinostrobus",
        "Athrotaxis",
        "Austrocedrus",
        "Callitris",
        "Callitropsis",
        "Calocedrus",
        "Chamaecyparis",
        "Cryptomeria",
        "Cunninghamia",
        "Cupressus",
        "Diselma",
        "Fitzroya",
        "Fokienia",
        "Glyptostrobus",
        "Hesperocyparis",
        "Juniperus",
        "Libocedrus",
        "Metasequoia",
        "Microbiota",
        "Neocallitropsis",
        "Papuacedrus",
        "Pilgerodendron",
        "Platycladus",
        "Sequoia",
        "Sequoiadendron",
        "Taiwania",
        "Taxodium",
        "Tetraclinis",
        "Thuja",
        "Thujopsis",
        "Widdringtonia",
        "Xanthocyparis",
        # Podocarpaceae
        "Acmopyle",
        "Afrocarpus",
        "Dacrycarpus",
        "Dacrydium",
        "Falcatifolium",
        "Halocarpus",
        "Lagarostrobos",
        "Lepidothamnus",
        "Manoao",
        "Microcachrys",
        "Nageia",
        "Parasitaxus",
        "Pherosphaera",
        "Phyllocladus",
        "Podocarpus",
        "Prumnopitys",
        "Retrophyllum",
        "Saxegothaea",
        "Sundacarpus",
        # Taxaceae
        "Amentotaxus",
        "Austrotaxus",
        "Cephalotaxus",
        "Pseudotaxus",
        "Taxus",
        "Torreya",
        # Sciadopityaceae
        "Sciadopitys",
        # Ginkgoaceae
        "Ginkgo",
        # Cycads
        "Bowenia",
        "Ceratozamia",
        "Cycas",
        "Dioon",
        "Encephalartos",
        "Lepidozamia",
        "Macrozamia",
        "Microcycas",
        "Stangeria",
        "Zamia",
        # Gnetales
        "Ephedra",
        "Gnetum",
        "Welwitschia",
    }
)


def extract_genus(species: str | None) -> str | None:
    """Return the capitalised genus from a species string, or None.

    Returns None for blank input or for family-level names (``*aceae``),
    which are not genera and cannot be resolved to a functional type.
    """
    if not isinstance(species, str):
        return None
    cleaned = species.strip()
    if not cleaned:
        return None
    genus = cleaned.split()[0].capitalize()
    if genus.lower().endswith("aceae"):
        return None
    return genus


def classify(species: str | None) -> str:
    """Classify a species/genus as angiosperm, gymnosperm, or unknown.

    Parameters
    ----------
    species
        Full species string ("Pinus sylvestris") or bare genus ("Pinus").
        Case-insensitive; surrounding whitespace is ignored.

    Returns
    -------
    str
        One of :data:`ANGIOSPERM`, :data:`GYMNOSPERM`, :data:`UNKNOWN`.
    """
    genus = extract_genus(species)
    if genus is None:
        return UNKNOWN
    return GYMNOSPERM if genus in GYMNOSPERM_GENERA else ANGIOSPERM

"""Tests for the taxon refinement of the functional-type offset.

The point of this layer is *restraint*: it must correct a taxon whose offset is
real and refuse one whose apparent offset is a single site's. Most of these
tests therefore check that it declines to act.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.sapwood_imputation.allometry import fit_loglog_with_group_offset
from src.sapwood_imputation.taxon_offsets import (
    TaxonOffsets,
    fit_taxon_offsets,
    normalise_genus,
    normalise_species,
)

GROUP, DBH, ASW, SITE, SPECIES = "ft", "dbh", "asw", "site", "sp"


def _stand(site: str, species: str, ft: str, factor: float, n: int = 12, seed: int = 0, sigma: float = 0.12):
    rng = np.random.default_rng(seed)
    dbh = rng.uniform(12, 55, size=n)
    return pd.DataFrame(
        {
            SITE: site,
            SPECIES: species,
            GROUP: ft,
            DBH: dbh,
            ASW: factor * dbh**1.7 * np.exp(rng.normal(0, sigma, size=n)),
        }
    )


def _corpus(frames) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True)


def _fit(df):
    return fit_loglog_with_group_offset(df, GROUP, DBH, ASW, min_n=2)


def _offsets(df, **kw) -> TaxonOffsets:
    return fit_taxon_offsets(df, _fit(df), GROUP, DBH, ASW, SITE, SPECIES, **kw)


def _consistent_corpus(seed: int = 0) -> pd.DataFrame:
    """Larix genuinely thin-sapwood at five separate sites; Pinus normal."""
    frames = []
    for i in range(5):
        frames.append(_stand(f"L{i}", "Larix decidua", "gymnosperm", 0.30, seed=seed + i))
    for i in range(6):
        frames.append(_stand(f"P{i}", "Pinus sylvestris", "gymnosperm", 1.0, seed=seed + 20 + i))
    for i in range(6):
        frames.append(_stand(f"Q{i}", "Quercus robur", "angiosperm", 1.2, seed=seed + 40 + i))
    return _corpus(frames)


@pytest.mark.unit
def test_a_consistent_taxon_earns_an_offset_in_the_right_direction() -> None:
    df = _consistent_corpus()
    taxon = _offsets(df)

    offset, level = taxon.lookup("Larix decidua")
    assert level in ("species", "genus")
    # Larix has LESS sapwood than the model gives it -> negative log offset.
    assert offset < 0
    # And the correction moves predictions the right way.
    assert np.exp(offset) < 1.0


@pytest.mark.unit
def test_a_taxon_at_one_site_earns_nothing() -> None:
    """Its 'taxon effect' would be that site's effect exactly."""
    df = _corpus(
        [
            _stand("ONLY", "Larix decidua", "gymnosperm", 0.25, n=40, seed=1),
            *[_stand(f"P{i}", "Pinus sylvestris", "gymnosperm", 1.0, seed=20 + i) for i in range(6)],
            *[_stand(f"Q{i}", "Quercus robur", "angiosperm", 1.2, seed=40 + i) for i in range(6)],
        ]
    )
    taxon = _offsets(df)

    assert taxon.lookup("Larix decidua") == (0.0, "functional_type")


@pytest.mark.unit
def test_a_taxon_whose_sites_disagree_earns_nothing() -> None:
    """The Populus case: one genus, ecologically opposite members.

    Its sites straddle the model, so the mean is not distinguishable from site
    noise and the gate must refuse it rather than average the two away.
    """
    frames = [
        _stand("A", "Populus euphratica", "angiosperm", 0.3, seed=1),
        _stand("B", "Populus euphratica", "angiosperm", 0.35, seed=2),
        _stand("C", "Populus tremuloides", "angiosperm", 3.0, seed=3),
        _stand("D", "Populus tremuloides", "angiosperm", 3.4, seed=4),
    ]
    frames += [_stand(f"Q{i}", "Quercus robur", "angiosperm", 1.2, seed=40 + i) for i in range(6)]
    frames += [_stand(f"P{i}", "Pinus sylvestris", "gymnosperm", 1.0, seed=20 + i) for i in range(6)]
    df = _corpus(frames)
    taxon = _offsets(df)

    # The GENUS must not earn an offset: its two halves cancel and disagree.
    genus_keys = {e.key for e in taxon.entries if e.level == "genus"}
    assert "Populus" not in genus_keys
    # The species, each consistent across its own two sites, may.
    assert taxon.lookup("Populus euphratica")[0] <= 0.0


@pytest.mark.unit
def test_two_trees_at_two_sites_cannot_manufacture_confidence() -> None:
    """Guards a real defect: unmoderated, this produced z=244 on 2 trees.

    With two sites the sample variance is unstable, and when the two site means
    land close together by luck it collapses toward zero and z explodes. The
    moderated variance must keep such a taxon's z modest.
    """
    freak = pd.concat(
        [
            pd.DataFrame({SITE: s, SPECIES: "Goupia glabra", GROUP: "angiosperm", DBH: [30.0], ASW: [1.9 * 30.0**1.7]})
            for s in ("G1", "G2")
        ],
        ignore_index=True,
    )
    frames = [freak]
    frames += [_stand(f"Q{i}", "Quercus robur", "angiosperm", 1.2, seed=40 + i) for i in range(6)]
    frames += [_stand(f"P{i}", "Pinus sylvestris", "gymnosperm", 1.0, seed=20 + i) for i in range(6)]
    df = _corpus(frames)

    taxon = _offsets(df)
    goupia = [e for e in taxon.entries if e.key.startswith("goupia") or e.key == "Goupia"]
    for e in goupia:
        assert e.z < 50, f"{e.key} kept an implausible z={e.z}"


@pytest.mark.unit
def test_offsets_are_shrunk_not_taken_at_face_value() -> None:
    df = _consistent_corpus()
    taxon = _offsets(df)
    assert taxon.entries
    for e in taxon.entries:
        # Shrinkage pulls toward zero and never past it, nor away from it.
        assert abs(e.offset) <= abs(e.raw_offset) + 1e-12
        assert e.offset * e.raw_offset >= 0


@pytest.mark.unit
def test_species_outranks_genus_when_both_qualify() -> None:
    df = _consistent_corpus()
    taxon = _offsets(df)
    if any(e.level == "species" and e.key == "larix decidua" for e in taxon.entries):
        assert taxon.lookup("Larix decidua")[1] == "species"


@pytest.mark.unit
def test_unknown_taxon_falls_back_to_the_unrefined_model() -> None:
    """Coverage must never drop: an unseen species is predicted as before."""
    taxon = _offsets(_consistent_corpus())
    for unknown in ("Wollemia nobilis", "", None, "Fabaceae"):
        assert taxon.lookup(unknown) == (0.0, "functional_type")


@pytest.mark.unit
def test_a_stricter_gate_never_earns_more_offsets() -> None:
    df = _consistent_corpus()
    lenient = _offsets(df, z_min=1.0)
    strict = _offsets(df, z_min=6.0)
    assert len(strict.entries) <= len(lenient.entries)


@pytest.mark.unit
def test_smaller_tau2_shrinks_harder() -> None:
    df = _consistent_corpus()
    weak = _offsets(df, tau2=0.001)
    strong = _offsets(df, tau2=1.0)
    weak_map = {e.key: e.offset for e in weak.entries}
    for e in strong.entries:
        if e.key in weak_map:
            assert abs(weak_map[e.key]) <= abs(e.offset) + 1e-12


@pytest.mark.unit
def test_invalid_parameters_are_refused() -> None:
    df = _consistent_corpus()
    with pytest.raises(ValueError):
        _offsets(df, tau2=0.0)


@pytest.mark.unit
def test_report_is_json_safe_and_self_describing() -> None:
    import json

    report = _offsets(_consistent_corpus()).to_report()
    json.dumps(report)
    assert report["z_min"] > 0 and report["tau2"] > 0
    for entry in report["offsets"]:
        assert {"level", "taxon", "offset_log", "z", "n_trees", "n_sites"} <= set(entry)
        assert entry["n_sites"] >= 2


@pytest.mark.unit
def test_normalisers_are_case_and_whitespace_insensitive() -> None:
    assert normalise_species("  Pinus   Sylvestris ") == "pinus sylvestris"
    assert normalise_genus("pinus sylvestris") == "Pinus"
    assert normalise_species(None) == ""
    assert normalise_genus("Fabaceae") == ""


@pytest.mark.unit
def test_empty_input_yields_no_offsets() -> None:
    df = _consistent_corpus()
    empty = df.iloc[0:0]
    fit = _fit(df)
    taxon = fit_taxon_offsets(empty, fit, GROUP, DBH, ASW, SITE, SPECIES)
    assert taxon.entries == ()
    assert taxon.lookup("Larix decidua") == (0.0, "functional_type")

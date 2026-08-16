"""Which site takes which route into the combined corpus.

The one rule that matters: a measured sapwood series always beats a derived one,
so a site present at both scales must never be converted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.sapwood_combined.membership import (
    EXTERNAL,
    INTERNAL,
    ROUTE_CONVERTED,
    ROUTE_NATIVE,
    Corpus,
    resolve_routes,
)


def _corpora(root: Path):
    return (
        Corpus(origin=EXTERNAL, plant_dir=root / "csv/plant", sapwood_dir=root / "csv/sapwood"),
        Corpus(origin=INTERNAL, plant_dir=root / "internal/plant", sapwood_dir=root / "internal/sapwood"),
    )


def _routes(root: Path):
    return resolve_routes(
        _corpora(root),
        plant_asw_dir=root / "csv/plant_asw_filled",
        sapwood_asw_dir=root / "csv/sapwood_asw_filled",
    )


@pytest.mark.unit
def test_every_site_is_routed_exactly_once(corpus: Path) -> None:
    routes = _routes(corpus)
    sites = [r.site for r in routes]
    assert sorted(sites) == ["EXT_BOTH", "EXT_PLANT", "EXT_SAP", "INT_PLANT", "INT_SAP"]
    assert len(sites) == len(set(sites)), "a site was routed twice"


@pytest.mark.unit
def test_a_site_at_both_scales_takes_the_native_sapwood_copy(corpus: Path) -> None:
    """The measured series wins; converting it would discard a measurement."""
    both = next(r for r in _routes(corpus) if r.site == "EXT_BOTH")
    assert both.route == ROUTE_NATIVE
    assert both.source_dir == corpus / "csv/sapwood"


@pytest.mark.unit
def test_plant_only_sites_are_converted_and_sapwood_only_are_native(corpus: Path) -> None:
    by_site = {r.site: r for r in _routes(corpus)}
    assert by_site["EXT_PLANT"].route == ROUTE_CONVERTED
    assert by_site["INT_PLANT"].route == ROUTE_CONVERTED
    assert by_site["EXT_SAP"].route == ROUTE_NATIVE
    assert by_site["INT_SAP"].route == ROUTE_NATIVE


@pytest.mark.unit
def test_the_two_routes_are_disjoint(corpus: Path) -> None:
    routes = _routes(corpus)
    native = {r.site for r in routes if r.route == ROUTE_NATIVE}
    converted = {r.site for r in routes if r.route == ROUTE_CONVERTED}
    assert not (native & converted)


@pytest.mark.unit
def test_origin_is_carried_through(corpus: Path) -> None:
    by_site = {r.site: r.origin for r in _routes(corpus)}
    assert by_site["EXT_PLANT"] == EXTERNAL
    assert by_site["INT_PLANT"] == INTERNAL
    assert by_site["INT_SAP"] == INTERNAL


@pytest.mark.unit
def test_each_route_points_at_the_filled_metadata_for_its_own_scale(corpus: Path) -> None:
    """A converted site needs the plant-scale ASW; a native one the sapwood-scale copy."""
    by_site = {r.site: r for r in _routes(corpus)}
    assert by_site["EXT_PLANT"].asw_md_path == corpus / "csv/plant_asw_filled/EXT_PLANT_plant_md.csv"
    assert by_site["EXT_SAP"].asw_md_path == corpus / "csv/sapwood_asw_filled/EXT_SAP_plant_md.csv"


@pytest.mark.unit
def test_a_filename_collision_between_corpora_is_refused(corpus: Path, tmp_path: Path) -> None:
    """Both corpora land in one flat directory, so a shared code would overwrite."""
    from src.sapwood_combined.tests.conftest import write_site

    write_site(corpus / "internal/plant", "EXT_SAP", ["EXT_SAP_Psy_Jt_1"])
    with pytest.raises(ValueError, match="EXT_SAP"):
        _routes(corpus)


@pytest.mark.unit
def test_routing_survives_a_case_difference_between_the_two_flavours(corpus: Path) -> None:
    """A measured series must not be replaced by a derived one over a spelling.

    If ``plant/`` spells a site differently from ``sapwood/``, an exact-match
    route test fails to see the native copy and converts the site instead --
    silently swapping a measurement for a quotient of an imputed area. This
    repository has had site-keyed lookups miss on a case change twice before.
    """
    from src.sapwood_combined.tests.conftest import write_site

    # Same stand, spelled lower case in the plant flavour only.
    write_site(corpus / "csv/plant", "ext_sap", ["EXT_SAP_Psy_Jt_1"])
    write_site(corpus / "csv/plant_asw_filled", "ext_sap", ["EXT_SAP_Psy_Jt_1"], kinds=("plant_md",))

    routes = {r.site: r for r in _routes(corpus)}
    assert "ext_sap" not in routes, "the lower-case spelling was routed as a second site"
    assert routes["EXT_SAP"].route == ROUTE_NATIVE
    assert routes["EXT_SAP"].source_dir == corpus / "csv/sapwood"


@pytest.mark.unit
def test_missing_scale_directory_is_skipped_not_fatal(tmp_path: Path) -> None:
    """A checkout without the internal corpus must still assemble the archive."""
    from src.sapwood_combined.tests.conftest import write_site

    write_site(tmp_path / "csv/sapwood", "EXT_SAP", ["EXT_SAP_Psy_Jt_1"], units="cm3 cm-2 h-1")
    write_site(tmp_path / "csv/sapwood_asw_filled", "EXT_SAP", ["EXT_SAP_Psy_Jt_1"], kinds=("plant_md",))
    routes = resolve_routes(
        _corpora(tmp_path),
        plant_asw_dir=tmp_path / "csv/plant_asw_filled",
        sapwood_asw_dir=tmp_path / "csv/sapwood_asw_filled",
    )
    assert [r.site for r in routes] == ["EXT_SAP"]


@pytest.mark.unit
def test_a_converted_site_without_filled_asw_is_refused(corpus: Path) -> None:
    """Silently skipping it would drop a site from the product with no trace."""
    (corpus / "csv/plant_asw_filled/EXT_PLANT_plant_md.csv").unlink()
    with pytest.raises(FileNotFoundError, match="EXT_PLANT"):
        _routes(corpus)

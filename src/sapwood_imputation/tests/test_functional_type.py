"""Tests for genus -> angiosperm/gymnosperm classification."""

from __future__ import annotations

import pytest
from src.sapwood_imputation.functional_type import (
    ANGIOSPERM,
    GYMNOSPERM,
    UNKNOWN,
    classify,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "species",
    [
        "Pinus sylvestris",
        "Picea abies",
        "Larix decidua",
        "Abies alba",
        "Pseudotsuga menziesii",
        "Juniperus thurifera",
        "Taxus baccata",
        "Thuja occidentalis",
        "Tsuga canadensis",
        "Cryptomeria japonica",
        "Agathis australis",
    ],
)
def test_known_gymnosperms(species: str) -> None:
    assert classify(species) == GYMNOSPERM


@pytest.mark.unit
@pytest.mark.parametrize(
    "species",
    [
        "Quercus ilex",
        "Fagus sylvatica",
        "Eucalyptus globulus",
        "Acer pseudoplatanus",
        "Populus tremula",
        "Olea europaea",
        "Liquidambar styraciflua",
        "Nothofagus pumilio",
    ],
)
def test_known_angiosperms(species: str) -> None:
    assert classify(species) == ANGIOSPERM


@pytest.mark.unit
def test_case_insensitive() -> None:
    assert classify("pinus SYLVESTRIS") == GYMNOSPERM
    assert classify("QUERCUS robur") == ANGIOSPERM


@pytest.mark.unit
def test_genus_only_string() -> None:
    assert classify("Pinus") == GYMNOSPERM
    assert classify("Quercus") == ANGIOSPERM


@pytest.mark.unit
def test_leading_trailing_whitespace() -> None:
    assert classify("  Pinus sylvestris  ") == GYMNOSPERM


@pytest.mark.unit
@pytest.mark.parametrize("value", ["", "   ", None])
def test_blank_or_none_is_unknown(value) -> None:
    assert classify(value) == UNKNOWN


@pytest.mark.unit
def test_family_name_is_unknown() -> None:
    # A family-level name (ends in -aceae) is not a genus -> cannot resolve.
    assert classify("Myrtaceae") == UNKNOWN

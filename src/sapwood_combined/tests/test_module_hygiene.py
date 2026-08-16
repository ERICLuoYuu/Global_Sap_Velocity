"""Structural properties of the package's own source, checked by parsing it.

One test, for one failure mode that has bitten this repository repeatedly and
that no behavioural test can see: a module defining the same top-level name
twice. The second definition wins silently, the first becomes unreachable, and
every test still passes -- because the surviving definition is usually the
intended one. The evidence that something is wrong exists only in the source.

This is not hypothetical here. ``manifest.py`` shipped with ``_corpora_block``
defined twice; the dead copy sat in the released module through a full review
cycle, three rounds of tests and 281 passing assertions. Ruff's F811 would have
caught it, but ruff is not installed in the canonical Palma environment, so the
lint that guards this never ran where the code is actually built.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1]


def _redefinitions(source: str) -> list[tuple[str, int, int]]:
    """Top-level names bound twice by a def/class, as (name, first, second)."""
    seen: dict[str, int] = {}
    found = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in seen:
                found.append((node.name, seen[node.name], node.lineno))
            seen[node.name] = node.lineno
    return found


@pytest.mark.unit
@pytest.mark.parametrize("module", sorted(PACKAGE.glob("*.py")), ids=lambda p: p.name)
def test_no_module_defines_the_same_name_twice(module: Path) -> None:
    clashes = _redefinitions(module.read_text(encoding="utf-8"))
    assert not clashes, "\n".join(
        f"{module.name}: {name!r} defined at line {first}, silently replaced at line {second}"
        for name, first, second in clashes
    )


@pytest.mark.unit
def test_the_detector_would_actually_notice() -> None:
    """A guard nobody has seen fail is a guard nobody knows works."""
    assert _redefinitions("def f():\n    pass\n\n\ndef f():\n    pass\n") == [("f", 1, 5)]
    assert _redefinitions("def f():\n    pass\n\n\ndef g():\n    pass\n") == []

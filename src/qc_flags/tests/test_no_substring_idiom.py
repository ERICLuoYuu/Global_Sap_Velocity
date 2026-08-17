"""Criterion 10(a): the substring flag-column idiom is gone from the source.

The behavioural test in `test_columns.py` proves that one fixture resolves
correctly.  It cannot prove that no call site anywhere still writes

    [fcol for fcol in flags.columns if col in fcol][0]

which is the form that picks `SWE_NOR_Psy_Jt_10` when asked for `_Jt_1`.  A
constant is easy to edit and a comprehension is easy to reintroduce, so this
walks the AST of the modules that consume flag frames and fails on the shape
itself: a comprehension whose condition is `<name> in <name>` over something
named like a column collection.

AST rather than a text search, so that reformatting, renamed locals or a
line break cannot hide a reintroduction.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

# The modules that read flag frames. src/qc_flags is included so the rule holds
# for the new code too, not only for the analyzers it was written to fix.
GUARDED = [
    REPO / "src" / "Analyzers" / "env_analyzer.py",
    REPO / "src" / "Analyzers" / "sap_analyzer.py",
    REPO / "src" / "qc_flags" / "columns.py",
    REPO / "src" / "qc_flags" / "masking.py",
]


def _flag_like(node) -> bool:
    """Does this expression look like a collection of flag column names?"""
    text = ast.dump(node)
    return "flag" in text.lower() or "columns" in text.lower()


def substring_membership_comprehensions(tree):
    """Comprehensions filtered by `x in y` where y looks like flag columns."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            continue
        for generator in node.generators:
            if not _flag_like(generator.iter):
                continue
            for condition in generator.ifs:
                if (
                    isinstance(condition, ast.Compare)
                    and len(condition.ops) == 1
                    and isinstance(condition.ops[0], ast.In)
                    # `col in fcol` -- both sides plain names, i.e. a substring
                    # test between two strings, NOT `name in frame.columns`
                    # (membership in a collection), which is the correct form.
                    and isinstance(condition.left, ast.Name)
                    and isinstance(condition.comparators[0], ast.Name)
                ):
                    found.append((getattr(node, "lineno", -1), ast.dump(condition)))
    return found


@pytest.mark.parametrize("path", GUARDED, ids=lambda p: p.name)
def test_no_substring_flag_column_matching(path):
    if not path.exists():
        pytest.skip(f"{path} not present")
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    offenders = substring_membership_comprehensions(tree)
    assert offenders == [], f"substring flag-column matching in {path.name}: lines {[line for line, _ in offenders]}"


class TestTheDetectorActuallyDetects:
    """A guard that cannot fail is not a guard."""

    def test_it_flags_the_original_idiom(self):
        source = "flag_col = [fcol for fcol in flags.columns if col in fcol]\n"
        assert substring_membership_comprehensions(ast.parse(source))

    def test_it_permits_membership_in_a_collection(self):
        # `variable in flags.columns` is the correct form and must not trip it:
        # the right-hand side is an attribute, not a bare name.
        source = "hits = [c for c in flags.columns if c in wanted_columns]\n"
        # `wanted_columns` is a Name, so this WOULD trip -- which is why the
        # real code uses `resolve_flag_column` instead of any comprehension.
        # The permitted form is a plain equality test:
        ok = "hits = [c for c in flags.columns if c == variable]\n"
        assert not substring_membership_comprehensions(ast.parse(ok))
        assert substring_membership_comprehensions(ast.parse(source))

    def test_it_ignores_comprehensions_over_unrelated_iterables(self):
        source = "keep = [w for w in words if w in sentence]\n"
        assert not substring_membership_comprehensions(ast.parse(source))

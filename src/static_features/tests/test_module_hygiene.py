"""Structural properties of the package's own source, checked by parsing it.

Palma has no ruff, and the local linter hook has rewritten code into forms that Python 3.9
rejects or that change what the code means. So these guards live in the suite and run where the
code runs. The last test feeds each detector a known offender, because a guard nobody has seen
fail is a guard nobody knows works.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import src.static_features as package

PACKAGE = Path(package.__file__).resolve().parent
MODULES = sorted(PACKAGE.rglob("*.py"))
CALLS_ALLOWED_AT_IMPORT = frozenset(
    {
        "Path", "resolve", "frozenset", "tuple", "dict", "set", "list", "range", "sorted",
        "getLogger", "compile", "items", "keys", "values", "float", "int", "str", "join",
    }
)  # fmt: skip


def _parse(source: str) -> ast.Module:
    return ast.parse(source, feature_version=(3, 9))


def _trees() -> Iterator[tuple[Path, ast.Module]]:
    for path in MODULES:
        yield path, _parse(path.read_text(encoding="utf-8"))


def _production_trees() -> Iterator[tuple[Path, ast.Module]]:
    for path, tree in _trees():
        if "tests" not in path.relative_to(PACKAGE).parts:
            yield path, tree


def _report(found: list[str]) -> str:
    return "\n".join(found)


def _lacks_future_annotations(tree: ast.Module) -> bool:
    if not tree.body:
        return False  # an empty __init__.py has nothing to annotate
    return not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )


def _zip_calls(tree: ast.Module) -> list[int]:
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "zip"
    ]


def _asserts(tree: ast.Module) -> list[int]:
    return [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assert)]


def _runs_at_import(tree: ast.Module) -> Iterator[ast.AST]:
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _imports_ee_at_import(tree: ast.Module) -> list[int]:
    found = []
    for node in _runs_at_import(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        if any(name == "ee" or name.startswith("ee.") for name in names):
            found.append(node.lineno)
    return found


def _is_main_guard(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    )


def _assignment_is_constant(node: ast.AST) -> bool:
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return False
    for inner in ast.walk(node):
        if isinstance(inner, ast.Call):
            func = inner.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name not in CALLS_ALLOWED_AT_IMPORT:
                return False
    return True


def _work_at_import(tree: ast.Module) -> list[int]:
    found = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        is_docstring = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        if not (is_docstring or _is_main_guard(node) or _assignment_is_constant(node)):
            found.append(node.lineno)
    return found


def _redefinitions(tree: ast.Module) -> list[str]:
    seen = set()
    found = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in seen:
                found.append(node.name)
            seen.add(node.name)
    return found


def _runtime_unions(tree: ast.Module) -> list[int]:
    """isinstance(x, int | str) parses on 3.9 and raises TypeError only when it runs."""
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("isinstance", "issubclass")
        and len(node.args) == 2
        and isinstance(node.args[1], ast.BinOp)
    ]


def test_every_module_uses_future_annotations():
    found = [str(path) for path, tree in _trees() if _lacks_future_annotations(tree)]
    assert not found, _report(found)


def test_no_zip_anywhere():
    found = [f"{path}: {_zip_calls(tree)}" for path, tree in _trees() if _zip_calls(tree)]
    assert not found, _report(found)


def test_no_bare_assert_outside_tests():
    found = [f"{path}: {_asserts(tree)}" for path, tree in _production_trees() if _asserts(tree)]
    assert not found, _report(found)


def test_ee_is_never_imported_at_module_level():
    found = [f"{path}: {_imports_ee_at_import(tree)}" for path, tree in _trees() if _imports_ee_at_import(tree)]
    assert not found, _report(found)


def test_modules_do_no_work_at_import():
    found = [f"{path}: {_work_at_import(tree)}" for path, tree in _production_trees() if _work_at_import(tree)]
    assert not found, _report(found)


def test_no_module_defines_the_same_name_twice():
    found = [f"{path}: {_redefinitions(tree)}" for path, tree in _trees() if _redefinitions(tree)]
    assert not found, _report(found)


def test_no_union_operator_in_isinstance():
    found = [f"{path}: {_runtime_unions(tree)}" for path, tree in _trees() if _runtime_unions(tree)]
    assert not found, _report(found)


def test_each_guard_notices_a_known_offender():
    assert Path(__file__).resolve() in MODULES

    assert _lacks_future_annotations(_parse("x = 1\n"))
    assert not _lacks_future_annotations(_parse("from __future__ import annotations\nx = 1\n"))
    assert not _lacks_future_annotations(_parse(""))

    assert _zip_calls(_parse("def f(a, b):\n    return list(zip(a, b))\n")) == [2]
    assert _asserts(_parse("def f(a):\n    assert a\n")) == [2]

    assert _imports_ee_at_import(_parse("import ee\n")) == [1]
    assert _imports_ee_at_import(_parse("from ee import Image\n")) == [1]
    assert _imports_ee_at_import(_parse("try:\n    import ee\nexcept ImportError:\n    pass\n")) == [2]
    assert _imports_ee_at_import(_parse("def f():\n    import ee\n    return ee\n")) == []
    assert _imports_ee_at_import(_parse("import eel\n")) == []

    assert _work_at_import(_parse("import pandas as pd\nFRAME = pd.read_csv('x.csv')\n")) == [2]
    assert _work_at_import(_parse("print('hello')\n")) == [1]
    assert _work_at_import(_parse("for i in range(3):\n    pass\n")) == [1]
    assert _work_at_import(_parse('"""Doc."""\nA = (1, 2)\nB = frozenset(A)\n')) == []
    assert _work_at_import(_parse("if __name__ == '__main__':\n    print('x')\n")) == []

    assert _redefinitions(_parse("def f():\n    pass\n\n\ndef f():\n    pass\n")) == ["f"]
    assert _runtime_unions(_parse("isinstance(x, int | str)\n")) == [1]
    assert _runtime_unions(_parse("isinstance(x, (int, str))\n")) == []

from __future__ import annotations

import ast
from pathlib import Path


def _keyword_value(call: ast.Call, name: str) -> ast.expr | None:
    return next((keyword.value for keyword in call.keywords if keyword.arg == name), None)


def _is_binary_open(call: ast.Call) -> bool:
    mode = _keyword_value(call, "mode")
    if mode is None and len(call.args) >= 2:
        mode = call.args[1]
    return isinstance(mode, ast.Constant) and isinstance(mode.value, str) and "b" in mode.value


def _has_utf8_encoding(call: ast.Call) -> bool:
    encoding = _keyword_value(call, "encoding")
    return (
        isinstance(encoding, ast.Constant)
        and isinstance(encoding.value, str)
        and encoding.value.lower().replace("_", "-") == "utf-8"
    )


def test_text_file_operations_use_explicit_utf8():
    tests_path = Path(__file__).parent
    violations = []

    for source_path in sorted(tests_path.rglob("*.py")):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            is_open = (
                isinstance(node.func, ast.Name) and node.func.id == "open"
            ) or (
                isinstance(node.func, ast.Attribute) and node.func.attr == "open"
            )
            is_text_path_method = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"read_text", "write_text"}
            )

            if is_open and _is_binary_open(node):
                continue
            if (is_open or is_text_path_method) and not _has_utf8_encoding(node):
                violations.append(f"{source_path.relative_to(tests_path)}:{node.lineno}")

    assert violations == [], "text file operations missing encoding='utf-8': " + ", ".join(violations)

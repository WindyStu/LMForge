from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
from pathlib import Path


def test_installed_lmforge_package_is_used() -> None:
    package = importlib.import_module("lmforge")
    distribution = importlib.metadata.distribution("lmforge")

    assert distribution.metadata["Name"] == "lmforge"
    assert Path(package.__file__).resolve() == (
        Path(__file__).parents[1] / "src" / "lmforge" / "__init__.py"
    ).resolve()


def test_legacy_cs336_basics_namespace_is_removed() -> None:
    assert importlib.util.find_spec("cs336_basics") is None

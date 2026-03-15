"""Tests that packaging metadata stays aligned with documented extras."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


def _load_pyproject() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    pyproject_path = root / "pyproject.toml"
    with pyproject_path.open("rb") as fh:
        return tomllib.load(fh)


def test_pyproject_defines_vision_extra() -> None:
    pyproject = _load_pyproject()
    optional_deps = pyproject["project"]["optional-dependencies"]

    assert "vision" in optional_deps
    assert "pillow>=12.1.1" in optional_deps["vision"]


def test_vision_dependencies_are_available_in_test_environments() -> None:
    pyproject = _load_pyproject()

    assert "pillow>=12.1.1" in pyproject["dependency-groups"]["test"]
    assert "pillow>=12.1.1" in pyproject["tool"]["cibuildwheel"]["test-requires"]

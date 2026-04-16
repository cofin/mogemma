"""Tests that tools/*.py default model IDs match the gs://gemma-data catalog.

The tools scripts (``smoke_test.py``, ``run_parity_gates.py``, ``validate.py``)
historically carried Gemma 3 / Gemma 3n defaults that were left behind by the
hub → GCS migration. This test guards against regression by asserting every
tool-level default resolves to an entry in :data:`mogemma.hub.KNOWN_GCS_MODELS`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from mogemma.hub import KNOWN_GCS_MODELS

if TYPE_CHECKING:
    from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TOOLS_DIR = _REPO_ROOT / "tools"


def _load_tool(name: str) -> ModuleType:
    """Load a ``tools/<name>.py`` script as a standalone module by file path."""
    module_name = f"_mogemma_tools_under_test_{name}"
    spec = importlib.util.spec_from_file_location(module_name, _TOOLS_DIR / f"{name}.py")
    assert spec is not None, f"Cannot locate tools/{name}.py"
    assert spec.loader is not None, f"tools/{name}.py has no loader"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class TestSmokeTestDefaults:
    def test_default_model_id_in_catalog(self) -> None:
        mod = _load_tool("smoke_test")
        assert mod.DEFAULT_MODEL_ID in KNOWN_GCS_MODELS


class TestRunParityGatesDefaults:
    def test_default_model_id_in_catalog(self) -> None:
        mod = _load_tool("run_parity_gates")
        assert mod.DEFAULT_MODEL_ID in KNOWN_GCS_MODELS


class TestValidateConstants:
    def test_text_model_id_in_catalog(self) -> None:
        mod = _load_tool("validate")
        assert mod.TEXT_MODEL_ID in KNOWN_GCS_MODELS

    def test_embed_model_id_in_catalog(self) -> None:
        mod = _load_tool("validate")
        assert mod.EMBED_MODEL_ID in KNOWN_GCS_MODELS

    def test_nano_model_id_in_catalog(self) -> None:
        mod = _load_tool("validate")
        assert mod.NANO_MODEL_ID in KNOWN_GCS_MODELS

    def test_no_gemma3n_branch_in_semantic_quality(self) -> None:
        """Guard against re-adding the gemma3n early-return in semantic quality.

        The old ``if 'gemma3n' in model_id: return`` branch skipped semantic
        validation for a model we no longer ship. It should not come back.
        """
        validate_source = (_TOOLS_DIR / "validate.py").read_text()
        assert "gemma3n" not in validate_source

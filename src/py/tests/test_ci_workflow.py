"""Smoke checks against the GitHub Actions workflow YAML.

Not a replacement for running the workflows themselves — just a quick regex-
level guard that catches obvious references to retired model IDs that would
fail at runtime if the workflow ever fires.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKFLOWS_DIR = _REPO_ROOT / ".github" / "workflows"


@pytest.fixture
def ci_yml_text() -> str:
    return (_WORKFLOWS_DIR / "ci.yml").read_text()


class TestNoStaleGemmaReferences:
    """CI workflow must not reference Gemma generations retired from the bucket.

    A 2026-04-16 probe confirmed ``gs://gemma-data`` only ships Gemma 4 model
    prefixes (``gemma4-e2b-it``, ``gemma4-e4b-it``, ``gemma4-26b-a4b-it``).
    Any ``gemma3-*`` or ``gemma3n-*`` reference in a workflow step is a
    latent failure waiting to fire.
    """

    _STALE_PATTERN = re.compile(r"gemma3[n-]|gemma3-")

    def test_no_stale_ids_in_ci_yml(self, ci_yml_text: str) -> None:
        matches = self._STALE_PATTERN.findall(ci_yml_text)
        assert not matches, f"Stale Gemma 3/3n refs remain in ci.yml: {matches}"


class TestCheckReleaseStructure:
    """Guardrails on the ``check-release`` job's shape.

    If the parity step is ever re-added, it must not point at a stale model
    or a Linux runner that cannot hold the checkpoint.
    """

    def test_parity_gate_step_is_absent_until_gpu_ci_modal_lands(self, ci_yml_text: str) -> None:
        """Parity step must stay removed until Modal runner lands.

        Linux CI runners lack the disk (~14 GB) to hold even the smallest
        working model (~36 GB after Orbax→safetensors conversion) and lack a
        GPU for real-weight parity. The step is intentionally removed until
        the ``gpu-ci-modal`` follow-on flow provides a Modal-backed runner.
        """
        assert "run_parity_gates.py" not in ci_yml_text, (
            "tools/run_parity_gates.py re-appeared in ci.yml — the gpu-ci-modal "
            "flow must land the Modal runner before re-enabling this step."
        )

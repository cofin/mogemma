"""Integration smoke tests for repository tools and workflow entry points."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mogemma.hub import KNOWN_GCS_MODELS

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parents[4]
TOOLS_DIR = ROOT / "tools"
CI_WORKFLOW_PATH = ROOT / ".github" / "workflows" / "ci.yml"
PERF_WORKFLOW_PATH = ROOT / ".github" / "workflows" / "perf-benchmark.yml"


def _load_tool(name: str) -> ModuleType:
    """Load a tools/<name>.py script as a standalone module by file path."""
    module_name = f"_mogemma_tools_under_test_{name}"
    spec = importlib.util.spec_from_file_location(module_name, TOOLS_DIR / f"{name}.py")
    assert spec is not None, f"Cannot locate tools/{name}.py"
    assert spec.loader is not None, f"tools/{name}.py has no loader"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("workflow_path", [CI_WORKFLOW_PATH, PERF_WORKFLOW_PATH])
def test_workflow_file_exists(workflow_path: Path) -> None:
    assert workflow_path.exists(), f"Missing {workflow_path}"


@pytest.mark.parametrize("snippet", ["push:", "- main", "pull_request:", "make benchmark", "upload-artifact"])
def test_perf_benchmark_workflow_shape(snippet: str) -> None:
    assert snippet in PERF_WORKFLOW_PATH.read_text()


@pytest.mark.parametrize(
    ("tool_name", "attribute"),
    [
        ("smoke_test", "DEFAULT_MODEL_ID"),
        ("run_parity_gates", "DEFAULT_MODEL_ID"),
        ("validate", "TEXT_MODEL_ID"),
        ("validate", "EMBED_MODEL_ID"),
        ("validate", "NANO_MODEL_ID"),
    ],
)
def test_tool_default_model_ids_are_in_gcs_catalog(tool_name: str, attribute: str) -> None:
    module = _load_tool(tool_name)
    assert getattr(module, attribute) in KNOWN_GCS_MODELS


def test_benchmark_script_runs_synthetic(tmp_path: Path) -> None:
    """Smoke-test that the benchmark entry point runs with stubs."""
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "benchmark.py"),
            "--mode",
            "generation",
            "--rounds",
            "2",
            "--warmup",
            "1",
            "--max-new-tokens",
            "8",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(tmp_path),
        check=False,
    )
    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"
    payload = json.loads(result.stdout)
    assert payload["mode"] == "generation"
    assert payload["is_synthetic"] is True

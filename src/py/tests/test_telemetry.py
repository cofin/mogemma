import importlib
import json
import struct
import sys
from pathlib import Path

import pytest


def _create_dummy_safetensors(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "model.safetensors").open("wb") as f:
        h = json.dumps({}).encode("utf-8")
        f.write(struct.pack("<Q", len(h)) + h)


def test_import_telemetry_has_no_global_provider_side_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing telemetry should not mutate the global tracer provider."""
    module_name = "mogemma.telemetry"
    sys.modules.pop(module_name, None)

    calls: list[object] = []

    def _set_provider(provider: object) -> None:
        calls.append(provider)

    monkeypatch.setattr("opentelemetry.trace.set_tracer_provider", _set_provider)

    importlib.import_module(module_name)
    assert calls == []

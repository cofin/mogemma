import importlib
import sys

import pytest


def test_import_telemetry_has_no_global_provider_side_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing telemetry should not mutate the global tracer provider."""
    module_name = "mogemma.telemetry"
    sys.modules.pop(module_name, None)

    calls: list[object] = []
    monkeypatch.setattr("opentelemetry.trace.set_tracer_provider", lambda provider: calls.append(provider))

    importlib.import_module(module_name)
    assert calls == []

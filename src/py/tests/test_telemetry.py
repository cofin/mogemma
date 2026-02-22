import importlib
import sys

import pytest


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

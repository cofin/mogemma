from __future__ import annotations

import pytest

from mogemma.backends import resolve_backend_id


def test_resolve_backend_id_defaults_to_cpu_mojo() -> None:
    assert resolve_backend_id("") == "cpu_mojo"
    assert resolve_backend_id("   ") == "cpu_mojo"


def test_resolve_backend_id_normalizes_known_aliases() -> None:
    assert resolve_backend_id("cpu") == "cpu_mojo"
    assert resolve_backend_id("CPU_MOJO") == "cpu_mojo"
    assert resolve_backend_id("gpu") == "gpu_mojo"
    assert resolve_backend_id("GPU_MOJO") == "gpu_mojo"


def test_resolve_backend_id_accepts_gpu_device_selector() -> None:
    assert resolve_backend_id("gpu:0") == "gpu_mojo"
    assert resolve_backend_id("gpu:17") == "gpu_mojo"


def test_resolve_backend_id_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="Unsupported backend"):
        resolve_backend_id("banana")

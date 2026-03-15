"""Tests for Pillow-backed image hydration contracts."""

from __future__ import annotations

import numpy as np
import pytest

import mogemma.hydration as hydration_module
from mogemma.hydration import ImageHydrator


def _block_pillow_import(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import_module = hydration_module.importlib.import_module
    missing_pillow = "No module named 'PIL'"

    def fake_import_module(name: str) -> object:
        if name == "PIL.Image":
            raise ModuleNotFoundError(missing_pillow)
        return original_import_module(name)

    monkeypatch.setattr(hydration_module.importlib, "import_module", fake_import_module)


def test_hydrate_bytes_requires_vision_extra_without_pillow(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_pillow_import(monkeypatch)

    hydrator = ImageHydrator()

    with pytest.raises(ImportError, match=r"mogemma\[vision\]"):
        hydrator.hydrate([b"not-a-real-image"])


def test_hydrate_ndarray_bypasses_pillow_import(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_pillow_import(monkeypatch)

    image = np.zeros((4, 5, 3), dtype=np.uint8)
    hydrated = ImageHydrator().hydrate([image])

    assert len(hydrated) == 1
    assert hydrated[0].dtype == np.uint8
    assert hydrated[0].shape == image.shape
    assert np.array_equal(hydrated[0], image)

"""Integration tests for automatic image hydration in the vision API."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import obstore as obs
import pytest
from obstore.store import LocalStore
from PIL import Image

from mogemma import GenerationConfig, SyncGemmaModel

_core = pytest.importorskip("mogemma._core", exc_type=ImportError)

if TYPE_CHECKING:
    from pathlib import Path


class _Encoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _StubTokenizer:
    def enable_truncation(self, **_: object) -> None:
        return None

    def enable_padding(self, **_: object) -> None:
        return None

    def encode(self, _text: str) -> _Encoding:
        return _Encoding([2, 3])

    def decode(self, _tokens: list[int]) -> str:
        return ""

    def token_to_id(self, _token: str) -> int | None:
        return 1


def _mock_llm_state() -> dict[str, int | str]:
    return {
        "_arena_ptr": 0,
        "_arena_size": 0,
        "pos": 0,
        "max_seq_len": 100,
        "num_layers": 1,
        "num_heads": 1,
        "num_kv_heads": 1,
        "head_dim": 128,
        "intermediate_size": 256,
        "kv_share_start": 0,
        "step_backend": "cpu",
        "arch": "nano",
        "k_cache": 0,
        "v_cache": 0,
        "session_kv_cache_len": 0,
        "debug_launch_count": 0,
        "freqs_cos": 0,
        "freqs_sin": 0,
        "step_scratch_len": 0,
    }


def _build_model(tmp_path: Path) -> SyncGemmaModel:
    config = GenerationConfig(model_path=tmp_path, device="cpu", temperature=0.0, top_k=1)
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "tokenizer.model").touch()

    with (
        patch("mogemma.model._initialize_llm", return_value=_mock_llm_state()),
        patch("mogemma.model.auto_loader", return_value=MagicMock()),
    ):
        return SyncGemmaModel(config, tokenizer=_StubTokenizer())


def _assert_hydrated_image(
    model: SyncGemmaModel,
    images: list[str | bytes],
    expected_shape: tuple[int, int, int],
    expected_rgb: tuple[int, int, int],
) -> None:
    with (
        patch.object(model._backend, "process_images") as mock_process,
        patch.object(model._backend, "step", return_value=np.asarray([0.0, 1.0], dtype=np.float32)),
    ):
        list(model.generate_stream("Describe", images=images))

    assert mock_process.called
    args, _ = mock_process.call_args
    hydrated_images = args[1]
    assert isinstance(hydrated_images[0], np.ndarray)
    assert hydrated_images[0].shape == expected_shape
    assert hydrated_images[0].dtype == np.uint8
    assert np.all(hydrated_images[0][0, 0] == list(expected_rgb))


def test_vision_e2e_hydration(tmp_path: Path) -> None:
    """Verify that path inputs are hydrated into uint8 RGB arrays."""
    img_path = tmp_path / "test_image.png"
    Image.new("RGB", (100, 100), color=(255, 0, 0)).save(img_path)

    store = LocalStore()
    obs.head(store, str(img_path))

    model = _build_model(tmp_path)
    _assert_hydrated_image(model, [str(img_path)], (100, 100, 3), (255, 0, 0))


def test_vision_e2e_hydrates_raw_bytes(tmp_path: Path) -> None:
    """Verify that raw image bytes are decoded before reaching the backend."""
    img_path = tmp_path / "test_image.png"
    Image.new("RGB", (64, 32), color=(0, 128, 255)).save(img_path)

    model = _build_model(tmp_path)
    _assert_hydrated_image(model, [img_path.read_bytes()], (32, 64, 3), (0, 128, 255))


def test_vision_mojo_resize_integration() -> None:
    """Reserve coverage for future end-to-end resize validation."""

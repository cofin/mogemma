"""Tests for ``OrbaxLoader`` (Orbax/OCDBT weight reader)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mogemma.orbax_loader import OrbaxLoader, _numpy_dtype_to_safetensors_str

if TYPE_CHECKING:
    from pathlib import Path


class TestCanLoad:
    """OrbaxLoader.can_load detects the OCDBT directory layout."""

    def test_returns_true_for_orbax_layout(self, tmp_path: Path) -> None:
        (tmp_path / "ocdbt.process_0").mkdir()
        (tmp_path / "manifest.ocdbt").write_bytes(b"")
        assert OrbaxLoader.can_load(tmp_path) is True

    def test_requires_process_dir(self, tmp_path: Path) -> None:
        (tmp_path / "manifest.ocdbt").write_bytes(b"")
        assert OrbaxLoader.can_load(tmp_path) is False

    def test_requires_manifest(self, tmp_path: Path) -> None:
        (tmp_path / "ocdbt.process_0").mkdir()
        assert OrbaxLoader.can_load(tmp_path) is False

    def test_returns_false_for_safetensors_dir(self, tmp_path: Path) -> None:
        (tmp_path / "model.safetensors").write_bytes(b"")
        assert OrbaxLoader.can_load(tmp_path) is False

    def test_returns_false_for_file(self, tmp_path: Path) -> None:
        f = tmp_path / "checkpoint.bin"
        f.write_bytes(b"")
        assert OrbaxLoader.can_load(f) is False


class TestNumpyDtypeMapping:
    """Numpy dtype → safetensors FFI dtype string."""

    def test_float32(self) -> None:
        assert _numpy_dtype_to_safetensors_str(np.dtype("float32")) == "F32"

    def test_float16(self) -> None:
        assert _numpy_dtype_to_safetensors_str(np.dtype("float16")) == "F16"

    def test_int32(self) -> None:
        assert _numpy_dtype_to_safetensors_str(np.dtype("int32")) == "I32"


class TestGetTensorMetadata:
    """get_tensor_metadata returns ``(ptr, shape, dtype)`` for every tensor loaded."""

    @pytest.fixture
    def orbax_dir(self, tmp_path: Path) -> Path:
        (tmp_path / "ocdbt.process_0").mkdir()
        (tmp_path / "manifest.ocdbt").write_bytes(b"")
        return tmp_path

    def test_contract_shape_dtype_ptr(self, orbax_dir: Path) -> None:
        """Loader should read every ``.zarray`` key and expose numpy-backed pointers."""
        fake_keys = [b"embed/.zarray", b"embed/0.0", b"layer0/.zarray"]
        embed_arr = np.ones((4, 8), dtype=np.float32)
        layer_arr = np.arange(16, dtype=np.int32).reshape(4, 4)
        arrays = {"embed": embed_arr, "layer0": layer_arr}

        def fake_open_tensor(self: OrbaxLoader, name: str) -> np.ndarray:
            return arrays[name]

        fake_kvstore = MagicMock()
        fake_kvstore.list.return_value.result.return_value = fake_keys
        fake_ts = MagicMock()
        fake_ts.KvStore.open.return_value.result.return_value = fake_kvstore

        with (
            patch.dict("sys.modules", {"tensorstore": fake_ts}),
            patch.object(OrbaxLoader, "_open_tensor", fake_open_tensor),
        ):
            loader = OrbaxLoader(orbax_dir)
            metadata = loader.get_tensor_metadata()

        assert set(metadata.keys()) == {"embed", "layer0"}
        embed_meta = metadata["embed"]
        assert embed_meta[1] == (4, 8)
        assert embed_meta[2] == "F32"
        assert isinstance(embed_meta[0], int)
        assert embed_meta[0] != 0

        layer_meta = metadata["layer0"]
        assert layer_meta[1] == (4, 4)
        assert layer_meta[2] == "I32"

        loader.close()
        # Arrays are dropped after close → metadata dict is empty on re-query.
        assert loader.get_tensor_metadata() == {}

    def test_context_manager_releases(self, orbax_dir: Path) -> None:
        fake_kvstore = MagicMock()
        fake_kvstore.list.return_value.result.return_value = [b"t/.zarray"]
        fake_ts = MagicMock()
        fake_ts.KvStore.open.return_value.result.return_value = fake_kvstore

        def fake_open_tensor(self: OrbaxLoader, name: str) -> np.ndarray:
            return np.zeros((2,), dtype=np.float32)

        with (
            patch.dict("sys.modules", {"tensorstore": fake_ts}),
            patch.object(OrbaxLoader, "_open_tensor", fake_open_tensor),
        ):
            with OrbaxLoader(orbax_dir) as loader:
                assert "t" in loader.get_tensor_metadata()
            # After __exit__ close() was called.
            assert loader.get_tensor_metadata() == {}


class TestStreamingHelpers:
    """Static helpers that read the checkpoint without eagerly loading every tensor."""

    @pytest.fixture
    def orbax_dir(self, tmp_path: Path) -> Path:
        (tmp_path / "ocdbt.process_0").mkdir()
        (tmp_path / "manifest.ocdbt").write_bytes(b"")
        return tmp_path

    def test_enumerate_tensors_does_not_materialize(self, orbax_dir: Path) -> None:
        """enumerate_tensors must not call _open_tensor for any key."""
        fake_kvstore = MagicMock()
        fake_kvstore.list.return_value.result.return_value = [
            b"embedder.input_embedding/.zarray",
            b"embedder.input_embedding/0.0",
            b"layer_0.attn.q_einsum.w/.zarray",
        ]
        fake_ts = MagicMock()
        fake_ts.KvStore.open.return_value.result.return_value = fake_kvstore

        open_tensor_calls: list[str] = []

        def spy_open_tensor(self: OrbaxLoader, name: str) -> np.ndarray:
            open_tensor_calls.append(name)
            return np.zeros((1,), dtype=np.float32)

        with (
            patch.dict("sys.modules", {"tensorstore": fake_ts}),
            patch.object(OrbaxLoader, "_open_tensor", spy_open_tensor),
        ):
            names = OrbaxLoader.enumerate_tensors(orbax_dir)

        assert names == ["embedder.input_embedding", "layer_0.attn.q_einsum.w"]
        assert open_tensor_calls == []

    def test_open_tensor_upcasts_bf16_to_f32(self, orbax_dir: Path) -> None:
        """open_tensor upcasts bfloat16 to float32 (matches FFI contract)."""
        # Build a fake bfloat16 array via a custom numpy dtype shim:
        # ml_dtypes provides bfloat16 if installed; fall back to constructing
        # an object whose `.dtype.name` is "bfloat16".
        fake_bf16 = MagicMock(spec=np.ndarray)
        fake_bf16.dtype = MagicMock()
        fake_bf16.dtype.name = "bfloat16"
        upcast_result = np.ones((3, 4), dtype=np.float32)
        fake_bf16.astype.return_value = upcast_result

        with patch.object(OrbaxLoader, "_open_tensor", return_value=fake_bf16):
            result = OrbaxLoader.open_tensor(orbax_dir, "some_tensor")

        fake_bf16.astype.assert_called_once_with(np.float32)
        assert result is upcast_result
        assert result.dtype == np.float32

    def test_open_tensor_preserves_f32(self, orbax_dir: Path) -> None:
        """F32 tensors pass through unchanged (no unnecessary copy)."""
        f32_arr = np.ones((2, 3), dtype=np.float32)
        with patch.object(OrbaxLoader, "_open_tensor", return_value=f32_arr):
            result = OrbaxLoader.open_tensor(orbax_dir, "some_tensor")
        assert result is f32_arr

    def test_open_tensor_forces_contiguous(self, orbax_dir: Path) -> None:
        """Non-contiguous arrays are re-copied so ctypes.data is valid."""
        base = np.arange(20, dtype=np.float32).reshape(4, 5)
        sliced = base[:, ::2]  # non-contiguous view
        assert not sliced.flags["C_CONTIGUOUS"]
        with patch.object(OrbaxLoader, "_open_tensor", return_value=sliced):
            result = OrbaxLoader.open_tensor(orbax_dir, "some_tensor")
        assert result.flags["C_CONTIGUOUS"]
        np.testing.assert_array_equal(result, sliced)


class TestAutoLoaderRoutesToOrbax:
    """auto_loader should reject raw Orbax for runtime loading."""

    def test_auto_loader_rejects_raw_orbax_runtime_loading(self, tmp_path: Path) -> None:
        (tmp_path / "ocdbt.process_0").mkdir()
        (tmp_path / "manifest.ocdbt").write_bytes(b"")

        from mogemma.loader import auto_loader

        with pytest.raises(RuntimeError, match=r"convert.*safetensors"):
            auto_loader(tmp_path)

    def test_auto_loader_errors_mention_both_formats(self, tmp_path: Path) -> None:
        from mogemma.loader import auto_loader

        with pytest.raises(FileNotFoundError) as excinfo:
            auto_loader(tmp_path)
        assert "safetensors" in str(excinfo.value).lower()
        assert "orbax" in str(excinfo.value).lower() or "ocdbt" in str(excinfo.value).lower()

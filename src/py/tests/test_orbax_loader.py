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


class TestAutoLoaderRoutesToOrbax:
    """auto_loader should return OrbaxLoader for Orbax directories."""

    def test_auto_loader_returns_orbax(self, tmp_path: Path) -> None:
        (tmp_path / "ocdbt.process_0").mkdir()
        (tmp_path / "manifest.ocdbt").write_bytes(b"")

        from mogemma.loader import auto_loader

        fake_kvstore = MagicMock()
        fake_kvstore.list.return_value.result.return_value = []
        fake_ts = MagicMock()
        fake_ts.KvStore.open.return_value.result.return_value = fake_kvstore

        with patch.dict("sys.modules", {"tensorstore": fake_ts}):
            loader = auto_loader(tmp_path)

        assert isinstance(loader, OrbaxLoader)

    def test_auto_loader_errors_mention_both_formats(self, tmp_path: Path) -> None:
        from mogemma.loader import auto_loader

        with pytest.raises(FileNotFoundError) as excinfo:
            auto_loader(tmp_path)
        assert "safetensors" in str(excinfo.value).lower()
        assert "orbax" in str(excinfo.value).lower() or "ocdbt" in str(excinfo.value).lower()

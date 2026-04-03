"""Tests for bf16-to-f32 conversion and dtype-aware safetensors loading."""

from __future__ import annotations

import json
import struct
from typing import TYPE_CHECKING

import numpy as np

from mogemma.loader import SafetensorsLoader, _bf16_to_f32

if TYPE_CHECKING:
    from pathlib import Path


class TestBf16ToF32:
    """Tests for the raw bf16 → f32 bit-shift conversion."""

    def test_zero(self) -> None:
        """bf16 zero (0x0000) should convert to f32 zero."""
        raw = np.array([0x0000], dtype=np.uint16)
        result = _bf16_to_f32(raw.tobytes(), shape=(1,))
        np.testing.assert_array_equal(result, np.array([0.0], dtype=np.float32))

    def test_one(self) -> None:
        """bf16 1.0 (0x3F80) should convert to f32 1.0."""
        raw = np.array([0x3F80], dtype=np.uint16)
        result = _bf16_to_f32(raw.tobytes(), shape=(1,))
        np.testing.assert_array_almost_equal(result, np.array([1.0], dtype=np.float32))

    def test_negative_one(self) -> None:
        """bf16 -1.0 (0xBF80) should convert to f32 -1.0."""
        raw = np.array([0xBF80], dtype=np.uint16)
        result = _bf16_to_f32(raw.tobytes(), shape=(1,))
        np.testing.assert_array_almost_equal(result, np.array([-1.0], dtype=np.float32))

    def test_small_value(self) -> None:
        """bf16 0.5 (0x3F00) should convert to f32 0.5."""
        raw = np.array([0x3F00], dtype=np.uint16)
        result = _bf16_to_f32(raw.tobytes(), shape=(1,))
        np.testing.assert_array_almost_equal(result, np.array([0.5], dtype=np.float32))

    def test_multi_element(self) -> None:
        """Convert a multi-element bf16 array."""
        # 1.0 (0x3F80), -2.0 (0xC000), 0.0 (0x0000)
        raw = np.array([0x3F80, 0xC000, 0x0000], dtype=np.uint16)
        result = _bf16_to_f32(raw.tobytes(), shape=(3,))
        np.testing.assert_array_almost_equal(result, np.array([1.0, -2.0, 0.0], dtype=np.float32))

    def test_shape_preserved(self) -> None:
        """Output should have the requested shape."""
        raw = np.array([0x3F80, 0x3F80, 0x3F80, 0x3F80], dtype=np.uint16)
        result = _bf16_to_f32(raw.tobytes(), shape=(2, 2))
        assert result.shape == (2, 2)
        assert result.dtype == np.float32

    def test_dtype_is_float32(self) -> None:
        raw = np.array([0x3F80], dtype=np.uint16)
        result = _bf16_to_f32(raw.tobytes(), shape=(1,))
        assert result.dtype == np.float32


def _make_safetensors_file(path: Path, tensors: dict[str, tuple[list[int], str, bytes]]) -> None:
    """Write a minimal safetensors file.

    *tensors* maps ``name -> (shape, dtype_str, raw_bytes)``.
    """
    header: dict[str, dict] = {}
    offset = 0
    data_parts: list[bytes] = []
    for name, (shape, dtype_str, raw) in tensors.items():
        header[name] = {"dtype": dtype_str, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        data_parts.append(raw)
        offset += len(raw)

    header_bytes = json.dumps(header).encode("utf-8")
    header_size = struct.pack("<Q", len(header_bytes))
    path.write_bytes(header_size + header_bytes + b"".join(data_parts))


class TestSafetensorsLoaderBf16:
    """Tests for SafetensorsLoader with bf16 dtype detection and conversion."""

    def test_bf16_tensor_converted_to_f32(self, tmp_path: Path) -> None:
        """bf16 tensor should be loaded as f32 with correct values."""
        # bf16 for [1.0, -1.0] = [0x3F80, 0xBF80]
        raw_bf16 = np.array([0x3F80, 0xBF80], dtype=np.uint16).tobytes()
        _make_safetensors_file(tmp_path / "model.safetensors", {"test_weight": ([2], "BF16", raw_bf16)})
        with SafetensorsLoader(tmp_path) as loader:
            meta = loader.get_tensor_metadata()
            assert "test_weight" in meta
            _ptr, shape, dtype = meta["test_weight"]
            assert shape == (2,)
            assert dtype == "F32"  # Reported as f32 after conversion

    def test_f32_tensor_passthrough(self, tmp_path: Path) -> None:
        """f32 tensor should pass through unchanged (zero-copy mmap)."""
        raw_f32 = np.array([1.0, -1.0], dtype=np.float32).tobytes()
        _make_safetensors_file(tmp_path / "model.safetensors", {"test_weight": ([2], "F32", raw_f32)})
        with SafetensorsLoader(tmp_path) as loader:
            meta = loader.get_tensor_metadata()
            _ptr, shape, dtype = meta["test_weight"]
            assert shape == (2,)
            assert dtype == "F32"

    def test_i8_tensor_passthrough(self, tmp_path: Path) -> None:
        """i8 (quantized) tensor should pass through unchanged."""
        raw_i8 = np.array([1, -1, 0, 127], dtype=np.int8).tobytes()
        _make_safetensors_file(tmp_path / "model.safetensors", {"quant_weight": ([4], "I8", raw_i8)})
        with SafetensorsLoader(tmp_path) as loader:
            meta = loader.get_tensor_metadata()
            _ptr, shape, dtype = meta["quant_weight"]
            assert shape == (4,)
            assert dtype == "I8"

    def test_bf16_converted_values_correct(self, tmp_path: Path) -> None:
        """Verify bf16 values are numerically correct after conversion."""
        # bf16: 0.5 (0x3F00), 2.0 (0x4000), -0.5 (0xBF00)
        raw_bf16 = np.array([0x3F00, 0x4000, 0xBF00], dtype=np.uint16).tobytes()
        _make_safetensors_file(tmp_path / "model.safetensors", {"w": ([3], "BF16", raw_bf16)})
        with SafetensorsLoader(tmp_path) as loader:
            meta = loader.get_tensor_metadata()
            ptr, _shape, _dtype = meta["w"]
            # Read the actual f32 values from the pointer
            arr = np.ctypeslib.as_array((np.ctypeslib.ctypes.c_float * 3).from_address(ptr))
            np.testing.assert_array_almost_equal(arr, [0.5, 2.0, -0.5])

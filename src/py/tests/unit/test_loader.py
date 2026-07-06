"""Unit tests for safetensors loader selection and dtype handling."""

from __future__ import annotations

import ctypes
import json
import struct
from typing import TYPE_CHECKING

import numpy as np
import pytest

from mogemma.loader import SafetensorsLoader, _bf16_to_f32, auto_loader

if TYPE_CHECKING:
    from pathlib import Path


def test_auto_loader_raises_on_empty_directory_and_mentions_supported_formats(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No supported model format") as exc_info:
        auto_loader(tmp_path)

    msg = str(exc_info.value).lower()
    assert "safetensors" in msg
    assert ("orbax" in msg) or ("ocdbt" in msg)


@pytest.mark.parametrize(
    ("files", "expected_result"),
    [({}, False), ({"model.safetensors": "{}"}, True), ({"model.safetensors.index.json": "{}"}, True)],
)
def test_safetensors_loader_can_load(files: dict[str, str], expected_result: object, tmp_path: Path) -> None:
    for relative_path, content in files.items():
        (tmp_path / relative_path).write_text(content)

    assert SafetensorsLoader.can_load(tmp_path) is expected_result


@pytest.mark.parametrize(
    ("raw", "shape", "expected"),
    [
        (np.array([0x0000], dtype=np.uint16), (1,), np.array([0.0], dtype=np.float32)),
        (np.array([0x3F80], dtype=np.uint16), (1,), np.array([1.0], dtype=np.float32)),
        (np.array([0xBF80], dtype=np.uint16), (1,), np.array([-1.0], dtype=np.float32)),
        (np.array([0x3F00], dtype=np.uint16), (1,), np.array([0.5], dtype=np.float32)),
        (np.array([0x3F80, 0xC000, 0x0000], dtype=np.uint16), (3,), np.array([1.0, -2.0, 0.0], dtype=np.float32)),
    ],
)
def test_bf16_to_f32_values(raw: np.ndarray, shape: tuple[int, ...], expected: np.ndarray) -> None:
    result = _bf16_to_f32(raw.tobytes(), shape=shape)

    np.testing.assert_array_almost_equal(result, expected)
    assert result.dtype == np.float32


def test_bf16_to_f32_preserves_requested_shape() -> None:
    raw = np.array([0x3F80, 0x3F80, 0x3F80, 0x3F80], dtype=np.uint16)

    result = _bf16_to_f32(raw.tobytes(), shape=(2, 2))

    assert result.shape == (2, 2)
    assert result.dtype == np.float32


def _make_safetensors_file(path: Path, tensors: dict[str, tuple[list[int], str, bytes]]) -> None:
    """Write a minimal safetensors file."""
    header: dict[str, dict[str, object]] = {}
    offset = 0
    data_parts: list[bytes] = []
    for name, (shape, dtype_str, raw) in tensors.items():
        header[name] = {"dtype": dtype_str, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        data_parts.append(raw)
        offset += len(raw)

    header_bytes = json.dumps(header).encode("utf-8")
    header_size = struct.pack("<Q", len(header_bytes))
    path.write_bytes(header_size + header_bytes + b"".join(data_parts))


@pytest.mark.parametrize(
    ("dtype_str", "raw", "expected_dtype"),
    [
        ("BF16", np.array([0x3F80, 0xBF80], dtype=np.uint16).tobytes(), "F32"),
        ("F32", np.array([1.0, -1.0], dtype=np.float32).tobytes(), "F32"),
        ("I8", np.array([1, -1, 0, 127], dtype=np.int8).tobytes(), "I8"),
    ],
)
def test_safetensors_loader_reports_supported_tensor_dtypes(
    dtype_str: str, raw: bytes, expected_dtype: str, tmp_path: Path
) -> None:
    header_shape = [2 if dtype_str != "I8" else 4]
    _make_safetensors_file(tmp_path / "model.safetensors", {"weight": (header_shape, dtype_str, raw)})

    with SafetensorsLoader(tmp_path) as loader:
        meta = loader.get_tensor_metadata()

    assert "weight" in meta
    _ptr, metadata_shape, dtype = meta["weight"]
    assert metadata_shape == (2 if dtype_str != "I8" else 4,)
    assert dtype == expected_dtype


def test_safetensors_loader_bf16_converted_values_are_readable(tmp_path: Path) -> None:
    raw_bf16 = np.array([0x3F00, 0x4000, 0xBF00], dtype=np.uint16).tobytes()
    _make_safetensors_file(tmp_path / "model.safetensors", {"w": ([3], "BF16", raw_bf16)})

    with SafetensorsLoader(tmp_path) as loader:
        meta = loader.get_tensor_metadata()
        ptr, _shape, _dtype = meta["w"]
        arr = np.ctypeslib.as_array((ctypes.c_float * 3).from_address(ptr))
        np.testing.assert_array_almost_equal(arr, [0.5, 2.0, -0.5])

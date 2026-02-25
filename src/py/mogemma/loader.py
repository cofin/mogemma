"""Weight loading utilities for Safetensors and Orbax/OCDBT checkpoints."""

from __future__ import annotations

import json
import mmap
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from typing_extensions import Self

if TYPE_CHECKING:
    from types import TracebackType


class SafetensorsLoader:
    """Manages memory-mapped Safetensors files and provides zero-copy memory pointers."""

    def __init__(self, model_path: str | Path) -> None:
        """Initialize the loader with a model directory."""
        self.model_path = Path(model_path)

        # Keep references to mmap objects and open files so they aren't garbage collected
        self.mmaps: dict[str, mmap.mmap] = {}
        self.file_objs: dict[str, Any] = {}

        self.tensor_file_map: dict[str, str] = {}
        self.tensor_metadata: dict[str, dict[str, Any]] = {}
        self.file_data_offsets: dict[str, int] = {}

        self._load_index()

    def _load_index(self) -> None:
        if self.model_path.is_file():
            self._mmap_file(self.model_path.name, self.model_path)
            return

        index_file = self.model_path / "model.safetensors.index.json"
        if index_file.exists():
            with index_file.open("r", encoding="utf-8") as f:
                index = json.load(f)
            self.tensor_file_map = index.get("weight_map", {})
            unique_files = set(self.tensor_file_map.values())
            for file_name in unique_files:
                file_path = self.model_path / file_name
                if not file_path.exists():
                    msg = f"Missing weights file: {file_path}"
                    raise FileNotFoundError(msg)
                self._mmap_file(file_name, file_path)
        else:
            single_file = self.model_path / "model.safetensors"
            if not single_file.exists():
                msg = f"No model.safetensors or index found in {self.model_path}"
                raise FileNotFoundError(msg)
            self._mmap_file("model.safetensors", single_file)

    def _mmap_file(self, file_name: str, file_path: Path) -> None:

        # Open file and map into memory
        f = file_path.open("rb")
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        self.file_objs[file_name] = f
        self.mmaps[file_name] = m

        # Parse the 8-byte header size
        header_size_bytes = m[:8]
        header_size = struct.unpack("<Q", header_size_bytes)[0]

        # Read the JSON header
        header_bytes = m[8 : 8 + header_size]
        header = json.loads(header_bytes)

        # Data starts right after the 8 byte length and the JSON header
        data_start = 8 + header_size
        self.file_data_offsets[file_name] = data_start

        for name, meta in header.items():
            if name == "__metadata__":
                continue
            self.tensor_file_map[name] = file_name
            self.tensor_metadata[name] = meta

    def get_tensor_metadata(self) -> dict[str, tuple[int, tuple[int, ...], str]]:
        """Return a mapping of tensor name to its (data_pointer, shape, dtype) for Mojo FFI."""
        result = {}
        for name, meta in self.tensor_metadata.items():
            file_name = self.tensor_file_map[name]
            m = self.mmaps[file_name]
            data_start = self.file_data_offsets[file_name]

            # The data offsets are relative to the end of the JSON header
            start_offset = data_start + meta["data_offsets"][0]

            # Since Python's mmap object does not directly expose its base memory address,
            # we can use numpy to safely get the pointer to the readonly buffer.
            arr = np.frombuffer(m, dtype=np.uint8)
            base_ptr = arr.ctypes.data
            tensor_ptr = base_ptr + start_offset

            shape = tuple(meta["shape"])
            dtype = str(meta["dtype"])
            result[name] = (tensor_ptr, shape, dtype)

        return result

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        """Context manager exit."""
        self.close()

    @staticmethod
    def can_load(model_path: Path) -> bool:
        """Return ``True`` when *model_path* contains safetensors files."""
        if model_path.is_file() and model_path.suffix == ".safetensors":
            return True
        if not model_path.is_dir():
            return False
        return (model_path / "model.safetensors").exists() or (model_path / "model.safetensors.index.json").exists()

    def close(self) -> None:
        """Close memory maps and files."""
        for m in self.mmaps.values():
            m.close()
        for f in self.file_objs.values():
            f.close()
        self.mmaps.clear()
        self.file_objs.clear()
        self.tensor_file_map.clear()
        self.tensor_metadata.clear()


class ModelLoader(Protocol):
    """Structural protocol for weight loaders (SafetensorsLoader, OrbaxLoader)."""

    model_path: Path

    def get_tensor_metadata(self) -> dict[str, tuple[int, tuple[int, ...], str]]:
        """Return ``{name: (data_ptr, shape, dtype_str)}`` for Mojo FFI."""
        ...

    def close(self) -> None:
        """Release underlying resources."""
        ...


def auto_loader(model_path: str | Path) -> ModelLoader:
    """Detect the checkpoint format at *model_path* and return the appropriate loader."""
    from .orbax_loader import OrbaxLoader  # noqa: PLC0415

    path = Path(model_path)

    if SafetensorsLoader.can_load(path):
        return SafetensorsLoader(path)

    if OrbaxLoader.can_load(path):
        return OrbaxLoader(path)

    msg = (
        f"No supported model format found in {path}. "
        "Expected safetensors files (model.safetensors) or an Orbax/OCDBT checkpoint (ocdbt.process_0/)."
    )
    raise FileNotFoundError(msg)

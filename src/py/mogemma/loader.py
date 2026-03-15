"""Weight loading utilities for Safetensors and Orbax/OCDBT checkpoints."""

from __future__ import annotations

import json
import logging
import mmap
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import obstore as obs
from obstore.store import LocalStore
from typing_extensions import Self

if TYPE_CHECKING:
    from types import TracebackType

logger = logging.getLogger(__name__)


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

    @staticmethod
    def _get_store_and_path(path: Path | str) -> tuple[LocalStore, str]:
        # obstore LocalStore("/") treats paths as relative to root,
        # so we strip leading slash from absolute paths.
        p = Path(path).resolve()
        return LocalStore("/"), str(p).lstrip("/")

    @staticmethod
    def _head_exists(store: LocalStore, path: str) -> bool:
        try:
            obs.head(store, path)
        except Exception as exc:  # noqa: BLE001
            logger.debug("obstore head failed for %s", path, exc_info=exc)
            return False
        else:
            return True

    def _load_index(self) -> None:
        store, p = self._get_store_and_path(self.model_path)
        is_file = self._head_exists(store, p)

        if is_file:
            self._mmap_file(self.model_path.name, self.model_path)
            return

        index_file = self.model_path / "model.safetensors.index.json"
        _, p_index = self._get_store_and_path(index_file)
        index_exists = self._head_exists(store, p_index)

        if index_exists:
            result = obs.get(store, p_index)
            index = json.loads(bytes(result.bytes()))
            self.tensor_file_map = index.get("weight_map", {})
            unique_files = set(self.tensor_file_map.values())
            for file_name in unique_files:
                file_path = self.model_path / file_name
                _, p_file = self._get_store_and_path(file_path)
                if not self._head_exists(store, p_file):
                    msg = f"Missing weights file: {file_path}"
                    raise FileNotFoundError(msg) from None
                self._mmap_file(file_name, file_path)
        else:
            single_file = self.model_path / "model.safetensors"
            _, p_single = self._get_store_and_path(single_file)
            if not self._head_exists(store, p_single):
                msg = f"No model.safetensors or index found in {self.model_path}"
                raise FileNotFoundError(msg) from None
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

        store, p = SafetensorsLoader._get_store_and_path(model_path)
        return SafetensorsLoader._head_exists(store, f"{p}/model.safetensors") or SafetensorsLoader._head_exists(
            store, f"{p}/model.safetensors.index.json"
        )

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

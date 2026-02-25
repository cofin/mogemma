"""Weight loading utilities for Orbax/OCDBT checkpoints (Google's native format)."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

if TYPE_CHECKING:
    from types import TracebackType

_TS = cast("Any", importlib.import_module("tensorstore"))


class OrbaxLoader:
    """Loads tensors from an Orbax/OCDBT checkpoint directory using TensorStore.

    Google's GCS ``gemma-data`` bucket distributes model weights in this format
    (Zarr arrays inside an OCDBT key-value store).  This loader reads each tensor
    into a contiguous numpy array and exposes the same ``get_tensor_metadata()``
    interface consumed by the Mojo FFI bridge.
    """

    def __init__(self, model_path: str | Path) -> None:
        """Initialize the loader from an Orbax/OCDBT checkpoint directory."""
        self.model_path = Path(model_path)
        self._arrays: dict[str, npt.NDArray[np.generic]] = {}
        self._load_checkpoint()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ocdbt_dir(model_path: Path) -> Path:
        """Return the OCDBT process directory inside *model_path*."""
        return model_path / "ocdbt.process_0"

    @staticmethod
    def can_load(model_path: Path) -> bool:
        """Return ``True`` when *model_path* looks like an Orbax checkpoint."""
        return (model_path / "ocdbt.process_0").is_dir() and (model_path / "manifest.ocdbt").exists()

    def _enumerate_tensors(self) -> list[str]:
        """List every tensor path stored in the OCDBT key-value store."""
        kvs = _TS.KvStore.open({"driver": "ocdbt", "base": f"file://{self._ocdbt_dir(self.model_path)}"}).result()
        listing: list[bytes] = kvs.list().result()
        return sorted({key.decode().rsplit("/", 1)[0] for key in listing if key.decode().endswith("/.zarray")})

    def _read_tensor(self, tensor_path: str) -> npt.NDArray[np.generic]:
        """Read a single tensor from the OCDBT store into a contiguous numpy array."""
        store = _TS.open({
            "driver": "zarr",
            "kvstore": {"driver": "ocdbt", "base": f"file://{self._ocdbt_dir(self.model_path)}"},
            "path": tensor_path,
        }).result()
        return np.ascontiguousarray(store.read().result())

    def _load_checkpoint(self) -> None:
        """Read every tensor into memory."""
        tensor_paths = self._enumerate_tensors()
        if not tensor_paths:
            msg = f"No tensors found in OCDBT checkpoint at {self.model_path}"
            raise FileNotFoundError(msg)

        for path in tensor_paths:
            self._arrays[path] = self._read_tensor(path)

    # ------------------------------------------------------------------
    # Public API (matches SafetensorsLoader)
    # ------------------------------------------------------------------

    def get_tensor_metadata(self) -> dict[str, tuple[int, tuple[int, ...], str]]:
        """Return ``{name: (data_ptr, shape, dtype_str)}`` for Mojo FFI."""
        result: dict[str, tuple[int, tuple[int, ...], str]] = {}
        for name, arr in self._arrays.items():
            ptr = arr.ctypes.data
            shape = tuple(arr.shape)
            dtype = str(arr.dtype)
            result[name] = (ptr, shape, dtype)
        return result

    def close(self) -> None:
        """Release numpy arrays."""
        self._arrays.clear()

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        """Context manager exit."""
        self.close()

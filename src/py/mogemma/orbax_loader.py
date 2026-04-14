"""Weight loading for Orbax/OCDBT checkpoints (Gemma 4 on GCS).

Orbax distributes Gemma 4 weights as an OCDBT (O(C)oncurrent (D)B(T)ree)
key-value store containing zarr arrays.  ``tensorstore`` is the only library
that can read the OCDBT format, so we use it to open the local checkpoint
(downloaded beforehand by :class:`mogemma.hub.HubManager`) and materialize
each tensor as a numpy array.

This loader exposes the same structural interface as
:class:`mogemma.loader.SafetensorsLoader` so that
:func:`mogemma.loader.auto_loader` can return either implementation
transparently.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from typing_extensions import Self

if TYPE_CHECKING:
    from types import TracebackType

logger = logging.getLogger(__name__)

_ZARRAY_SUFFIX = ".zarray"


def _numpy_dtype_to_safetensors_str(dtype: np.dtype) -> str:
    """Map a numpy dtype to the safetensors-style dtype string used by the FFI."""
    name = dtype.name
    mapping = {
        "float64": "F64",
        "float32": "F32",
        "float16": "F16",
        "bfloat16": "BF16",
        "int64": "I64",
        "int32": "I32",
        "int16": "I16",
        "int8": "I8",
        "uint8": "U8",
        "bool": "BOOL",
    }
    return mapping.get(name) or name.upper()


class OrbaxLoader:
    """Load Gemma 4 weights from a local Orbax/OCDBT checkpoint.

    Tensors are read eagerly into ``float32`` numpy arrays (bfloat16 is
    upcast to match :class:`SafetensorsLoader`'s FFI contract) and kept
    alive for the lifetime of the loader so that Mojo can consume the
    raw pointers without fear of GC.
    """

    def __init__(self, model_path: str | Path) -> None:
        """Open *model_path* and materialize every tensor in the checkpoint."""
        self.model_path = Path(model_path)
        # Keep numpy arrays alive so their ``ctypes.data`` pointers remain valid.
        self._tensors: dict[str, np.ndarray] = {}
        self._tensor_dtypes: dict[str, str] = {}
        self._load()

    @staticmethod
    def _ocdbt_base_url(model_path: Path) -> str:
        """Return the ``file://`` URL for the OCDBT process directory."""
        return f"file://{model_path}/ocdbt.process_0/"

    @staticmethod
    def can_load(model_path: Path) -> bool:
        """Return ``True`` when *model_path* is an Orbax/OCDBT checkpoint directory."""
        if not model_path.is_dir():
            return False
        return (model_path / "ocdbt.process_0").is_dir() and (model_path / "manifest.ocdbt").exists()

    def _enumerate_tensor_names(self) -> list[str]:
        """List tensor names by scanning the OCDBT kvstore for ``.zarray`` entries."""
        import tensorstore as ts  # noqa: PLC0415 — heavy dep, only imported when loading Orbax

        kvstore = ts.KvStore.open({"driver": "ocdbt", "base": self._ocdbt_base_url(self.model_path)}).result()
        keys = kvstore.list().result()

        tensor_names: set[str] = set()
        for raw_key in keys:
            key = raw_key.decode() if isinstance(raw_key, bytes) else str(raw_key)
            if key.endswith(_ZARRAY_SUFFIX):
                name = key[: -len(_ZARRAY_SUFFIX)].rstrip("/")
                if name:
                    tensor_names.add(name)
        return sorted(tensor_names)

    def _open_tensor(self, name: str) -> np.ndarray:
        """Read the zarr array at *name* from the OCDBT store into a numpy array."""
        import tensorstore as ts  # noqa: PLC0415 — heavy dep, lazy-imported

        spec: dict[str, Any] = {
            "driver": "zarr",
            "kvstore": {"driver": "ocdbt", "base": self._ocdbt_base_url(self.model_path), "path": name},
        }
        store = ts.open(spec).result()
        return np.asarray(store.read().result())

    def _load(self) -> None:
        for name in self._enumerate_tensor_names():
            arr = self._open_tensor(name)
            dtype_name = arr.dtype.name
            if dtype_name == "bfloat16":
                # Upcast BF16 → F32 to match SafetensorsLoader's FFI contract.
                arr = arr.astype(np.float32)
                self._tensor_dtypes[name] = "F32"
            else:
                self._tensor_dtypes[name] = _numpy_dtype_to_safetensors_str(arr.dtype)
            # Ensure the array owns contiguous memory so ``ctypes.data`` is valid.
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)
            self._tensors[name] = arr

    @classmethod
    def enumerate_tensors(cls, model_path: str | Path) -> list[str]:
        """Return the tensor names in *model_path* without materializing any data.

        Intended for streaming consumers (e.g. Orbax→safetensors conversion) that
        want to iterate tensors without paying the full-checkpoint RAM cost of
        :meth:`__init__`.
        """
        stub = cls.__new__(cls)
        stub.model_path = Path(model_path)
        return stub._enumerate_tensor_names()

    @classmethod
    def open_tensor(cls, model_path: str | Path, name: str) -> np.ndarray:
        """Read a single tensor lazily and return a contiguous float-array copy.

        BF16 is upcast to F32 to match :meth:`__init__`'s FFI contract. The
        returned array is owned by the caller (not cached on the loader).
        """
        stub = cls.__new__(cls)
        stub.model_path = Path(model_path)
        arr = stub._open_tensor(name)
        if arr.dtype.name == "bfloat16":
            arr = arr.astype(np.float32)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return arr

    def get_tensor_metadata(self) -> dict[str, tuple[int, tuple[int, ...], str]]:
        """Return ``{name: (data_ptr, shape, dtype_str)}`` for Mojo FFI."""
        result: dict[str, tuple[int, tuple[int, ...], str]] = {}
        for name, arr in self._tensors.items():
            result[name] = (arr.ctypes.data, tuple(arr.shape), self._tensor_dtypes[name])
        return result

    def close(self) -> None:
        """Release numpy arrays."""
        self._tensors.clear()
        self._tensor_dtypes.clear()

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        """Context manager exit."""
        self.close()

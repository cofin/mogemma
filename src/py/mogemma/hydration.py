"""Image hydration utilities for multimodal model inputs."""

from __future__ import annotations

import importlib
import io
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import obstore as obs
from obstore.store import LocalStore

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt


class ImageHydrator:
    """Handles loading and decoding of images from various sources using obstore."""

    def __init__(self) -> None:
        """Initialize the local object store used for path-based hydration."""
        self._store = LocalStore()

    def hydrate(self, inputs: Sequence[str | Path | bytes | npt.NDArray[np.generic]]) -> list[npt.NDArray[np.uint8]]:
        """Convert a sequence of inputs into a list of raw RGB uint8 arrays."""
        results: list[npt.NDArray[np.uint8]] = []
        for item in inputs:
            if isinstance(item, (str, Path)):
                data = self._load_from_path(item)
                results.append(self._decode(data))
            elif isinstance(item, bytes):
                results.append(self._decode(item))
            elif isinstance(item, np.ndarray):
                results.append(item.astype(np.uint8))
            else:
                msg = f"Unsupported image input type: {type(item)}"
                raise TypeError(msg)
        return results

    def _load_from_path(self, path: str | Path) -> bytes:
        """Load raw bytes from a local or remote path using obstore."""
        # Convert path to string relative to store root if needed
        path_str = str(path)
        result = obs.get(self._store, path_str)
        return bytes(result.bytes())

    def _decode(self, data: bytes) -> npt.NDArray[np.uint8]:
        """Decode image bytes into a numpy array.

        Automatic hydration of path and byte inputs requires Pillow via the
        optional ``mogemma[vision]`` extra. Pre-decoded ndarray inputs bypass
        this decoding path entirely.
        """
        try:
            image_module = importlib.import_module("PIL.Image")
            img = image_module.open(io.BytesIO(data)).convert("RGB")
            return np.asarray(img)
        except ModuleNotFoundError as exc:
            if exc.name is not None and not exc.name.startswith("PIL"):
                raise
            msg = (
                "Automatic image decoding requires Pillow. "
                "Install with: pip install 'mogemma[vision]'. "
                "Pre-decoded numpy uint8 RGB arrays do not require the vision extra."
            )
            raise ImportError(msg) from None

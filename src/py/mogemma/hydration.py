from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import obstore as obs
from obstore.store import LocalStore

if TYPE_CHECKING:
    import numpy.typing as npt


class ImageHydrator:
    """Handles loading and decoding of images from various sources using obstore."""

    def __init__(self) -> None:
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
        
        Currently uses Pillow as the engine. If Pillow is missing, 
        attempts a basic raw extraction or raises a helpful error.
        """
        try:
            from PIL import Image
            import io
            
            img = Image.open(io.BytesIO(data)).convert("RGB")
            return np.asarray(img)
        except ImportError:
            # Fallback or error
            msg = "Pillow is required for JPEG/PNG decoding. Install with: pip install 'mogemma[vision]'"
            raise ImportError(msg) from None

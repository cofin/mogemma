"""Image hydration utilities for multimodal model inputs."""

from __future__ import annotations

import importlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import obstore as obs
from obstore.store import LocalStore

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

# Patch size for Gemma 4 SigLIP vision encoder
PATCH_SIZE = 16

# Token budget table: num_tokens -> (target_h, target_w)
# Each target must be divisible by PATCH_SIZE (16).
TOKEN_BUDGETS: dict[int, list[tuple[int, int]]] = {
    70: [(280, 280)],
    140: [(280, 560), (560, 280)],
    280: [(560, 560)],
    560: [(560, 1120), (1120, 560)],
    1120: [(1120, 1120)],
}

# Sorted budget keys for iteration
_BUDGET_KEYS = sorted(TOKEN_BUDGETS.keys())


@dataclass
class ImageInput:
    """Preprocessed image ready for the vision encoder."""

    patches: npt.NDArray[np.float32]  # [num_tokens, patch_size*patch_size*3]
    grid_h: int  # number of patch rows
    grid_w: int  # number of patch columns
    num_tokens: int  # grid_h * grid_w


def select_token_budget(
    h: int, w: int, max_tokens: int = 560
) -> tuple[int, int, int]:
    """Select the best token budget for the given image dimensions.

    Returns:
        (target_h, target_w, num_tokens) — the resolution to resize to and
        the resulting patch count.
    """
    # Clamp max_tokens to available budgets
    capped = max(b for b in _BUDGET_KEYS if b <= max_tokens) if max_tokens >= _BUDGET_KEYS[0] else _BUDGET_KEYS[0]

    aspect = h / w if w > 0 else 1.0
    best_target: tuple[int, int] | None = None
    best_diff = float("inf")

    for budget in _BUDGET_KEYS:
        if budget > capped:
            break
        for target_h, target_w in TOKEN_BUDGETS[budget]:
            target_aspect = target_h / target_w
            diff = abs(aspect - target_aspect)
            if diff < best_diff:
                best_diff = diff
                best_target = (target_h, target_w)

    if best_target is None:
        best_target = TOKEN_BUDGETS[_BUDGET_KEYS[0]][0]

    target_h, target_w = best_target
    num_tokens = (target_h // PATCH_SIZE) * (target_w // PATCH_SIZE)
    return target_h, target_w, num_tokens


class ImageHydrator:
    """Handles loading, preprocessing, and patch extraction for Gemma 4 vision."""

    def __init__(self) -> None:
        """Initialize the local object store used for path-based hydration."""
        self._store = LocalStore()

    def hydrate(
        self, inputs: Sequence[str | Path | bytes | npt.NDArray[np.generic]]
    ) -> list[ImageInput]:
        """Convert a sequence of inputs into preprocessed ImageInput objects."""
        results: list[ImageInput] = []
        for item in inputs:
            if isinstance(item, np.ndarray):
                rgb = item.astype(np.uint8) if item.dtype != np.uint8 else item
                results.append(self.preprocess_image(rgb))
            elif isinstance(item, (str, Path)):
                data = self._load_from_path(item)
                rgb = self._decode(data)
                results.append(self.preprocess_image(rgb))
            elif isinstance(item, bytes):
                rgb = self._decode(item)
                results.append(self.preprocess_image(rgb))
            else:
                msg = f"Unsupported image input type: {type(item)}"
                raise TypeError(msg)
        return results

    def preprocess_image(
        self, rgb_array: npt.NDArray[np.uint8], max_tokens: int = 560
    ) -> ImageInput:
        """Resize, normalize, and extract patches from an RGB image.

        Steps:
          1. Select target resolution via token budget
          2. Resize to target (PIL bicubic)
          3. Normalize: (pixel/255 - 0.5) / 0.5  →  [-1, 1]
          4. Extract non-overlapping 16×16 patches
        """
        h, w = rgb_array.shape[:2]
        target_h, target_w, num_tokens = select_token_budget(h, w, max_tokens)

        # Resize to target
        if h != target_h or w != target_w:
            resized = self._resize(rgb_array, target_h, target_w)
        else:
            resized = rgb_array

        # Crop to largest multiple of PATCH_SIZE (handles targets like 280 not divisible by 16)
        grid_h = target_h // PATCH_SIZE
        grid_w = target_w // PATCH_SIZE
        crop_h = grid_h * PATCH_SIZE
        crop_w = grid_w * PATCH_SIZE
        resized = resized[:crop_h, :crop_w]

        # Normalize: SigLIP normalization
        normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        patches = (
            normalized.reshape(grid_h, PATCH_SIZE, grid_w, PATCH_SIZE, 3)
            .transpose(0, 2, 1, 3, 4)
            .reshape(num_tokens, PATCH_SIZE * PATCH_SIZE * 3)
        )

        return ImageInput(
            patches=patches, grid_h=grid_h, grid_w=grid_w, num_tokens=num_tokens
        )

    def _resize(self, img: npt.NDArray[np.uint8], target_h: int, target_w: int) -> npt.NDArray[np.uint8]:
        """Resize image using PIL bicubic interpolation."""
        try:
            image_module = importlib.import_module("PIL.Image")
            pil_img = image_module.fromarray(img)
            resized = pil_img.resize((target_w, target_h), image_module.BICUBIC)
            return np.asarray(resized)
        except ModuleNotFoundError as exc:
            if exc.name is not None and not exc.name.startswith("PIL"):
                raise
            msg = (
                "Image resizing requires Pillow. "
                "Install with: pip install 'mogemma[vision]'."
            )
            raise ImportError(msg) from None

    def _load_from_path(self, path: str | Path) -> bytes:
        """Load raw bytes from a local or remote path using obstore."""
        path_str = str(path)
        result = obs.get(self._store, path_str)
        return bytes(result.bytes())

    def _decode(self, data: bytes) -> npt.NDArray[np.uint8]:
        """Decode image bytes into a numpy array.

        Automatic hydration of path and byte inputs requires Pillow via the
        optional ``mogemma[vision]`` extra.
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

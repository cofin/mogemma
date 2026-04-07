"""Tests for Gemma 4 vision preprocessing: token budgets, normalization, patches."""

from __future__ import annotations

import numpy as np

from mogemma.hydration import PATCH_SIZE, ImageHydrator, ImageInput, select_token_budget


class TestTokenBudget:
    """Token budget selection for variable resolution."""

    def test_square_image_default(self) -> None:
        target_h, target_w, num_tokens = select_token_budget(1000, 1000)
        # Square 1:1 aspect best matches (560,560) at budget 280
        assert target_h == target_w  # square target for square input
        assert num_tokens == (target_h // PATCH_SIZE) * (target_w // PATCH_SIZE)

    def test_small_image_min_budget(self) -> None:
        target_h, target_w, num_tokens = select_token_budget(100, 100, max_tokens=70)
        assert num_tokens == (target_h // PATCH_SIZE) * (target_w // PATCH_SIZE)
        assert target_h == 280
        assert target_w == 280

    def test_wide_landscape(self) -> None:
        target_h, target_w, _num_tokens = select_token_budget(300, 900)
        assert target_w >= target_h

    def test_tall_portrait(self) -> None:
        target_h, target_w, _num_tokens = select_token_budget(900, 300)
        assert target_h >= target_w

    def test_max_tokens_cap(self) -> None:
        _, _, num_tokens = select_token_budget(2000, 2000, max_tokens=280)
        # Should not exceed the capped budget's patch count
        assert num_tokens <= (560 // PATCH_SIZE) * (560 // PATCH_SIZE)

    def test_all_budgets_produce_valid_patch_counts(self) -> None:
        """All selected targets should produce non-zero patch counts."""
        for max_t in [70, 140, 280, 560, 1120]:
            h, w, n = select_token_budget(500, 500, max_tokens=max_t)
            assert n > 0, f"zero patches for budget {max_t}"
            assert n == (h // PATCH_SIZE) * (w // PATCH_SIZE)


class TestImageInput:
    """ImageInput dataclass."""

    def test_fields(self) -> None:
        patches = np.zeros((70, 768), dtype=np.float32)
        img_input = ImageInput(patches=patches, grid_h=7, grid_w=10, num_tokens=70)
        assert img_input.grid_h == 7
        assert img_input.grid_w == 10
        assert img_input.num_tokens == 70
        assert img_input.patches.shape == (70, 768)


class TestPreprocessImage:
    """End-to-end preprocessing: resize, normalize, patch."""

    def test_normalization_range(self) -> None:
        """SigLIP normalization maps [0,255] -> [-1, 1]."""
        hydrator = ImageHydrator()
        # Use size that's exact multiple of PATCH_SIZE
        size = PATCH_SIZE * 17  # 272
        white = np.full((size, size, 3), 255, dtype=np.uint8)
        result = hydrator.preprocess_image(white, max_tokens=1120)
        assert np.allclose(result.patches, 1.0, atol=1e-5)

    def test_normalization_black(self) -> None:
        size = PATCH_SIZE * 17
        black = np.zeros((size, size, 3), dtype=np.uint8)
        hydrator = ImageHydrator()
        result = hydrator.preprocess_image(black, max_tokens=1120)
        assert np.allclose(result.patches, -1.0, atol=1e-5)

    def test_patch_shape(self) -> None:
        """Patches should be [N, 16*16*3] = [N, 768]."""
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        hydrator = ImageHydrator()
        result = hydrator.preprocess_image(img, max_tokens=280)
        assert result.patches.shape[1] == PATCH_SIZE * PATCH_SIZE * 3
        assert result.num_tokens == result.patches.shape[0]

    def test_grid_dimensions(self) -> None:
        img = np.zeros((800, 1600, 3), dtype=np.uint8)
        hydrator = ImageHydrator()
        result = hydrator.preprocess_image(img, max_tokens=560)
        assert result.num_tokens == result.grid_h * result.grid_w
        assert result.grid_h > 0
        assert result.grid_w > 0

    def test_resize_applied(self) -> None:
        """Non-standard image size should be resized to target."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        hydrator = ImageHydrator()
        result = hydrator.preprocess_image(img, max_tokens=140)
        assert result.patches.shape[0] > 0
        assert result.patches.dtype == np.float32


class TestHydrateIntegration:
    """Test the full hydrate() pipeline returning ImageInput list."""

    def test_hydrate_numpy_array(self) -> None:
        hydrator = ImageHydrator()
        img = np.zeros((280, 280, 3), dtype=np.uint8)
        results = hydrator.hydrate([img])
        assert len(results) == 1
        assert isinstance(results[0], ImageInput)
        assert results[0].patches.shape[1] == PATCH_SIZE * PATCH_SIZE * 3

    def test_hydrate_multiple(self) -> None:
        hydrator = ImageHydrator()
        img1 = np.zeros((280, 280, 3), dtype=np.uint8)
        img2 = np.full((560, 560, 3), 128, dtype=np.uint8)
        results = hydrator.hydrate([img1, img2])
        assert len(results) == 2

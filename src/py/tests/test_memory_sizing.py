"""Tests for hybrid KV cache memory sizing."""

from __future__ import annotations

import pytest

from mogemma.model import compute_kv_cache_memory


class TestComputeKVCacheMemory:
    """Tests for compute_kv_cache_memory()."""

    def test_all_sliding_layers(self) -> None:
        """All sliding layers: each gets window_size * kv_stride * 4 bytes * 2 (K+V)."""
        result = compute_kv_cache_memory(
            num_layers=4,
            layer_types=["sliding"] * 4,
            window_size=1024,
            max_context_len=8192,
            num_kv_heads=8,
            head_dim=128,
        )
        # 4 layers * 1024 slots * 8 heads * 128 dim * 4 bytes * 2 (K+V)
        expected = 4 * 1024 * 8 * 128 * 4 * 2
        assert result == expected

    def test_all_full_layers(self) -> None:
        """All full layers: each gets max_context_len * kv_stride * 4 bytes * 2."""
        result = compute_kv_cache_memory(
            num_layers=4, layer_types=["full"] * 4, window_size=1024, max_context_len=8192, num_kv_heads=8, head_dim=128
        )
        expected = 4 * 8192 * 8 * 128 * 4 * 2
        assert result == expected

    def test_mixed_layers(self) -> None:
        """Mix of sliding and full layers sums correctly."""
        result = compute_kv_cache_memory(
            num_layers=4,
            layer_types=["sliding", "sliding", "full", "sliding"],
            window_size=512,
            max_context_len=4096,
            num_kv_heads=4,
            head_dim=64,
        )
        kv_stride = 4 * 64
        sliding_elements = 512 * kv_stride
        full_elements = 4096 * kv_stride
        total_elements = 3 * sliding_elements + 1 * full_elements
        expected = total_elements * 4 * 2
        assert result == expected

    def test_realistic_31b_config(self) -> None:
        """Realistic 31B model: 60 layers, mostly sliding, every 6th full."""
        layer_types = ["full" if i % 6 == 5 else "sliding" for i in range(60)]
        num_full = sum(1 for lt in layer_types if lt == "full")
        num_sliding = 60 - num_full

        result = compute_kv_cache_memory(
            num_layers=60, layer_types=layer_types, window_size=1024, max_context_len=8192, num_kv_heads=8, head_dim=256
        )
        kv_stride = 8 * 256
        expected = (num_sliding * 1024 + num_full * 8192) * kv_stride * 4 * 2
        assert result == expected
        # Verify it's much less than uniform allocation
        uniform = 60 * 8192 * kv_stride * 4 * 2
        assert result < uniform

    def test_layer_types_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="layer_types length"):
            compute_kv_cache_memory(
                num_layers=4,
                layer_types=["sliding"] * 3,
                window_size=1024,
                max_context_len=8192,
                num_kv_heads=8,
                head_dim=128,
            )

    def test_max_context_less_than_window_raises(self) -> None:
        with pytest.raises(ValueError, match=r"max_context_len.*must be >= window_size"):
            compute_kv_cache_memory(
                num_layers=4,
                layer_types=["sliding"] * 4,
                window_size=1024,
                max_context_len=512,
                num_kv_heads=8,
                head_dim=128,
            )

    def test_zero_layers(self) -> None:
        result = compute_kv_cache_memory(
            num_layers=0, layer_types=[], window_size=1024, max_context_len=8192, num_kv_heads=8, head_dim=128
        )
        assert result == 0

    def test_single_sliding_layer(self) -> None:
        result = compute_kv_cache_memory(
            num_layers=1, layer_types=["sliding"], window_size=512, max_context_len=8192, num_kv_heads=1, head_dim=64
        )
        expected = 1 * 512 * 1 * 64 * 4 * 2
        assert result == expected

    def test_single_full_layer(self) -> None:
        result = compute_kv_cache_memory(
            num_layers=1, layer_types=["full"], window_size=512, max_context_len=4096, num_kv_heads=1, head_dim=64
        )
        expected = 1 * 4096 * 1 * 64 * 4 * 2
        assert result == expected

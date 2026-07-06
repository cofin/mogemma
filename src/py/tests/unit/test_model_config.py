"""Unit tests for Gemma 4 model configuration, support metadata, and architecture parsing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mogemma.backends import resolve_device_selection
from mogemma.config import EmbeddingConfig, GenerationConfig
from mogemma.hub import KNOWN_GCS_MODELS
from mogemma.model import (
    Gemma4Variant,
    _detect_gemma4_variant,
    _initialize_llm,
    _parse_gemma4_architecture,
    compute_kv_cache_memory,
)
from mogemma.model_support import GEMMA4_MODEL_SUPPORT, OFFICIAL_GEMMA4_MODELS, RuntimeSupport, get_gemma4_model_support

if TYPE_CHECKING:
    from collections.abc import Callable

KVMemoryCase = tuple[int, list[str], int, int, int, int, int]


def _write_config(model_dir: Path, config: dict[str, object]) -> Path:
    config_path = model_dir / "config.json"
    config_path.write_text(json.dumps(config))
    return config_path


def _gemma4_31b_config() -> dict[str, object]:
    return {
        "model_type": "gemma4",
        "num_hidden_layers": 4,
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "intermediate_size": 256,
        "vocab_size": 1000,
        "sliding_window_size": 1024,
        "partial_rotary_factor": 0.5,
        "attention_k_eq_v": True,
        "layer_types": ["sliding", "sliding", "full", "sliding"],
    }


def _e2b_config() -> dict[str, object]:
    return {
        "model_type": "gemma4",
        "num_hidden_layers": 35,
        "hidden_size": 1536,
        "hidden_size_per_layer_input": 256,
        "vocab_size_per_layer_input": 262144,
        "use_double_wide_mlp": True,
        "sliding_window_size": 512,
        "partial_rotary_factor": 0.5,
        "attention_k_eq_v": False,
        "layer_types": ["sliding"] * 35,
        "kv_sharing_layer_map": [-1] * 15 + [i % 15 for i in range(20)],
    }


def _e4b_config() -> dict[str, object]:
    return {
        "model_type": "gemma4",
        "num_hidden_layers": 42,
        "hidden_size": 2560,
        "hidden_size_per_layer_input": 256,
        "vocab_size_per_layer_input": 262144,
        "use_double_wide_mlp": False,
        "sliding_window_size": 512,
        "attention_k_eq_v": False,
        "layer_types": ["sliding"] * 42,
        "kv_sharing_layer_map": [-1] * 24 + list(range(18)),
    }


def _moe_26b_config() -> dict[str, object]:
    return {
        "model_type": "gemma4",
        "num_hidden_layers": 30,
        "hidden_size": 3584,
        "num_local_experts": 128,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 704,
        "sliding_window_size": 1024,
        "partial_rotary_factor": 0.5,
        "attention_k_eq_v": True,
        "layer_types": ["sliding"] * 30,
    }


def _vision_config() -> dict[str, object]:
    config = _gemma4_31b_config()
    config.update({
        "hidden_size": 3584,
        "image_token_index": 255999,
        "vision_config": {
            "num_hidden_layers": 27,
            "hidden_size": 1152,
            "num_attention_heads": 16,
            "intermediate_size": 4304,
            "patch_size": 16,
        },
    })
    return config


@pytest.mark.parametrize("config_cls", [GenerationConfig, EmbeddingConfig])
def test_default_model_path_is_available_gcs_model(config_cls: type[GenerationConfig | EmbeddingConfig]) -> None:
    config = config_cls()
    assert str(config.model_path) == "google/gemma-4-E4B-it"
    assert str(config.model_path) in KNOWN_GCS_MODELS


@pytest.mark.parametrize(
    ("field", "expected"),
    [("top_k", 64), ("top_p", 0.95), ("temperature", 1.0), ("max_tokens", 128), ("max_image_tokens", 560)],
)
def test_generation_config_defaults(field: str, expected: float) -> None:
    assert getattr(GenerationConfig(), field) == expected


def test_generation_config_accepts_custom_max_image_tokens() -> None:
    config = GenerationConfig(model_path="google/gemma-4-26B-A4B-it", max_image_tokens=1120)
    assert config.max_image_tokens == 1120


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: GenerationConfig(temperature=-1.0), "temperature"),
        (lambda: GenerationConfig(top_k=-1), "top_k"),
        (lambda: GenerationConfig(top_p=1.5), "top_p"),
        (lambda: GenerationConfig(model_path=""), "model_path"),
        (lambda: GenerationConfig(max_tokens=0), "max_tokens"),
    ],
)
def test_generation_config_rejects_invalid_values(factory: Callable[[], GenerationConfig], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        factory()


def test_support_matrix_tracks_official_gemma4_models() -> None:
    assert {"google/gemma-4-12B-it", "google/gemma-4-31B-it"}.issubset(OFFICIAL_GEMMA4_MODELS)
    assert set(GEMMA4_MODEL_SUPPORT) == set(OFFICIAL_GEMMA4_MODELS)


@pytest.mark.parametrize("model_id", ["google/gemma-4-12B-it", "google/gemma-4-31B-it"])
def test_official_support_is_separate_from_gcs_availability(model_id: str) -> None:
    assert model_id in OFFICIAL_GEMMA4_MODELS
    assert model_id not in KNOWN_GCS_MODELS


def test_12b_support_metadata_is_recognized_but_runtime_unsupported() -> None:
    support = get_gemma4_model_support("google/gemma-4-12B-it")

    assert support.variant == "gemma4_dense_12b_unified"
    assert support.official is True
    assert support.gcs_available is False
    assert support.text is RuntimeSupport.UNSUPPORTED
    assert support.image is RuntimeSupport.UNSUPPORTED
    assert support.audio is RuntimeSupport.UNSUPPORTED
    assert support.unified_multimodal is RuntimeSupport.REQUIRES_FOLLOWUP


@pytest.mark.parametrize("model_id", ["google/gemma-4-E2B-it", "google/gemma-4-E4B-it"])
def test_e2b_and_e4b_audio_status_is_explicit(model_id: str) -> None:
    support = get_gemma4_model_support(model_id)
    assert support.gcs_available is True
    assert support.text is RuntimeSupport.SUPPORTED
    assert support.audio is RuntimeSupport.UNSUPPORTED
    assert "audio" in support.notes.lower()


def test_mtp_and_thinking_template_support_is_explicit_followup() -> None:
    for support in GEMMA4_MODEL_SUPPORT.values():
        assert support.mtp is RuntimeSupport.REQUIRES_FOLLOWUP
        assert support.thinking is RuntimeSupport.REQUIRES_FOLLOWUP


@pytest.mark.parametrize(
    ("variant", "value"),
    [
        (Gemma4Variant.DENSE_31B, "gemma4_dense_31b"),
        (Gemma4Variant.DENSE_12B_UNIFIED, "gemma4_dense_12b_unified"),
        (Gemma4Variant.DENSE_E2B, "gemma4_dense_e2b"),
        (Gemma4Variant.DENSE_E4B, "gemma4_dense_e4b"),
        (Gemma4Variant.MOE_26B_A4B, "gemma4_moe_26b"),
    ],
)
def test_gemma4_variant_values(variant: Gemma4Variant, value: str) -> None:
    assert variant.value == value
    assert isinstance(variant, str)


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"hidden_size": 4096}, Gemma4Variant.DENSE_31B),
        (
            {
                "architectures": ["Gemma4UnifiedForConditionalGeneration"],
                "model_type": "gemma4_unified",
                "text_config": {"hidden_size": 3840, "num_hidden_layers": 48, "model_type": "gemma4_unified_text"},
            },
            Gemma4Variant.DENSE_12B_UNIFIED,
        ),
        ({"num_experts": 64}, Gemma4Variant.MOE_26B_A4B),
        ({"num_experts": 0}, Gemma4Variant.DENSE_31B),
        ({"hidden_size_per_layer_input": 2048, "use_double_wide_mlp": True}, Gemma4Variant.DENSE_E2B),
        ({"hidden_size_per_layer_input": 2048, "use_double_wide_mlp": False}, Gemma4Variant.DENSE_E4B),
        ({"hidden_size_per_layer_input": 2048}, Gemma4Variant.DENSE_E4B),
    ],
)
def test_detect_gemma4_variant(config: dict[str, object], expected: Gemma4Variant, tmp_path: Path) -> None:
    _write_config(tmp_path, config)
    assert _detect_gemma4_variant(tmp_path) is expected


def test_detect_gemma4_variant_requires_config_json(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _detect_gemma4_variant(tmp_path)


class _FakeLoader:
    model_path = Path("fake-12b")

    def get_tensor_metadata(self) -> dict[str, tuple[int, tuple[int, ...], str]]:
        msg = "12B runtime gate should run before tensor metadata loading"
        raise AssertionError(msg)

    def close(self) -> None:
        pass


class _FakeBackend:
    backend_id = "fake"

    def init_model(self, _metadata: object) -> object:
        msg = "12B runtime gate should run before Mojo init"
        raise AssertionError(msg)


def test_12b_unified_runtime_is_rejected_before_mojo_init(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        {
            "architectures": ["Gemma4UnifiedForConditionalGeneration"],
            "model_type": "gemma4_unified",
            "text_config": {"hidden_size": 3840, "num_hidden_layers": 48},
        },
    )

    with pytest.raises(RuntimeError, match="Gemma 4 12B unified multimodal runtime is recognized but not implemented"):
        _initialize_llm(
            _FakeLoader(),
            _FakeBackend(),  # type: ignore[arg-type]
            device_selection=resolve_device_selection("cpu"),
            model_type="generation",
            model_path=tmp_path,
        )


@pytest.mark.parametrize(
    ("config", "expected_overrides", "expected_layer_types"),
    [
        (_gemma4_31b_config(), {"window_size": 1024, "partial_rotary_factor": 0.5, "k_eq_v": 1}, [0, 0, 1, 0]),
        (
            _e2b_config(),
            {
                "window_size": 512,
                "partial_rotary_factor": 0.5,
                "k_eq_v": 0,
                "hidden_size_per_layer_input": 256,
                "vocab_size_per_layer_input": 262144,
                "use_double_wide_mlp": 1,
                "kv_sharing_layer_count": 35,
            },
            [0] * 35,
        ),
        (
            _e4b_config(),
            {
                "window_size": 512,
                "partial_rotary_factor": 0.5,
                "k_eq_v": 0,
                "hidden_size_per_layer_input": 256,
                "vocab_size_per_layer_input": 262144,
                "kv_sharing_layer_count": 42,
            },
            [0] * 42,
        ),
        (
            _moe_26b_config(),
            {
                "window_size": 1024,
                "partial_rotary_factor": 0.5,
                "k_eq_v": 1,
                "num_experts": 128,
                "moe_top_k": 8,
                "moe_intermediate_size": 704,
            },
            [0] * 30,
        ),
        (
            _vision_config(),
            {
                "window_size": 1024,
                "partial_rotary_factor": 0.5,
                "k_eq_v": 1,
                "num_vision_layers": 27,
                "vision_hidden_size": 1152,
                "vision_num_heads": 16,
                "vision_intermediate_size": 4304,
                "image_token_id": 255999,
            },
            [0, 0, 1, 0],
        ),
    ],
)
def test_parse_gemma4_architecture_overrides(
    config: dict[str, object],
    expected_overrides: dict[str, int | float],
    expected_layer_types: list[int],
    tmp_path: Path,
) -> None:
    _write_config(tmp_path, config)
    overrides, layer_types = _parse_gemma4_architecture(tmp_path)

    for key, expected in expected_overrides.items():
        assert overrides[key] == expected
    assert layer_types == expected_layer_types


def test_parse_gemma4_architecture_defaults_without_config(tmp_path: Path) -> None:
    overrides, layer_types = _parse_gemma4_architecture(tmp_path)
    assert overrides == {}
    assert layer_types == []


def test_parse_gemma4_architecture_defaults_for_missing_fields(tmp_path: Path) -> None:
    _write_config(tmp_path, {"model_type": "gemma4"})
    overrides, layer_types = _parse_gemma4_architecture(tmp_path)
    assert overrides == {"window_size": 1024, "partial_rotary_factor": 0.5, "k_eq_v": 0}
    assert layer_types == []


def test_parse_gemma4_architecture_accepts_sliding_window_alias(tmp_path: Path) -> None:
    _write_config(tmp_path, {"sliding_window": 512})
    overrides, _ = _parse_gemma4_architecture(tmp_path)
    assert overrides["window_size"] == 512


@pytest.mark.parametrize(
    "case",
    [
        (4, ["sliding"] * 4, 1024, 8192, 8, 128, 4 * 1024 * 8 * 128 * 4 * 2),
        (4, ["full"] * 4, 1024, 8192, 8, 128, 4 * 8192 * 8 * 128 * 4 * 2),
        (4, ["sliding", "sliding", "full", "sliding"], 512, 4096, 4, 64, (3 * 512 + 4096) * 4 * 64 * 4 * 2),
        (1, ["sliding"], 512, 8192, 1, 64, 512 * 64 * 4 * 2),
        (1, ["full"], 512, 4096, 1, 64, 4096 * 64 * 4 * 2),
        (0, [], 1024, 8192, 8, 128, 0),
    ],
)
def test_compute_kv_cache_memory(case: KVMemoryCase) -> None:
    num_layers, layer_types, window_size, max_context_len, num_kv_heads, head_dim, expected = case
    assert (
        compute_kv_cache_memory(
            num_layers=num_layers,
            layer_types=layer_types,
            window_size=window_size,
            max_context_len=max_context_len,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        == expected
    )


def test_compute_kv_cache_memory_realistic_31b_is_smaller_than_uniform_cache() -> None:
    layer_types = ["full" if i % 6 == 5 else "sliding" for i in range(60)]
    num_full = sum(1 for layer_type in layer_types if layer_type == "full")
    num_sliding = 60 - num_full

    result = compute_kv_cache_memory(
        num_layers=60, layer_types=layer_types, window_size=1024, max_context_len=8192, num_kv_heads=8, head_dim=256
    )
    kv_stride = 8 * 256
    expected = (num_sliding * 1024 + num_full * 8192) * kv_stride * 4 * 2
    uniform = 60 * 8192 * kv_stride * 4 * 2

    assert result == expected
    assert result < uniform


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (
            lambda: compute_kv_cache_memory(
                num_layers=4,
                layer_types=["sliding"] * 3,
                window_size=1024,
                max_context_len=8192,
                num_kv_heads=8,
                head_dim=128,
            ),
            "layer_types length",
        ),
        (
            lambda: compute_kv_cache_memory(
                num_layers=4,
                layer_types=["sliding"] * 4,
                window_size=1024,
                max_context_len=512,
                num_kv_heads=8,
                head_dim=128,
            ),
            "max_context_len",
        ),
    ],
)
def test_compute_kv_cache_memory_rejects_invalid_inputs(factory: Callable[[], int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        factory()

"""Unit tests for the official Gemma 4 12B nested config shape."""

from __future__ import annotations

import json
from pathlib import Path

from mogemma.model import Gemma4Variant, _detect_gemma4_variant, _parse_gemma4_architecture


GEMMA4_12B_CONFIG = {
    "architectures": ["Gemma4UnifiedForConditionalGeneration"],
    "model_type": "gemma4_unified",
    "image_token_id": 258880,
    "audio_token_id": 258881,
    "text_config": {
        "model_type": "gemma4_unified_text",
        "hidden_size": 3840,
        "intermediate_size": 15360,
        "num_hidden_layers": 48,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "num_global_key_value_heads": 1,
        "head_dim": 256,
        "global_head_dim": 512,
        "attention_k_eq_v": True,
        "sliding_window": 1024,
        "max_position_embeddings": 262144,
        "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
        "rope_parameters": {
            "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0},
            "sliding_attention": {"rope_theta": 10000.0},
        },
    },
}


def test_12b_real_config_detection(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(GEMMA4_12B_CONFIG))

    assert _detect_gemma4_variant(tmp_path) is Gemma4Variant.DENSE_12B_UNIFIED


def test_12b_real_config_overrides(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(GEMMA4_12B_CONFIG))

    overrides, layer_types = _parse_gemma4_architecture(tmp_path)

    assert overrides["hidden_size"] == 3840
    assert overrides["head_dim"] == 256
    assert overrides["global_head_dim"] == 512
    assert overrides["num_global_key_value_heads"] == 1
    assert overrides["max_seq_len"] == 262144
    assert overrides["window_size"] == 1024
    assert overrides["partial_rotary_factor"] == 0.25
    assert overrides["image_token_id"] == 258880
    assert overrides["audio_token_id"] == 258881
    assert layer_types == [0, 0, 0, 0, 0, 1]

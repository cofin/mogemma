"""Model wrappers for Gemma 4 inference."""

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Generator, Sequence
from enum import Enum
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from .backends import DeviceSelection, EmbeddingBackend, GenerationBackend, resolve_device_selection
from .backends import resolve_embedding_backend as _resolve_embedding_backend_impl
from .backends import resolve_generation_backend as _resolve_generation_backend_impl
from .config import EmbeddingConfig, GenerationConfig
from .hub import HubManager
from .hydration import ImageHydrator
from .loader import ModelLoader, auto_loader
from .telemetry import tracer
from .typing import SENTENCEPIECE_INSTALLED, _SPProcessorImpl

try:
    from . import _core
except ImportError:
    # Keep import optional for development/testing environments.
    _core = None


class _EncodedToken(Protocol):
    ids: Sequence[int]


class _Tokenizer:
    """Wrapper for sentencepiece to match internal tokenizer needs."""

    def __init__(self, model_path: str) -> None:
        if _SPProcessorImpl is None:
            msg = "sentencepiece runtime is unavailable"
            raise ModuleNotFoundError(msg)
        self.sp = _SPProcessorImpl(model_file=model_path)
        self._max_length: int | None = None

    def enable_truncation(self, *, max_length: int, **_: object) -> None:
        self._max_length = max_length

    def enable_padding(self, **_: object) -> None:
        pass

    def encode_batch(self, text: Sequence[str]) -> Sequence[_EncodedToken]:
        return [self.encode(t) for t in text]

    def encode(self, text: str) -> _EncodedToken:
        ids = self.sp.EncodeAsIds(text)
        if self._max_length is not None:
            ids = ids[: self._max_length]

        class _Result:
            def __init__(self, ids: list[int]) -> None:
                self.ids = ids

        return cast("_EncodedToken", _Result(ids))

    def decode(self, ids: Sequence[int]) -> str:
        return str(self.sp.DecodeIds(list(ids)))

    def token_to_id(self, token: str) -> int | None:
        token_id = self.sp.piece_to_id(token)
        if token_id == self.sp.unk_id() and token not in ("<unk>", " "):
            return None
        return int(token_id)


_EXPECTED_MATRIX_DIMS = 2
_BOS_TOKEN_ID = 2
_EOS_TOKEN_ID_ALIASES = ("<end_of_turn>", "</s>", "<eos>", "<|eos|>")
_TURN_START = "<start_of_turn>"
_TURN_END = "<end_of_turn>"


class Gemma4Variant(str, Enum):
    """Enumeration of supported Gemma 4 architectural variants."""

    DENSE_31B = "gemma4_dense_31b"
    DENSE_12B_UNIFIED = "gemma4_dense_12b_unified"
    DENSE_E2B = "gemma4_dense_e2b"
    DENSE_E4B = "gemma4_dense_e4b"
    MOE_26B_A4B = "gemma4_moe_26b"


_GEMMA4_12B_UNSUPPORTED_RUNTIME_MESSAGE = "Gemma 4 12B unified multimodal runtime is recognized but not implemented."
_AUDIO_UNSUPPORTED_RUNTIME_MESSAGE = "Audio input is recognized but not implemented for this Gemma 4 runtime."
_ConfigScalar = int | float | str


def _resolve_model_path(raw_model_path: str | Path, cache_path: str | Path | None = None) -> Path:
    """Resolve user-supplied model input consistently for all model types."""
    return HubManager(cache_path=cache_path).resolve_model(str(raw_model_path), download_if_missing=True, strict=True)


def _core_unavailable_message(model_type: str) -> str:
    if model_type == "embedding":
        return (
            "Mojo core is unavailable for embeddings. "
            "Build/install the `mogemma._core` extension before calling embed()/embed_tokens()."
        )
    return (
        "Mojo core is unavailable for text generation. "
        "Build/install the `mogemma._core` extension before calling generate()."
    )


def _detect_gemma4_variant(model_dir: Path) -> Gemma4Variant:
    """Classify Gemma 4 model variant from HuggingFace config.json."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        msg = f"No config.json found in {model_dir}"
        raise FileNotFoundError(msg)

    config = json.loads(config_path.read_text())

    architectures = config.get("architectures", [])
    if (
        config.get("model_type") == "gemma4_unified"
        or config.get("text_config", {}).get("model_type") == "gemma4_unified_text"
        or (isinstance(architectures, list) and "Gemma4UnifiedForConditionalGeneration" in architectures)
    ):
        return Gemma4Variant.DENSE_12B_UNIFIED

    if config.get("num_local_experts", config.get("num_experts", 0)) > 0:
        return Gemma4Variant.MOE_26B_A4B

    if "hidden_size_per_layer_input" in config:
        if config.get("use_double_wide_mlp", False):
            return Gemma4Variant.DENSE_E2B
        return Gemma4Variant.DENSE_E4B

    return Gemma4Variant.DENSE_31B


def _raise_for_unsupported_gemma4_runtime(model_path: Path | None) -> None:
    if model_path is None:
        return
    try:
        variant = _detect_gemma4_variant(model_path)
    except FileNotFoundError:
        return
    if variant is Gemma4Variant.DENSE_12B_UNIFIED:
        raise RuntimeError(_GEMMA4_12B_UNSUPPORTED_RUNTIME_MESSAGE)


def _gemma4_text_config(config: dict[str, object]) -> dict[str, object]:
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        return cast("dict[str, object]", text_config)
    return config


def _config_int(value: object) -> int:
    return int(cast("_ConfigScalar", value))


def _config_float(value: object) -> float:
    return float(cast("_ConfigScalar", value))


def _gemma4_full_rope_config(text_config: dict[str, object]) -> dict[str, object]:
    rope_params = text_config.get("rope_parameters")
    if not isinstance(rope_params, dict):
        return {}
    full_rope = rope_params.get("full_attention")
    if isinstance(full_rope, dict):
        return cast("dict[str, object]", full_rope)
    return {}


def _layer_type_to_int(layer_type: object) -> int:
    return 1 if layer_type in {"full", "full_attention"} else 0


def _gemma4_text_overrides(text_config: dict[str, object]) -> dict[str, int]:
    text_overrides = {
        "hidden_size": "hidden_size",
        "intermediate_size": "intermediate_size",
        "num_hidden_layers": "num_layers",
        "num_attention_heads": "num_heads",
        "num_key_value_heads": "num_kv_heads",
        "num_global_key_value_heads": "num_global_key_value_heads",
        "head_dim": "head_dim",
        "global_head_dim": "global_head_dim",
        "max_position_embeddings": "max_seq_len",
    }
    overrides: dict[str, int] = {}
    for config_key, override_key in text_overrides.items():
        value = text_config.get(config_key)
        if value is not None:
            overrides[override_key] = _config_int(value)
    return overrides


def _gemma4_layer_types(text_config: dict[str, object]) -> list[int]:
    layer_types_raw = text_config.get("layer_types", [])
    if not isinstance(layer_types_raw, list):
        return []
    return [_layer_type_to_int(layer_type) for layer_type in layer_types_raw]


def _token_id_override(config: dict[str, object], canonical_key: str, legacy_key: str) -> int | None:
    token_id = config.get(canonical_key)
    if token_id is None:
        token_id = config.get(legacy_key)
    if token_id is None:
        return None
    return _config_int(token_id)


def _parse_gemma4_architecture(model_dir: Path) -> tuple[dict[str, int | float], list[int]]:
    """Extract Gemma 4 architecture fields from config.json for Mojo init.

    Returns:
        A tuple of (overrides_dict, layer_types_list).
        layer_types_list is a list of ints (0=sliding, 1=full), empty if not in config.
    """
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return {}, []

    config = json.loads(config_path.read_text())
    text_config = _gemma4_text_config(config)

    overrides: dict[str, int | float] = {}

    # Sliding window size
    window_size = text_config.get("sliding_window_size", text_config.get("sliding_window", 1024))
    overrides["window_size"] = _config_int(window_size)

    # Partial rotary factor for full attention layers
    full_rope = _gemma4_full_rope_config(text_config)
    partial_rotary_factor = full_rope.get("partial_rotary_factor", text_config.get("partial_rotary_factor", 0.5))
    overrides["partial_rotary_factor"] = _config_float(partial_rotary_factor)

    # K=V weight sharing (1=enabled, 0=disabled)
    k_eq_v = text_config.get("attention_k_eq_v", False)
    overrides["k_eq_v"] = 1 if k_eq_v else 0

    overrides.update(_gemma4_text_overrides(text_config))

    # Layer types: convert ["sliding", "full", ...] to [0, 1, ...]
    layer_types = _gemma4_layer_types(text_config)

    # Vision config (if present)
    vision_config = config.get("vision_config")
    if isinstance(vision_config, dict):
        overrides["num_vision_layers"] = int(vision_config.get("num_hidden_layers", 0))
        overrides["vision_hidden_size"] = int(vision_config.get("hidden_size", 0))
        overrides["vision_num_heads"] = int(vision_config.get("num_attention_heads", 0))
        overrides["vision_intermediate_size"] = int(vision_config.get("intermediate_size", 0))

    # Image token ID
    image_token_id = _token_id_override(config, "image_token_id", "image_token_index")
    if image_token_id is not None:
        overrides["image_token_id"] = image_token_id

    # PLE (E2B/E4B)  # noqa: ERA001
    ple_dim = config.get("hidden_size_per_layer_input")
    if ple_dim is not None:
        overrides["hidden_size_per_layer_input"] = int(ple_dim)
        overrides["vocab_size_per_layer_input"] = int(config.get("vocab_size_per_layer_input", 262144))

    # Double-wide MLP (E2B only)
    if config.get("use_double_wide_mlp", False):
        overrides["use_double_wide_mlp"] = 1

    # KV sharing (E2B/E4B)
    kv_sharing_map = config.get("kv_sharing_layer_map")
    if isinstance(kv_sharing_map, list):
        overrides["kv_sharing_layer_count"] = len(kv_sharing_map)

    # Audio token ID
    audio_token_id = _token_id_override(config, "audio_token_id", "audio_token_index")
    if audio_token_id is not None:
        overrides["audio_token_id"] = audio_token_id

    # MoE
    num_experts = config.get("num_local_experts", config.get("num_experts", 0))
    if num_experts > 0:
        overrides["num_experts"] = int(num_experts)
        overrides["moe_top_k"] = int(config.get("num_experts_per_tok", 8))
        moe_intermediate = config.get("moe_intermediate_size", 704)
        overrides["moe_intermediate_size"] = int(moe_intermediate)

    return overrides, layer_types


def compute_kv_cache_memory(  # noqa: PLR0913
    num_layers: int, layer_types: list[str], window_size: int, max_context_len: int, num_kv_heads: int, head_dim: int
) -> int:
    """Compute total bytes for the hybrid KV cache (K + V arenas combined).

    Args:
        num_layers: Total number of transformer layers.
        layer_types: Per-layer type strings ("sliding" or "full").
        window_size: Sliding-window size in tokens.
        max_context_len: Maximum context length for full-attention layers.
        num_kv_heads: Number of KV heads per layer.
        head_dim: Dimension per attention head.

    Returns:
        Total bytes for both K and V caches combined.

    Raises:
        ValueError: If max_context_len < window_size or layer_types length mismatch.
    """
    if len(layer_types) != num_layers:
        msg = f"layer_types length ({len(layer_types)}) != num_layers ({num_layers})"
        raise ValueError(msg)
    if max_context_len < window_size:
        msg = f"max_context_len ({max_context_len}) must be >= window_size ({window_size})"
        raise ValueError(msg)

    kv_stride = num_kv_heads * head_dim
    total_elements = 0
    for lt in layer_types:
        if lt == "full":
            total_elements += max_context_len * kv_stride
        else:
            total_elements += window_size * kv_stride

    # Float32 = 4 bytes, x2 for K and V
    return total_elements * 4 * 2


def _normalize_architecture_overrides(overrides: dict[str, int | float] | None) -> dict[str, int | float] | None:
    if overrides is None:
        return None
    normalized: dict[str, int | float] = {}
    for key, value in overrides.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = "architecture_overrides values must be numeric (int|float)"
            raise TypeError(msg)
        normalized[key] = value
    return normalized


def _invoke_init_model_with_options(
    _core: object,
    metadata: dict[str, tuple[int, tuple[int, ...], str]],
    overrides: dict[str, object],
    descriptor: dict[str, object],
) -> object:
    init_model_with_options = getattr(_core, "init_model_with_options", None)
    if callable(init_model_with_options):
        return init_model_with_options(metadata, overrides, descriptor)
    return None


def _invoke_legacy_init_model(
    _core: object,
    metadata: dict[str, tuple[int, tuple[int, ...], str]],
    overrides: dict[str, int | float] | None,
    backend: GenerationBackend | EmbeddingBackend,
) -> object:
    if overrides is None:
        return backend.init_model(metadata)

    init_model_fn = getattr(_core, "init_model", None)
    if not callable(init_model_fn):
        msg = "Mojo core does not expose init_model(metadata, architecture_overrides)"
        raise TypeError(msg)
    try:
        return init_model_fn(metadata, overrides)
    except TypeError as exc:
        msg = (
            "Mojo core init_model does not accept architecture_overrides. "
            "Rebuild/install a compatible mogemma._core extension."
        )
        raise RuntimeError(msg) from exc


def _initialize_llm(  # noqa: PLR0913
    loader: ModelLoader,
    backend: GenerationBackend | EmbeddingBackend,
    *,
    device_selection: DeviceSelection,
    model_type: str,
    architecture_overrides: dict[str, int | float] | None = None,
    model_path: Path | None = None,
) -> object:
    _raise_for_unsupported_gemma4_runtime(model_path)

    if _core is None:
        raise RuntimeError(_core_unavailable_message(model_type))

    metadata = loader.get_tensor_metadata()

    # Merge Gemma 4 config.json fields into architecture overrides
    merged_overrides: dict[str, int | float] = {}
    layer_types: list[int] = []
    if model_path is not None:
        parsed_overrides, layer_types = _parse_gemma4_architecture(model_path)
        merged_overrides.update(parsed_overrides)
    if architecture_overrides is not None:
        merged_overrides.update(architecture_overrides)  # user overrides win
    normalized_overrides = _normalize_architecture_overrides(merged_overrides or None)
    descriptor = device_selection.as_runtime_descriptor()

    # Build the overrides dict for Mojo, adding layer_types as a native list
    mojo_overrides: dict[str, object] = dict(normalized_overrides or {})
    if layer_types:
        mojo_overrides["layer_types"] = layer_types

    try:
        llm = _invoke_init_model_with_options(_core, metadata, mojo_overrides, descriptor)
        if llm is None:
            llm = _invoke_legacy_init_model(_core, metadata, normalized_overrides, backend)
    except ValueError:
        raise
    except Exception as exc:
        msg = f"{model_type} model failed to initialize from '{loader.model_path}': {exc}"
        raise RuntimeError(msg) from exc

    if isinstance(llm, dict):
        llm.setdefault("device_selection", descriptor)
        llm.setdefault("device_backend", device_selection.backend)
        llm.setdefault("device_kind", device_selection.device_kind)
        llm.setdefault("device_index", device_selection.device_index)
        llm.setdefault("device_request", device_selection.requested)
        llm.setdefault("device_availability_source", device_selection.availability_source)
        llm.setdefault("device_strict", device_selection.strict)
        if normalized_overrides:
            llm.setdefault("architecture_overrides", normalized_overrides)
    return llm


def _resolve_generation_backend(device: str) -> GenerationBackend:
    if _core is None:
        raise RuntimeError(_core_unavailable_message("generation"))
    return _resolve_generation_backend_impl(device=device, core_module=_core)


def _resolve_embedding_backend(device: str) -> EmbeddingBackend:
    if _core is None:
        raise RuntimeError(_core_unavailable_message("embedding"))
    return _resolve_embedding_backend_impl(device=device, core_module=_core)


def _sample_next_token(logits: npt.ArrayLike, *, temperature: float, top_k: int, top_p: float) -> int:
    """Sample the next token ID from backend logits."""
    logits_array = np.asarray(logits, dtype=np.float64).reshape(-1)
    if logits_array.size == 0:
        msg = "backend returned empty logits"
        raise ValueError(msg)

    if temperature <= 0:
        return int(np.argmax(logits_array))

    filtered_logits = logits_array / temperature

    if 0 < top_k < filtered_logits.size:
        top_indices = np.argpartition(filtered_logits, -top_k)[-top_k:]
        top_mask = np.zeros(filtered_logits.size, dtype=bool)
        top_mask[top_indices] = True
        filtered_logits = np.where(top_mask, filtered_logits, -np.inf)

    if 0.0 < top_p < 1.0:
        sorted_indices = np.argsort(filtered_logits)[::-1]
        sorted_logits = filtered_logits[sorted_indices]
        finite_mask = np.isfinite(sorted_logits)

        if finite_mask.any():
            finite_logits = sorted_logits[finite_mask]
            finite_logits = finite_logits - np.max(finite_logits)
            finite_probs = np.exp(finite_logits)
            finite_probs = finite_probs / finite_probs.sum()

            cumulative = np.cumsum(finite_probs)
            overflow = cumulative > top_p
            if overflow.any():
                first_overflow = int(np.argmax(overflow))
                remove_indices = sorted_indices[finite_mask][first_overflow + 1 :]
                filtered_logits[remove_indices] = -np.inf

    if not np.isfinite(filtered_logits).any():
        return int(np.argmax(logits_array))

    stable_logits = filtered_logits - np.max(filtered_logits)
    probs = np.exp(stable_logits)
    probs_sum = probs.sum()

    if probs_sum <= 0 or not np.isfinite(probs_sum):
        return int(np.argmax(logits_array))

    probs = probs / probs_sum
    return int(np.random.default_rng().choice(probs.size, p=probs))


def _normalize_eos_token_id(tokenizer: _Tokenizer) -> int:
    """Return an EOS token id from tokenizer aliases."""
    for eos_token in _EOS_TOKEN_ID_ALIASES:
        eos_token_id = tokenizer.token_to_id(eos_token)
        if eos_token_id is not None:
            return int(eos_token_id)
    return 0


def _is_instruction_tuned_model(model_path: Path, config_model_path: str | Path) -> bool:
    """Return True when the configured model looks like an instruction-tuned Gemma variant."""
    configured = str(config_model_path).lower()
    resolved = model_path.name.lower()
    return configured.endswith("-it") or resolved.endswith("-it")


def _format_gemma4_prompt(prompt: str | list[dict[str, str]], *, system_prompt: str | None = None) -> str:
    """Format prompt using Gemma 4 chat template.

    Supports plain strings, system prompts, and multi-turn message lists.
    """
    if isinstance(prompt, list):
        return _format_messages(prompt)

    if _TURN_START in prompt or _TURN_END in prompt:
        return prompt

    parts: list[str] = []
    if system_prompt is not None:
        parts.append(f"{_TURN_START}system\n{system_prompt}\n{_TURN_END}\n")
    parts.append(f"{_TURN_START}user\n{prompt}\n{_TURN_END}\n{_TURN_START}model\n")
    return "".join(parts)


def _format_messages(messages: list[dict[str, str]]) -> str:
    """Format a list of role/content messages into Gemma 4 chat template."""
    parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"{_TURN_START}{role}\n{content}\n{_TURN_END}\n")
    parts.append(f"{_TURN_START}model\n")
    return "".join(parts)


def _reset_llm_session_state(llm: object) -> None:
    """Reset mutable session fields before a new generation run."""
    if not isinstance(llm, dict):
        return

    llm["pos"] = 0
    if _core is not None and hasattr(_core, "reset_cache"):
        _core.reset_cache(llm)


class SyncEmbeddingModel:
    """Python interface for the Gemma 4 embedding engine."""

    def __init__(self, config: EmbeddingConfig | str | None = None, tokenizer: _Tokenizer | None = None) -> None:
        """Initialize the embedding model.

        Args:
            config: Model ID string, ``EmbeddingConfig``, or ``None`` for defaults.
            tokenizer: Optional pre-built tokenizer instance.
        """
        if config is None:
            config = EmbeddingConfig()
        elif isinstance(config, str):
            config = EmbeddingConfig(model_path=config)
        self.config = config
        self._tokenizer = tokenizer

        self._device_selection: DeviceSelection = resolve_device_selection(config.device)
        # Resolve model path (Hub or local)
        self.model_path = _resolve_model_path(config.model_path, config.cache_path)
        self._loader = auto_loader(self.model_path)
        self._backend = _resolve_embedding_backend(self._device_selection.effective_device)
        self._backend_id = self._backend.backend_id

        # Initialize Mojo core
        self._llm: object | None = _initialize_llm(
            self._loader,
            self._backend,
            device_selection=self._device_selection,
            model_type="embedding",
            architecture_overrides=config.architecture_overrides,
            model_path=self.model_path,
        )

    def _ensure_tokenizer(self) -> _Tokenizer:
        if self._tokenizer is not None:
            return self._tokenizer
        if not SENTENCEPIECE_INSTALLED:
            msg = (
                "Text tokenization requires 'sentencepiece' dependency. "
                "Install with: pip install 'mogemma[llm]' or call embed_tokens(...) with pre-tokenized inputs."
            )
            raise ModuleNotFoundError(msg)

        sp_path = self.model_path / "tokenizer.model"
        if sp_path.exists():
            self._tokenizer = _Tokenizer(str(sp_path))
            return self._tokenizer

        msg = f"No tokenizer.model found in {self.model_path}"
        raise FileNotFoundError(msg)

    def _embed_token_array(self, tokens: Sequence[Sequence[int]], input_count: int) -> npt.NDArray[np.float32]:
        if self._llm is None:
            raise RuntimeError(_core_unavailable_message("embedding"))

        if input_count > 0:
            expected_len = len(tokens[0])
            if any(len(row) != expected_len for row in tokens):
                msg = (
                    "all token sequences in a batch must have the exact same length "
                    "(padding is required for batch inference)"
                )
                raise ValueError(msg)

        raw_embeddings = self._backend.generate_embeddings(self._llm, tokens)
        embeddings = np.asarray(raw_embeddings, dtype=np.float32)

        if embeddings.ndim != _EXPECTED_MATRIX_DIMS:
            msg = f"expected 2D embeddings from backend, got {embeddings.ndim}D"
            raise ValueError(msg)
        if embeddings.shape[0] != input_count:
            msg = f"backend returned {embeddings.shape[0]} embedding rows for {input_count} inputs"
            raise ValueError(msg)
        return embeddings

    def embed(self, text: str | list[str]) -> npt.NDArray[np.float32]:
        """Generate embeddings for text by tokenizing in Python, then running Mojo inference."""
        with tracer.start_as_current_span("SyncEmbeddingModel.embed") as span:
            span.set_attribute("device", self._device_selection.requested)
            span.set_attribute("backend", self._device_selection.backend)
            if isinstance(text, str):
                text = [text]
            if not text:
                msg = "text input must contain at least one value"
                raise ValueError(msg)

            span.set_attribute("text_count", len(text))

        tokenizer = self._ensure_tokenizer()
        tokenizer.enable_truncation(max_length=self.config.max_sequence_length)
        tokenizer.enable_padding()
        encoded = tokenizer.encode_batch(text)
        tokens = [e.ids for e in encoded]
        return self._embed_token_array(tokens, len(text))

    def embed_tokens(self, tokens: Sequence[Sequence[int]] | npt.NDArray[np.int32]) -> npt.NDArray[np.float32]:
        """Generate embeddings directly from pre-tokenized IDs using Mojo inference."""
        token_list = []
        if isinstance(tokens, np.ndarray):
            if tokens.ndim != _EXPECTED_MATRIX_DIMS:
                msg = f"tokens must be a 2D array of token IDs, got shape {tokens.shape}"
                raise ValueError(msg)
            token_list = tokens.tolist()
        else:
            token_list = list(tokens)

        if len(token_list) == 0:
            msg = "tokens must contain at least one row"
            raise ValueError(msg)
        return self._embed_token_array(token_list, len(token_list))

    @property
    def tokenizer(self) -> _Tokenizer:
        """Access the tokenizer (loads lazily)."""
        return self._ensure_tokenizer()

    def close(self) -> None:
        """Release underlying resources and free the memory arena."""
        if hasattr(self, "_llm") and self._llm is not None and _core is not None and hasattr(_core, "free_arena"):
            _core.free_arena(self._llm)
            self._llm = None
        if hasattr(self, "_loader"):
            self._loader.close()

    def __del__(self) -> None:
        """Cleanup on garbage collection."""
        self.close()

    def __enter__(self) -> Self:
        """Enter the context manager."""
        return self

    def __exit__(self, *args: object) -> None:
        """Exit the context manager and release resources."""
        self.close()


class SyncGemmaModel:
    """Python interface for the Gemma 4 text generation engine."""

    def __init__(self, config: GenerationConfig | str | None = None, tokenizer: _Tokenizer | None = None) -> None:
        """Initialize the text model.

        Args:
            config: Model ID string, ``GenerationConfig``, or ``None`` for defaults.
            tokenizer: Optional pre-built tokenizer instance.
        """
        if config is None:
            config = GenerationConfig()
        elif isinstance(config, str):
            config = GenerationConfig(model_path=config)
        self.config = config
        self._tokenizer = tokenizer

        self._device_selection: DeviceSelection = resolve_device_selection(config.device)
        # Resolve model path (Hub or local)
        self.model_path = _resolve_model_path(config.model_path, config.cache_path)
        self._loader = auto_loader(self.model_path)
        self._instruction_tuned = _is_instruction_tuned_model(self.model_path, config.model_path)
        self._backend = _resolve_generation_backend(self._device_selection.effective_device)
        self._backend_id = self._backend.backend_id

        # Initialize Mojo core
        self._llm: object | None = _initialize_llm(
            self._loader,
            self._backend,
            device_selection=self._device_selection,
            model_type="generation",
            architecture_overrides=config.architecture_overrides,
            model_path=self.model_path,
        )

    def _ensure_tokenizer(self) -> _Tokenizer:
        if self._tokenizer is not None:
            return self._tokenizer
        if not SENTENCEPIECE_INSTALLED:
            msg = "Text generation requires 'sentencepiece' dependency. Install with: pip install 'mogemma[llm]'"
            raise ModuleNotFoundError(msg)

        sp_path = self.model_path / "tokenizer.model"
        if sp_path.exists():
            self._tokenizer = _Tokenizer(str(sp_path))
            return self._tokenizer

        msg = f"No tokenizer.model found in {self.model_path}"
        raise FileNotFoundError(msg)

    def _get_image_token_id(self) -> int | None:
        """Get the image placeholder token ID from the LLM config, if vision is enabled."""
        if isinstance(self._llm, dict):
            token_id = self._llm.get("image_token_id", 0)
            if token_id and int(token_id) > 0:
                return int(token_id)
        return None

    def _get_audio_token_id(self) -> int | None:
        """Get the audio placeholder token ID from the LLM config, if audio is enabled."""
        if isinstance(self._llm, dict):
            token_id = self._llm.get("audio_token_id", 0)
            if token_id and int(token_id) > 0:
                return int(token_id)
        return None

    def generate(
        self,
        prompt: str,
        images: Sequence[str | Path | bytes | npt.NDArray[np.generic]] | None = None,
        audio: Sequence[str | Path | bytes | npt.NDArray[np.generic]] | None = None,
        *,
        system_prompt: str | None = None,
    ) -> str:
        """Generate text from the given prompt."""
        return "".join(list(self.generate_stream(prompt, images=images, audio=audio, system_prompt=system_prompt)))

    def generate_stream(  # noqa: C901, PLR0912, PLR0915
        self,
        prompt: str,
        images: Sequence[str | Path | bytes | npt.NDArray[np.generic]] | None = None,
        audio: Sequence[str | Path | bytes | npt.NDArray[np.generic]] | None = None,
        *,
        system_prompt: str | None = None,
    ) -> Generator[str, None, None]:
        """Generate text as a stream of tokens."""
        if audio is not None:
            raise RuntimeError(_AUDIO_UNSUPPORTED_RUNTIME_MESSAGE)

        tokenizer = self._ensure_tokenizer()
        prompt_to_encode = (
            _format_gemma4_prompt(prompt, system_prompt=system_prompt) if self._instruction_tuned else prompt
        )
        with tracer.start_as_current_span("SyncGemmaModel.generate_stream") as span:
            span.set_attribute("device", self._device_selection.requested)
            span.set_attribute("backend", self._device_selection.backend)
            span.set_attribute("prompt_length", len(prompt))
            tokenizer.enable_truncation(max_length=self.config.max_sequence_length)
            tokenizer.enable_padding()
            encoded = tokenizer.encode(prompt_to_encode)
            tokens = encoded.ids
            if not tokens or tokens[0] != _BOS_TOKEN_ID:
                tokens = [_BOS_TOKEN_ID, *list(tokens)]

        if self._llm is None:
            raise RuntimeError(_core_unavailable_message("generation"))

        _reset_llm_session_state(self._llm)

        # Process images through vision encoder if provided
        image_token_id = self._get_image_token_id()
        vision_embeddings: list[object] = []
        if images is not None:
            hydrated = ImageHydrator().hydrate(images)
            self._backend.process_images(self._llm, hydrated)
            if isinstance(self._llm, dict):
                vision_embeddings = list(self._llm.get("vision_embeddings", []))

        # Process audio through audio encoder if provided
        audio_token_id = self._get_audio_token_id()
        audio_embeddings: list[object] = []
        if audio is not None:
            from .hydration import AudioHydrator  # noqa: PLC0415

            audio_hydrated = AudioHydrator().hydrate(audio)
            self._backend.process_audio(self._llm, audio_hydrated)
            if isinstance(self._llm, dict):
                audio_embeddings = list(self._llm.get("audio_embeddings", []))

        # Prefill with token merging: replace <image>/<audio> placeholders with embeddings
        vision_idx = 0
        audio_idx = 0
        for t in tokens[:-1]:
            tok = int(t)
            if image_token_id is not None and tok == image_token_id and vision_idx < len(vision_embeddings):
                self._backend.step_with_embedding(self._llm, vision_embeddings[vision_idx])
                vision_idx += 1
            elif audio_token_id is not None and tok == audio_token_id and audio_idx < len(audio_embeddings):
                self._backend.step_with_embedding(self._llm, audio_embeddings[audio_idx])
                audio_idx += 1
            else:
                self._backend.step(self._llm, tok, self.config.temperature, self.config.top_k, self.config.top_p)

        if tokens:
            current_token = int(tokens[-1])
        else:
            eos_token_id = _normalize_eos_token_id(tokenizer)
            current_token = int(eos_token_id)

        eos_token_id = _normalize_eos_token_id(tokenizer)
        for _ in range(self.config.max_tokens):
            logits = self._backend.step(
                self._llm, current_token, self.config.temperature, self.config.top_k, self.config.top_p
            )

            next_token = _sample_next_token(
                logits, temperature=self.config.temperature, top_k=self.config.top_k, top_p=self.config.top_p
            )
            if next_token == eos_token_id:
                return

            decoded = tokenizer.decode([next_token])
            if not isinstance(decoded, str):  # pyright: ignore[reportUnnecessaryIsInstance]
                msg = "tokenizer.decode returned non-string output"
                raise TypeError(msg)
            if not decoded:
                return

            yield decoded
            current_token = next_token

    @property
    def tokenizer(self) -> _Tokenizer:
        """Access the tokenizer (loads lazily)."""
        return self._ensure_tokenizer()

    def close(self) -> None:
        """Release underlying resources and free the memory arena."""
        if hasattr(self, "_llm") and self._llm is not None and _core is not None and hasattr(_core, "free_arena"):
            _core.free_arena(self._llm)
            self._llm = None
        if hasattr(self, "_loader"):
            self._loader.close()

    def __del__(self) -> None:
        """Cleanup on garbage collection."""
        self.close()

    def __enter__(self) -> Self:
        """Enter the context manager."""
        return self

    def __exit__(self, *args: object) -> None:
        """Exit the context manager and release resources."""
        self.close()


class AsyncGemmaModel:
    """Asynchronous wrapper for SyncGemmaModel."""

    def __init__(self, config: GenerationConfig | str | None = None) -> None:
        """Initialize the async model.

        Args:
            config: Model ID string, ``GenerationConfig``, or ``None`` for defaults.
        """
        self._model = SyncGemmaModel(config)

    async def generate(
        self,
        prompt: str,
        images: Sequence[str | Path | bytes | npt.NDArray[np.generic]] | None = None,
        audio: Sequence[str | Path | bytes | npt.NDArray[np.generic]] | None = None,
        *,
        system_prompt: str | None = None,
    ) -> str:
        """Generate text asynchronously."""
        return await asyncio.to_thread(self._model.generate, prompt, images, audio, system_prompt=system_prompt)

    async def generate_stream(
        self,
        prompt: str,
        images: Sequence[str | Path | bytes | npt.NDArray[np.generic]] | None = None,
        audio: Sequence[str | Path | bytes | npt.NDArray[np.generic]] | None = None,
        *,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        """Generate text as an async stream of tokens."""
        generator = self._model.generate_stream(prompt, images=images, audio=audio, system_prompt=system_prompt)

        def get_next() -> str | None:
            try:
                return next(generator)
            except StopIteration:
                return None

        try:
            while True:
                token = await asyncio.to_thread(get_next)
                if token is None:
                    break
                yield token
        except asyncio.CancelledError:
            generator.close()
            raise
        finally:
            with contextlib.suppress(RuntimeError):
                generator.close()

    @property
    def tokenizer(self) -> _Tokenizer:
        """Access to the underlying tokenizer."""
        return self._model.tokenizer


class AsyncEmbeddingModel:
    """Asynchronous wrapper for SyncEmbeddingModel."""

    def __init__(self, config: EmbeddingConfig | str | None = None) -> None:
        """Initialize the async embedding model.

        Args:
            config: Model ID string, ``EmbeddingConfig``, or ``None`` for defaults.
        """
        self._model = SyncEmbeddingModel(config)

    async def embed(self, text: str | list[str]) -> npt.NDArray[np.float32]:
        """Generate embeddings for text asynchronously."""
        return await asyncio.to_thread(self._model.embed, text)

    async def embed_tokens(self, tokens: Sequence[Sequence[int]] | npt.NDArray[np.int32]) -> npt.NDArray[np.float32]:
        """Generate embeddings from pre-tokenized IDs asynchronously."""
        return await asyncio.to_thread(self._model.embed_tokens, tokens)

    @property
    def tokenizer(self) -> _Tokenizer:
        """Access to the underlying tokenizer."""
        return self._model.tokenizer

    def close(self) -> None:
        """Release underlying resources."""
        if hasattr(self, "_model"):
            self._model.close()

    def __del__(self) -> None:
        """Cleanup on garbage collection."""
        self.close()

    async def __aenter__(self) -> Self:
        """Enter the async context manager."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit the async context manager and release resources."""
        self.close()

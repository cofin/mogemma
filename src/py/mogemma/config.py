"""Configuration objects for model runtime settings."""

from dataclasses import dataclass, field
from pathlib import Path

_EMPTY_PATH_MSG = "model_path must be a non-empty local path or a valid Google model id"
_EMPTY_CACHE_PATH_MSG = "cache_path must be an absolute or relative directory path"
_EMPTY_SEQUENCE_MSG = "max_sequence_length must be greater than 0"
_INVALID_BATCH_SIZE_MSG = "batch_size must be greater than 0"
_INVALID_TOKENS_MSG = "max_tokens must be greater than 0"
_EMPTY_TOKENIZER_PATH_HINT = "Use an existing local directory or a valid Google model id"
_INVALID_ARCH_OVERRIDES_MSG = "architecture_overrides must be a dict[str, int | float] or None"


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for Gemma 4 Embedding generation."""

    model_path: Path | str = "google/gemma-4-26B-A4B-it"
    """Path to the local Gemma 4 model weights or Google model ID."""

    cache_path: Path | str | None = None
    """Optional base directory for downloaded model cache."""

    device: str = "cpu"
    """Execution device (e.g., 'cpu', 'gpu')."""

    architecture_overrides: dict[str, int | float] | None = None
    """Optional architecture values passed through to Mojo init."""

    max_sequence_length: int = 512
    """Maximum input sequence length."""

    batch_size: int = 1
    """Inference batch size."""

    metadata: dict[str, str] = field(default_factory=dict)
    """Additional model metadata."""

    def __post_init__(self) -> None:
        """Validate configuration."""
        model_path_str = str(self.model_path)
        if not model_path_str:
            raise ValueError(_EMPTY_PATH_MSG)

        if model_path_str in {".", ".."}:
            raise ValueError(_EMPTY_PATH_MSG)
        if self.cache_path is not None and str(self.cache_path) in {"", ".", ".."}:
            raise ValueError(_EMPTY_CACHE_PATH_MSG)

        if self.max_sequence_length <= 0:
            raise ValueError(_EMPTY_SEQUENCE_MSG)
        if self.batch_size <= 0:
            raise ValueError(_INVALID_BATCH_SIZE_MSG)
        _validate_architecture_overrides(self.architecture_overrides)


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for Gemma 4 Text generation."""

    model_path: Path | str = "google/gemma-4-26B-A4B-it"
    """Path to the local Gemma 4 model weights or Google model ID."""

    cache_path: Path | str | None = None
    """Optional base directory for downloaded model cache."""

    device: str = "cpu"
    """Execution device (e.g., 'cpu', 'gpu')."""

    architecture_overrides: dict[str, int | float] | None = None
    """Optional architecture values passed through to Mojo init."""

    max_sequence_length: int = 512
    """Maximum input sequence length."""

    max_tokens: int = 128
    """Maximum number of tokens to generate."""

    temperature: float = 1.0
    """Sampling temperature."""

    top_k: int = 64
    """Top-k sampling parameter."""

    top_p: float = 0.95
    """Top-p (nucleus) sampling parameter."""

    max_image_tokens: int = 560
    """Maximum token budget per image for vision preprocessing."""

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.temperature < 0:
            msg = "temperature must be non-negative"
            raise ValueError(msg)
        if self.top_k < 0:
            msg = "top_k must be non-negative"
            raise ValueError(msg)
        if not (0.0 <= self.top_p <= 1.0):
            msg = "top_p must be between 0.0 and 1.0"
            raise ValueError(msg)

        model_path_str = str(self.model_path)
        if not model_path_str:
            raise ValueError(_EMPTY_PATH_MSG)
        if model_path_str in {".", ".."}:
            raise ValueError(_EMPTY_PATH_MSG)
        if self.cache_path is not None and str(self.cache_path) in {"", ".", ".."}:
            raise ValueError(_EMPTY_CACHE_PATH_MSG)
        if self.max_sequence_length <= 0:
            raise ValueError(_EMPTY_SEQUENCE_MSG)
        if self.max_tokens <= 0:
            raise ValueError(_INVALID_TOKENS_MSG)
        _validate_architecture_overrides(self.architecture_overrides)


def _validate_architecture_overrides(overrides: dict[str, int | float] | None) -> None:
    if overrides is None:
        return
    if not isinstance(overrides, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(_INVALID_ARCH_OVERRIDES_MSG)
    for key, value in overrides.items():
        if not isinstance(key, str) or not key:  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(_INVALID_ARCH_OVERRIDES_MSG)
        if isinstance(value, bool) or not isinstance(value, (int, float)):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(_INVALID_ARCH_OVERRIDES_MSG)

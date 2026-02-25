"""Configuration objects for model runtime settings."""

from dataclasses import dataclass, field
from pathlib import Path

_EMPTY_PATH_MSG = "model_path must be a non-empty local path or a valid Google model id"
_EMPTY_SEQUENCE_MSG = "max_sequence_length must be greater than 0"
_INVALID_BATCH_SIZE_MSG = "batch_size must be greater than 0"
_INVALID_TOKENS_MSG = "max_new_tokens must be greater than 0"
_EMPTY_TOKENIZER_PATH_HINT = "Use an existing local directory or a valid Google model id"


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for Gemma 3 Embedding generation."""

    model_path: Path | str = "gemma3-270m-it"
    """Path to the local Gemma 3 model weights or Google model ID."""

    device: str = "cpu"
    """Execution device (e.g., 'cpu', 'gpu')."""

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

        if self.max_sequence_length <= 0:
            raise ValueError(_EMPTY_SEQUENCE_MSG)
        if self.batch_size <= 0:
            raise ValueError(_INVALID_BATCH_SIZE_MSG)


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for Gemma 3 Text generation."""

    model_path: Path | str = "gemma3-270m-it"
    """Path to the local Gemma 3 model weights or Google model ID."""

    device: str = "cpu"
    """Execution device (e.g., 'cpu', 'gpu')."""

    max_sequence_length: int = 512
    """Maximum input sequence length."""

    max_new_tokens: int = 128
    """Maximum number of tokens to generate."""

    temperature: float = 1.0
    """Sampling temperature."""

    top_k: int = 50
    """Top-k sampling parameter."""

    top_p: float = 1.0
    """Top-p (nucleus) sampling parameter."""

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
        if self.max_sequence_length <= 0:
            raise ValueError(_EMPTY_SEQUENCE_MSG)
        if self.max_new_tokens <= 0:
            raise ValueError(_INVALID_TOKENS_MSG)

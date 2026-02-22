"""Model wrappers for Gemma 3 inference."""

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from .config import EmbeddingConfig, GenerationConfig
from .hub import HubManager
from .telemetry import tracer
from .typing import _TokenizerImpl

try:
    from . import _core
except ImportError:
    # Allow fallback for development/testing if .so is missing
    _core = None


class _EncodedToken(Protocol):
    ids: Sequence[int]


class _Tokenizer(Protocol):
    def enable_truncation(self, *, max_length: int, **_: object) -> None: ...

    def enable_padding(self, **_: object) -> None: ...

    def encode_batch(self, text: Sequence[str]) -> Sequence[_EncodedToken]: ...

    def encode(self, text: str) -> _EncodedToken: ...

    def decode(self, ids: Sequence[int]) -> str: ...

    def token_to_id(self, token: str) -> int | None: ...


_EXPECTED_MATRIX_DIMS = 2


def _initialize_llm(model_path: str, *, model_type: str) -> object:
    if _core is None:
        if model_type == "embedding":
            msg = (
                "Mojo core is unavailable for embeddings. "
                "Build/install the `mogemma._core` extension before calling embed()/embed_tokens()."
            )
        else:
            msg = (
                "Mojo core is unavailable for text generation. "
                "Build/install the `mogemma._core` extension before calling generate()."
            )
        raise RuntimeError(msg)

    try:
        return _core.init_model(model_path)
    except Exception as exc:
        msg = f"{model_type} model failed to initialize from '{model_path}': {exc}"
        raise RuntimeError(msg) from exc


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


class EmbeddingModel:
    """Python interface for the Gemma 3 embedding engine."""

    def __init__(self, config: EmbeddingConfig, tokenizer: _Tokenizer | None = None) -> None:
        """Initialize the embedding model."""
        self.config = config
        self._tokenizer = tokenizer

        # Resolve model path (Hub or local)
        self.model_path = HubManager().resolve_model(
            str(config.model_path),
            download_if_missing=True,
            strict=True,
        )

        # Initialize Mojo core
        self._llm: object | None = _initialize_llm(str(self.model_path), model_type="embedding")

    def _ensure_tokenizer(self) -> _Tokenizer:
        if self._tokenizer is not None:
            return self._tokenizer
        if _TokenizerImpl is None:
            msg = (
                "Text tokenization requires optional dependency 'tokenizers'. "
                "Install with: pip install 'mogemma[text]' or call embed_tokens(...) with pre-tokenized inputs."
            )
            raise ModuleNotFoundError(msg)
        self._tokenizer = cast("_Tokenizer", _TokenizerImpl.from_pretrained(str(self.model_path)))
        return self._tokenizer

    def _embed_token_array(self, tokens: npt.NDArray[np.int32], input_count: int) -> npt.NDArray[np.float32]:
        if _core is None or self._llm is None:
            msg = (
                "Mojo core is unavailable for embeddings. "
                "Build/install the `mogemma._core` extension before calling embed()/embed_tokens()."
            )
            raise RuntimeError(msg)

        raw_embeddings = _core.generate_embeddings(self._llm, tokens)
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
        with tracer.start_as_current_span("EmbeddingModel.embed") as span:
            if isinstance(text, str):
                text = [text]

            span.set_attribute("text_count", len(text))

        tokenizer = self._ensure_tokenizer()
        tokenizer.enable_truncation(max_length=self.config.max_sequence_length)
        tokenizer.enable_padding()
        encoded = tokenizer.encode_batch(text)
        tokens = np.asarray([e.ids for e in encoded], dtype=np.int32)
        return self._embed_token_array(tokens, len(text))

    def embed_tokens(self, tokens: npt.ArrayLike) -> npt.NDArray[np.float32]:
        """Generate embeddings directly from pre-tokenized IDs using Mojo inference."""
        token_array = np.asarray(tokens, dtype=np.int32)
        if token_array.ndim != _EXPECTED_MATRIX_DIMS:
            msg = f"tokens must be a 2D array of token IDs, got shape {token_array.shape}"
            raise ValueError(msg)
        return self._embed_token_array(token_array, int(token_array.shape[0]))

    @property
    def tokenizer(self) -> _Tokenizer:
        """Access the tokenizer (loads lazily)."""
        return self._ensure_tokenizer()


class SyncGemmaModel:
    """Python interface for the Gemma 3 text generation engine."""

    def __init__(self, config: GenerationConfig, tokenizer: _Tokenizer | None = None) -> None:
        """Initialize the text model."""
        self.config = config
        self._tokenizer = tokenizer

        # Resolve model path (Hub or local)
        self.model_path = HubManager().resolve_model(
            str(config.model_path),
            download_if_missing=True,
            strict=True,
        )

        # Initialize Mojo core
        self._llm: object | None = _initialize_llm(str(self.model_path), model_type="generation")

    def _ensure_tokenizer(self) -> _Tokenizer:
        if self._tokenizer is not None:
            return self._tokenizer
        if _TokenizerImpl is None:
            msg = "Text generation requires optional dependency 'tokenizers'. Install with: pip install 'mogemma[text]'"
            raise ModuleNotFoundError(msg)
        self._tokenizer = cast("_Tokenizer", _TokenizerImpl.from_pretrained(str(self.model_path)))
        return self._tokenizer

    def generate(self, prompt: str) -> str:
        """Generate text from the given prompt."""
        return "".join(list(self.generate_stream(prompt)))

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """Generate text as a stream of tokens."""
        tokenizer = self._ensure_tokenizer()
        with tracer.start_as_current_span("SyncGemmaModel.generate_stream") as span:
            span.set_attribute("prompt_length", len(prompt))
            tokenizer.enable_truncation(max_length=self.config.max_sequence_length)
            tokenizer.enable_padding()
            encoded = tokenizer.encode(prompt)
            tokens = encoded.ids

        if _core is None or self._llm is None:
            msg = (
                "Mojo core is unavailable for text generation. "
                "Build/install the `mogemma._core` extension before calling generate()."
            )
            raise RuntimeError(msg)

        if tokens:
            current_token = int(tokens[-1])
        else:
            eos_token_id = tokenizer.token_to_id("<eos>")
            if eos_token_id is None:
                eos_token_id = 0
            current_token = int(eos_token_id)

        for _ in range(self.config.max_new_tokens):
            logits = _core.step(self._llm, current_token, self.config.temperature, self.config.top_k, self.config.top_p)

            next_token = _sample_next_token(
                logits, temperature=self.config.temperature, top_k=self.config.top_k, top_p=self.config.top_p
            )
            decoded = tokenizer.decode([next_token])
            if isinstance(decoded, str):
                yield decoded
            current_token = next_token

    @property
    def tokenizer(self) -> _Tokenizer:
        """Access the tokenizer (loads lazily)."""
        return self._ensure_tokenizer()


class AsyncGemmaModel:
    """Asynchronous wrapper for SyncGemmaModel."""

    def __init__(self, config: GenerationConfig) -> None:
        """Initialize the async model."""
        self._model = SyncGemmaModel(config)

    async def generate(self, prompt: str) -> str:
        """Generate text asynchronously."""
        return await asyncio.to_thread(self._model.generate, prompt)

    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        """Generate text as an async stream of tokens."""
        generator = self._model.generate_stream(prompt)

        def get_next() -> str | None:
            try:
                return next(generator)
            except StopIteration:
                return None

        while True:
            token = await asyncio.to_thread(get_next)
            if token is None:
                break
            yield token

    @property
    def tokenizer(self) -> _Tokenizer:
        """Access to the underlying tokenizer."""
        return self._model.tokenizer

"""Model wrappers for Gemma 3 inference."""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Generator, Sequence
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from .config import EmbeddingConfig, GenerationConfig
from .hub import HubManager
from .loader import ModelLoader, auto_loader
from .telemetry import tracer
from .typing import SENTENCEPIECE_INSTALLED, _SPProcessorImpl

try:
    from . import _core
except ImportError:
    # Allow fallback for development/testing if .so is missing
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
_INSTRUCTION_START = "<start_of_turn>"
_INSTRUCTION_END = "<end_of_turn>"


def _resolve_model_path(raw_model_path: str | Path) -> Path:
    """Resolve user-supplied model input consistently for all model types."""
    return HubManager().resolve_model(str(raw_model_path), download_if_missing=True, strict=True)


def _initialize_llm(loader: ModelLoader, *, model_type: str) -> object:
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
        metadata = loader.get_tensor_metadata()
        return _core.init_model(metadata)
    except Exception as exc:
        msg = f"{model_type} model failed to initialize from '{loader.model_path}': {exc}"
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


def _format_instruction_prompt(prompt: str) -> str:
    """Wrap plain user prompts in Gemma instruction-turn format."""
    if _INSTRUCTION_START in prompt or _INSTRUCTION_END in prompt:
        return prompt
    return f"{_INSTRUCTION_START}user\n{prompt}\n{_INSTRUCTION_END}\n{_INSTRUCTION_START}model\n"


class EmbeddingModel:
    """Python interface for the Gemma 3 embedding engine."""

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

        # Resolve model path (Hub or local)
        self.model_path = _resolve_model_path(config.model_path)
        self._loader = auto_loader(self.model_path)

        # Initialize Mojo core
        self._llm: object | None = _initialize_llm(self._loader, model_type="embedding")

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


class SyncGemmaModel:
    """Python interface for the Gemma 3 text generation engine."""

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

        # Resolve model path (Hub or local)
        self.model_path = _resolve_model_path(config.model_path)
        self._loader = auto_loader(self.model_path)
        self._instruction_tuned = _is_instruction_tuned_model(self.model_path, config.model_path)

        # Initialize Mojo core
        self._llm: object | None = _initialize_llm(self._loader, model_type="generation")

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

    def generate(self, prompt: str) -> str:
        """Generate text from the given prompt."""
        return "".join(list(self.generate_stream(prompt)))

    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """Generate text as a stream of tokens."""
        tokenizer = self._ensure_tokenizer()
        prompt_to_encode = _format_instruction_prompt(prompt) if self._instruction_tuned else prompt
        with tracer.start_as_current_span("SyncGemmaModel.generate_stream") as span:
            span.set_attribute("prompt_length", len(prompt))
            tokenizer.enable_truncation(max_length=self.config.max_sequence_length)
            tokenizer.enable_padding()
            encoded = tokenizer.encode(prompt_to_encode)
            tokens = encoded.ids
            if not tokens or tokens[0] != _BOS_TOKEN_ID:
                tokens = [_BOS_TOKEN_ID, *list(tokens)]

        if _core is None or self._llm is None:
            msg = (
                "Mojo core is unavailable for text generation. "
                "Build/install the `mogemma._core` extension before calling generate()."
            )
            raise RuntimeError(msg)

        if isinstance(self._llm, dict):
            self._llm["pos"] = 0

        for t in tokens[:-1]:
            _core.step(self._llm, int(t), self.config.temperature, self.config.top_k, self.config.top_p)

        if tokens:
            current_token = int(tokens[-1])
        else:
            eos_token_id = _normalize_eos_token_id(tokenizer)
            current_token = int(eos_token_id)

        eos_token_id = _normalize_eos_token_id(tokenizer)
        for _ in range(self.config.max_tokens):
            logits = _core.step(self._llm, current_token, self.config.temperature, self.config.top_k, self.config.top_p)

            next_token = _sample_next_token(
                logits, temperature=self.config.temperature, top_k=self.config.top_k, top_p=self.config.top_p
            )
            if next_token == eos_token_id:
                return

            decoded = tokenizer.decode([next_token])
            if not isinstance(decoded, str):
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


class AsyncGemmaModel:
    """Asynchronous wrapper for SyncGemmaModel."""

    def __init__(self, config: GenerationConfig | str | None = None) -> None:
        """Initialize the async model.

        Args:
            config: Model ID string, ``GenerationConfig``, or ``None`` for defaults.
        """
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

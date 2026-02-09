import asyncio
from collections.abc import AsyncIterator

from .config import GenerationConfig
from .model import GemmaModel


class AsyncGemmaModel:
    """Asynchronous wrapper for GemmaModel."""

    def __init__(self, config: GenerationConfig) -> None:
        """Initialize the async model."""
        self._model = GemmaModel(config)

    async def generate(self, prompt: str) -> str:
        """Generate text asynchronously."""
        return await asyncio.to_thread(self._model.generate, prompt)

    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        """Generate text as an async stream of tokens."""
        # For streaming, we need to iterate over the generator in a thread
        # and yield back to the event loop.

        generator = self._model.generate_stream(prompt)

        def get_next():
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
    def tokenizer(self):
        """Access to the underlying tokenizer."""
        return self._model.tokenizer

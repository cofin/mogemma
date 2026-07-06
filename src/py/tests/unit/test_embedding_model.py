"""Tests for SyncEmbeddingModel and AsyncEmbeddingModel."""

from __future__ import annotations

import inspect

import pytest

from mogemma import AsyncEmbeddingModel, SyncEmbeddingModel
from mogemma.config import EmbeddingConfig


class TestSyncEmbeddingModelImport:
    """Verify SyncEmbeddingModel is importable and correctly named."""

    def test_import_from_package(self) -> None:
        from mogemma import SyncEmbeddingModel as M

        assert M is SyncEmbeddingModel

    def test_class_name(self) -> None:
        assert SyncEmbeddingModel.__name__ == "SyncEmbeddingModel"


class TestAsyncEmbeddingModelImport:
    """Verify AsyncEmbeddingModel is importable and correctly named."""

    def test_import_from_package(self) -> None:
        from mogemma import AsyncEmbeddingModel as M

        assert M is AsyncEmbeddingModel

    def test_class_name(self) -> None:
        assert AsyncEmbeddingModel.__name__ == "AsyncEmbeddingModel"


class TestSyncEmbeddingModelConfig:
    """Test SyncEmbeddingModel accepts the same config forms as the old EmbeddingModel."""

    def test_accepts_embedding_config(self) -> None:
        """Constructor accepts EmbeddingConfig without error (will fail at init due to missing model, not config)."""
        config = EmbeddingConfig(model_path="/nonexistent/path")
        with pytest.raises(Exception):  # noqa: B017
            SyncEmbeddingModel(config)

    def test_accepts_string_path(self) -> None:
        """Constructor accepts a string model path."""
        with pytest.raises(Exception):  # noqa: B017
            SyncEmbeddingModel("/nonexistent/path")

    def test_embed_requires_core(self) -> None:
        """embed() raises RuntimeError when Mojo core is unavailable."""
        config = EmbeddingConfig(model_path="/nonexistent/path")
        with pytest.raises(Exception):  # noqa: B017
            SyncEmbeddingModel(config)

    def test_embed_tokens_requires_core(self) -> None:
        """embed_tokens() raises RuntimeError when Mojo core is unavailable."""
        config = EmbeddingConfig(model_path="/nonexistent/path")
        with pytest.raises(Exception):  # noqa: B017
            SyncEmbeddingModel(config)


class TestAsyncEmbeddingModelConfig:
    """Test AsyncEmbeddingModel wraps SyncEmbeddingModel."""

    def test_accepts_string_path(self) -> None:
        """Constructor accepts a string model path (fails at init, not config)."""
        with pytest.raises(Exception):  # noqa: B017
            AsyncEmbeddingModel("/nonexistent/path")

    def test_accepts_embedding_config(self) -> None:
        """Constructor accepts EmbeddingConfig."""
        config = EmbeddingConfig(model_path="/nonexistent/path")
        with pytest.raises(Exception):  # noqa: B017
            AsyncEmbeddingModel(config)


class TestAsyncEmbeddingModelProtocol:
    """Verify AsyncEmbeddingModel has the expected async methods."""

    @pytest.mark.parametrize("method_name", ["embed", "embed_tokens"])
    def test_async_methods_are_coroutines(self, method_name: str) -> None:
        assert inspect.iscoroutinefunction(getattr(AsyncEmbeddingModel, method_name))

    @pytest.mark.parametrize("method_name", ["__aenter__", "__aexit__"])
    def test_async_context_manager_methods_exist(self, method_name: str) -> None:
        assert hasattr(AsyncEmbeddingModel, method_name)


class TestSyncEmbeddingModelProtocol:
    """Verify SyncEmbeddingModel has the expected sync methods."""

    @pytest.mark.parametrize("method_name", ["embed", "embed_tokens", "close"])
    def test_sync_methods_are_callable(self, method_name: str) -> None:
        assert callable(getattr(SyncEmbeddingModel, method_name, None))

    @pytest.mark.parametrize("method_name", ["__enter__", "__exit__"])
    def test_sync_context_manager_methods_exist(self, method_name: str) -> None:
        assert hasattr(SyncEmbeddingModel, method_name)

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

    def test_no_embedding_model_alias(self) -> None:
        """EmbeddingModel alias was removed — importing it should fail."""
        with pytest.raises((ImportError, AttributeError)):
            from mogemma import EmbeddingModel  # noqa: F401

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

    def test_has_embed(self) -> None:
        assert inspect.iscoroutinefunction(AsyncEmbeddingModel.embed)

    def test_has_embed_tokens(self) -> None:
        assert inspect.iscoroutinefunction(AsyncEmbeddingModel.embed_tokens)

    def test_has_aenter(self) -> None:
        assert hasattr(AsyncEmbeddingModel, "__aenter__")

    def test_has_aexit(self) -> None:
        assert hasattr(AsyncEmbeddingModel, "__aexit__")


class TestSyncEmbeddingModelProtocol:
    """Verify SyncEmbeddingModel has the expected sync methods."""

    def test_has_embed(self) -> None:
        assert callable(getattr(SyncEmbeddingModel, "embed", None))

    def test_has_embed_tokens(self) -> None:
        assert callable(getattr(SyncEmbeddingModel, "embed_tokens", None))

    def test_has_close(self) -> None:
        assert callable(getattr(SyncEmbeddingModel, "close", None))

    def test_context_manager(self) -> None:
        assert hasattr(SyncEmbeddingModel, "__enter__")
        assert hasattr(SyncEmbeddingModel, "__exit__")

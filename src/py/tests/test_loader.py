"""Tests for safetensors-only loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from mogemma.loader import SafetensorsLoader, auto_loader


class TestAutoLoader:
    def test_raises_on_empty_directory(self, tmp_path: Path) -> None:
        """auto_loader should raise FileNotFoundError for a directory with no safetensors."""
        with pytest.raises(FileNotFoundError, match="No supported model format"):
            auto_loader(tmp_path)

    def test_error_message_no_orbax_mention(self, tmp_path: Path) -> None:
        """Error message should not mention Orbax or OCDBT."""
        with pytest.raises(FileNotFoundError) as exc_info:
            auto_loader(tmp_path)
        assert "Orbax" not in str(exc_info.value)
        assert "OCDBT" not in str(exc_info.value)


class TestSafetensorsLoaderCanLoad:
    def test_false_for_empty_dir(self, tmp_path: Path) -> None:
        assert SafetensorsLoader.can_load(tmp_path) is False

    def test_true_for_single_safetensors_file(self, tmp_path: Path) -> None:
        (tmp_path / "model.safetensors").write_bytes(b"")
        assert SafetensorsLoader.can_load(tmp_path) is True

    def test_true_for_index_file(self, tmp_path: Path) -> None:
        (tmp_path / "model.safetensors.index.json").write_text("{}")
        assert SafetensorsLoader.can_load(tmp_path) is True

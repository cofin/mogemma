"""Tests for video frame extraction in hydration.py."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from mogemma.hydration import ImageHydrator, ImageInput, VIDEO_EXTENSIONS


class TestVideoDetection:
    """Video file detection by extension."""

    def test_video_extensions(self) -> None:
        for ext in [".mp4", ".webm", ".avi", ".mov", ".gif"]:
            assert ext in VIDEO_EXTENSIONS

    def test_non_video_not_detected(self) -> None:
        assert ".png" not in VIDEO_EXTENSIONS
        assert ".jpg" not in VIDEO_EXTENSIONS


class TestVideoFrameExtraction:
    """Frame extraction from video files."""

    @pytest.fixture
    def has_ffmpeg(self) -> bool:
        return shutil.which("ffmpeg") is not None

    def test_max_frames_cap(self) -> None:
        hydrator = ImageHydrator()
        assert hydrator._max_video_frames == 32

    def test_max_duration_cap(self) -> None:
        hydrator = ImageHydrator()
        assert hydrator._max_video_duration == 60

    def test_video_frame_budget(self) -> None:
        """Video frames should use lowest token budget."""
        hydrator = ImageHydrator()
        assert hydrator._video_frame_budget == 70

    def test_error_without_ffmpeg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should raise clear error when ffmpeg is unavailable."""
        monkeypatch.setattr(shutil, "which", lambda _: None)
        hydrator = ImageHydrator()
        with pytest.raises(RuntimeError, match="ffmpeg"):
            hydrator._extract_video_frames("/fake/video.mp4")

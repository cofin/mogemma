"""Tests for AudioHydrator (Ch6 Task 6.9)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from mogemma.hydration import AudioHydrator, AudioInput, AUDIO_EXTENSIONS


def _write_wav(path: Path, samples: np.ndarray, sr: int = 16000) -> None:
    pcm = (samples * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


class TestAudioExtensionDetection:
    def test_wav_detected(self) -> None:
        assert ".wav" in AUDIO_EXTENSIONS

    def test_non_audio_not_detected(self) -> None:
        assert ".png" not in AUDIO_EXTENSIONS
        assert ".mp4" not in AUDIO_EXTENSIONS


class TestAudioHydrator:
    def test_hydrate_wav_path(self, tmp_path: Path) -> None:
        samples = np.zeros(16000, dtype=np.float32)
        wav_path = tmp_path / "test.wav"
        _write_wav(wav_path, samples)
        hydrator = AudioHydrator()
        results = hydrator.hydrate([wav_path])
        assert len(results) == 1
        assert isinstance(results[0], AudioInput)
        assert results[0].features.shape[0] == 80
        assert results[0].num_tokens > 0

    def test_hydrate_wav_bytes(self, tmp_path: Path) -> None:
        samples = np.zeros(16000, dtype=np.float32)
        wav_path = tmp_path / "test.wav"
        _write_wav(wav_path, samples)
        hydrator = AudioHydrator()
        results = hydrator.hydrate([wav_path.read_bytes()])
        assert len(results) == 1
        assert isinstance(results[0], AudioInput)

    def test_hydrate_numpy_array(self) -> None:
        audio = np.zeros(16000, dtype=np.float32)
        hydrator = AudioHydrator()
        results = hydrator.hydrate([audio])
        assert len(results) == 1
        assert results[0].features.shape[0] == 80

    def test_num_tokens_capped(self, tmp_path: Path) -> None:
        # 30 seconds → many frames, but num_tokens should be capped at 750
        samples = np.zeros(16000 * 30, dtype=np.float32)
        wav_path = tmp_path / "long.wav"
        _write_wav(wav_path, samples)
        hydrator = AudioHydrator()
        results = hydrator.hydrate([wav_path])
        assert results[0].num_tokens <= 750

    def test_hydrate_multiple(self, tmp_path: Path) -> None:
        for i in range(3):
            _write_wav(tmp_path / f"a{i}.wav", np.zeros(8000, dtype=np.float32))
        hydrator = AudioHydrator()
        results = hydrator.hydrate([tmp_path / f"a{i}.wav" for i in range(3)])
        assert len(results) == 3

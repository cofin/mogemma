"""Tests for audio feature extraction (Ch6 Task 6.8)."""

from __future__ import annotations

import struct
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest

from mogemma.audio import load_audio, mel_spectrogram, extract_audio_features


def _write_wav(path: Path, samples: np.ndarray, sr: int = 16000, channels: int = 1) -> None:
    """Write a WAV file from float32 samples [-1, 1]."""
    pcm = (samples * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


class TestLoadAudio:
    def test_loads_wav_mono(self, tmp_path: Path) -> None:
        samples = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
        wav_path = tmp_path / "test.wav"
        _write_wav(wav_path, samples)
        audio = load_audio(wav_path)
        assert audio.dtype == np.float32
        assert audio.ndim == 1
        assert len(audio) == 16000

    def test_loads_wav_stereo_to_mono(self, tmp_path: Path) -> None:
        mono = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
        stereo = np.column_stack([mono, mono]).flatten()
        wav_path = tmp_path / "stereo.wav"
        _write_wav(wav_path, stereo, channels=2)
        audio = load_audio(wav_path)
        assert audio.ndim == 1

    def test_truncates_to_30s(self, tmp_path: Path) -> None:
        long_audio = np.zeros(16000 * 60, dtype=np.float32)  # 60s
        wav_path = tmp_path / "long.wav"
        _write_wav(wav_path, long_audio)
        audio = load_audio(wav_path, max_seconds=30)
        assert len(audio) == 16000 * 30

    def test_loads_from_bytes(self, tmp_path: Path) -> None:
        samples = np.zeros(8000, dtype=np.float32)
        wav_path = tmp_path / "test.wav"
        _write_wav(wav_path, samples)
        audio = load_audio(wav_path.read_bytes())
        assert audio.dtype == np.float32
        assert len(audio) > 0

    def test_output_range(self, tmp_path: Path) -> None:
        samples = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
        # Pad to reasonable length
        samples = np.tile(samples, 4000)
        wav_path = tmp_path / "range.wav"
        _write_wav(wav_path, samples)
        audio = load_audio(wav_path)
        assert np.all(audio >= -1.01)
        assert np.all(audio <= 1.01)


class TestMelSpectrogram:
    def test_output_shape(self) -> None:
        audio = np.zeros(16000, dtype=np.float32)  # 1 second
        mel = mel_spectrogram(audio)
        assert mel.shape[0] == 80  # n_mels
        assert mel.shape[1] > 0  # num_frames

    def test_expected_frame_count(self) -> None:
        # 1 second @ 16kHz, hop_length=160 → ~100 frames
        audio = np.zeros(16000, dtype=np.float32)
        mel = mel_spectrogram(audio, hop_length=160)
        # floor((16000 - 400) / 160) + 1 = 98
        assert mel.shape[1] >= 95
        assert mel.shape[1] <= 105

    def test_nonzero_for_tone(self) -> None:
        t = np.arange(16000) / 16000.0
        audio = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        mel = mel_spectrogram(audio)
        assert np.any(mel > -10)  # log-mel should have energy

    def test_custom_params(self) -> None:
        audio = np.zeros(16000, dtype=np.float32)
        mel = mel_spectrogram(audio, n_fft=512, hop_length=256, n_mels=40)
        assert mel.shape[0] == 40


class TestExtractAudioFeatures:
    def test_returns_correct_shape(self, tmp_path: Path) -> None:
        samples = np.zeros(16000, dtype=np.float32)
        wav_path = tmp_path / "test.wav"
        _write_wav(wav_path, samples)
        features = extract_audio_features(wav_path)
        assert features.shape[0] == 80
        assert features.shape[1] > 0

    def test_pads_short_audio(self, tmp_path: Path) -> None:
        samples = np.zeros(1600, dtype=np.float32)  # 0.1s
        wav_path = tmp_path / "short.wav"
        _write_wav(wav_path, samples)
        features = extract_audio_features(wav_path, max_frames=750)
        assert features.shape[1] == 750  # padded

    def test_truncates_long_audio(self, tmp_path: Path) -> None:
        samples = np.zeros(16000 * 40, dtype=np.float32)  # 40s
        wav_path = tmp_path / "long.wav"
        _write_wav(wav_path, samples)
        features = extract_audio_features(wav_path, max_frames=750)
        assert features.shape[1] == 750  # truncated

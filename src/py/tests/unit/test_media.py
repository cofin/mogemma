"""Unit tests for audio, image, and video input preprocessing."""

from __future__ import annotations

import shutil
import wave
from typing import TYPE_CHECKING

import numpy as np
import pytest

from mogemma.audio import extract_audio_features, load_audio, mel_spectrogram
from mogemma.hydration import (
    AUDIO_EXTENSIONS,
    PATCH_SIZE,
    VIDEO_EXTENSIONS,
    AudioHydrator,
    AudioInput,
    ImageHydrator,
    ImageInput,
    select_token_budget,
)
from mogemma.model import Gemma4Variant, SyncGemmaModel

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _write_wav(path: Path, samples: np.ndarray, sr: int = 16000, channels: int = 1) -> None:
    """Write a WAV file from float32 samples in [-1, 1]."""
    pcm = (samples * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def test_audio_extension_detection() -> None:
    assert ".wav" in AUDIO_EXTENSIONS
    assert ".png" not in AUDIO_EXTENSIONS
    assert ".mp4" not in AUDIO_EXTENSIONS


def test_video_extension_detection() -> None:
    for extension in [".mp4", ".webm", ".avi", ".mov", ".gif"]:
        assert extension in VIDEO_EXTENSIONS
    assert ".png" not in VIDEO_EXTENSIONS


@pytest.mark.parametrize(
    ("attribute", "expected"), [("_max_video_frames", 32), ("_max_video_duration", 60), ("_video_frame_budget", 70)]
)
def test_video_frame_limits(attribute: str, expected: int) -> None:
    assert getattr(ImageHydrator(), attribute) == expected


def test_video_frame_extraction_errors_without_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: None)

    with pytest.raises(RuntimeError, match="ffmpeg"):
        ImageHydrator()._extract_video_frames("/fake/video.mp4")


def test_load_audio_reads_wav_mono(tmp_path: Path) -> None:
    samples = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
    wav_path = tmp_path / "test.wav"
    _write_wav(wav_path, samples)

    audio = load_audio(wav_path)

    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert len(audio) == 16000


def test_load_audio_converts_stereo_wav_to_mono(tmp_path: Path) -> None:
    mono = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
    stereo = np.column_stack([mono, mono]).flatten()
    wav_path = tmp_path / "stereo.wav"
    _write_wav(wav_path, stereo, channels=2)

    assert load_audio(wav_path).ndim == 1


def test_load_audio_truncates_and_preserves_range(tmp_path: Path) -> None:
    samples = np.tile(np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32), 16000 * 15)
    wav_path = tmp_path / "long.wav"
    _write_wav(wav_path, samples)

    audio = load_audio(wav_path, max_seconds=30)

    assert len(audio) == 16000 * 30
    assert np.all(audio >= -1.01)
    assert np.all(audio <= 1.01)


def test_load_audio_reads_wav_bytes(tmp_path: Path) -> None:
    wav_path = tmp_path / "test.wav"
    _write_wav(wav_path, np.zeros(8000, dtype=np.float32))

    audio = load_audio(wav_path.read_bytes())

    assert audio.dtype == np.float32
    assert len(audio) > 0


@pytest.mark.parametrize(
    ("audio", "kwargs", "expected_mels"),
    [
        (np.zeros(16000, dtype=np.float32), {}, 80),
        (np.zeros(16000, dtype=np.float32), {"n_fft": 512, "hop_length": 256, "n_mels": 40}, 40),
    ],
)
def test_mel_spectrogram_shape(audio: np.ndarray, kwargs: dict[str, int], expected_mels: int) -> None:
    mel = mel_spectrogram(audio, **kwargs)
    assert mel.shape[0] == expected_mels
    assert mel.shape[1] > 0


def test_mel_spectrogram_frame_count_and_tone_energy() -> None:
    one_second = np.zeros(16000, dtype=np.float32)
    mel = mel_spectrogram(one_second, hop_length=160)
    assert 95 <= mel.shape[1] <= 105

    t = np.arange(16000) / 16000.0
    tone = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    assert np.any(mel_spectrogram(tone) > -10)


@pytest.mark.parametrize(
    ("samples", "max_frames", "expected_frames"),
    [
        (np.zeros(16000, dtype=np.float32), None, None),
        (np.zeros(1600, dtype=np.float32), 750, 750),
        (np.zeros(16000 * 40, dtype=np.float32), 750, 750),
    ],
)
def test_extract_audio_features_shape(
    samples: np.ndarray, max_frames: int | None, expected_frames: int | None, tmp_path: Path
) -> None:
    wav_path = tmp_path / "test.wav"
    _write_wav(wav_path, samples)

    features = (
        extract_audio_features(wav_path)
        if max_frames is None
        else extract_audio_features(wav_path, max_frames=max_frames)
    )

    assert features.shape[0] == 80
    if expected_frames is None:
        assert features.shape[1] > 0
    else:
        assert features.shape[1] == expected_frames


def test_audio_hydrator_accepts_wav_path_bytes_and_numpy(tmp_path: Path) -> None:
    wav_path = tmp_path / "test.wav"
    _write_wav(wav_path, np.zeros(16000, dtype=np.float32))
    hydrator = AudioHydrator()

    path_result = hydrator.hydrate([wav_path])[0]
    bytes_result = hydrator.hydrate([wav_path.read_bytes()])[0]
    array_result = hydrator.hydrate([np.zeros(16000, dtype=np.float32)])[0]

    assert isinstance(path_result, AudioInput)
    assert isinstance(bytes_result, AudioInput)
    assert isinstance(array_result, AudioInput)
    assert path_result.features.shape[0] == 80
    assert array_result.features.shape[0] == 80


def test_audio_hydrator_caps_tokens_and_handles_multiple_inputs(tmp_path: Path) -> None:
    long_wav = tmp_path / "long.wav"
    _write_wav(long_wav, np.zeros(16000 * 30, dtype=np.float32))
    for index in range(3):
        _write_wav(tmp_path / f"a{index}.wav", np.zeros(8000, dtype=np.float32))

    hydrator = AudioHydrator()

    assert hydrator.hydrate([long_wav])[0].num_tokens <= 750
    assert len(hydrator.hydrate([tmp_path / f"a{index}.wav" for index in range(3)])) == 3


@pytest.mark.parametrize(
    ("height", "width", "max_tokens", "predicate"),
    [
        (1000, 1000, 1120, lambda h, w: h == w),
        (100, 100, 70, lambda h, w: h == 280 and w == 280),
        (300, 900, 1120, lambda h, w: w >= h),
        (900, 300, 1120, lambda h, w: h >= w),
    ],
)
def test_select_token_budget(height: int, width: int, max_tokens: int, predicate: Callable[[int, int], bool]) -> None:
    target_h, target_w, num_tokens = select_token_budget(height, width, max_tokens=max_tokens)
    assert num_tokens == (target_h // PATCH_SIZE) * (target_w // PATCH_SIZE)
    assert predicate(target_h, target_w)


def test_select_token_budget_respects_caps() -> None:
    _, _, num_tokens = select_token_budget(2000, 2000, max_tokens=280)
    assert num_tokens <= (560 // PATCH_SIZE) * (560 // PATCH_SIZE)

    for max_tokens in [70, 140, 280, 560, 1120]:
        target_h, target_w, tokens = select_token_budget(500, 500, max_tokens=max_tokens)
        assert tokens > 0
        assert tokens == (target_h // PATCH_SIZE) * (target_w // PATCH_SIZE)


def test_image_input_fields() -> None:
    patches = np.zeros((70, 768), dtype=np.float32)
    image_input = ImageInput(patches=patches, grid_h=7, grid_w=10, num_tokens=70)

    assert image_input.grid_h == 7
    assert image_input.grid_w == 10
    assert image_input.num_tokens == 70
    assert image_input.patches.shape == (70, 768)


@pytest.mark.parametrize(("value", "expected"), [(255, 1.0), (0, -1.0)])
def test_preprocess_image_normalization(value: int, expected: float) -> None:
    size = PATCH_SIZE * 17
    image = np.full((size, size, 3), value, dtype=np.uint8)

    result = ImageHydrator().preprocess_image(image, max_tokens=1120)

    assert np.allclose(result.patches, expected, atol=1e-5)


def test_preprocess_image_patch_shape_grid_and_resize() -> None:
    hydrator = ImageHydrator()

    square = hydrator.preprocess_image(np.zeros((500, 500, 3), dtype=np.uint8), max_tokens=280)
    wide = hydrator.preprocess_image(np.zeros((800, 1600, 3), dtype=np.uint8), max_tokens=560)
    resized = hydrator.preprocess_image(np.zeros((100, 200, 3), dtype=np.uint8), max_tokens=140)

    assert square.patches.shape[1] == PATCH_SIZE * PATCH_SIZE * 3
    assert square.num_tokens == square.patches.shape[0]
    assert wide.num_tokens == wide.grid_h * wide.grid_w
    assert wide.grid_h > 0
    assert wide.grid_w > 0
    assert resized.patches.shape[0] > 0
    assert resized.patches.dtype == np.float32


def test_image_hydrator_returns_image_inputs() -> None:
    hydrator = ImageHydrator()
    img1 = np.zeros((280, 280, 3), dtype=np.uint8)
    img2 = np.full((560, 560, 3), 128, dtype=np.uint8)

    results = hydrator.hydrate([img1, img2])

    assert len(results) == 2
    assert isinstance(results[0], ImageInput)
    assert results[0].patches.shape[1] == PATCH_SIZE * PATCH_SIZE * 3


def test_generation_rejects_audio_before_tokenization() -> None:
    model = object.__new__(SyncGemmaModel)

    with pytest.raises(RuntimeError, match="Audio input is recognized but not implemented"):
        next(model.generate_stream("describe this", audio=[np.zeros(16000, dtype=np.float32)]))


def test_generation_rejects_12b_audio_before_tokenization() -> None:
    model = object.__new__(SyncGemmaModel)
    model._variant = Gemma4Variant.DENSE_12B_UNIFIED

    with pytest.raises(RuntimeError, match="Gemma 4 12B unified audio input is recognized but not implemented"):
        next(model.generate_stream("describe this", audio=[np.zeros(16000, dtype=np.float32)]))


def test_generation_rejects_12b_images_before_hydration(monkeypatch: pytest.MonkeyPatch) -> None:
    model = object.__new__(SyncGemmaModel)
    model._variant = Gemma4Variant.DENSE_12B_UNIFIED

    def _fail_hydrate(self: ImageHydrator, images: object) -> object:
        msg = "12B image gate should run before ImageHydrator"
        raise AssertionError(msg)

    monkeypatch.setattr(ImageHydrator, "hydrate", _fail_hydrate)

    with pytest.raises(RuntimeError, match="Gemma 4 12B unified image input is recognized but not implemented"):
        next(model.generate_stream("describe this", images=[np.zeros((280, 280, 3), dtype=np.uint8)]))

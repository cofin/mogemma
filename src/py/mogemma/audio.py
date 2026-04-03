"""Audio feature extraction for Gemma 4 E2B/E4B multimodal inputs.

Uses only numpy.fft for STFT and stdlib wave for WAV I/O — no scipy,
librosa, or torchaudio dependencies.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

# Audio constants
SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 30
MAX_AUDIO_SAMPLES = SAMPLE_RATE * MAX_AUDIO_SECONDS  # 480000


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(n_mels: int, n_fft: int, sr: int) -> npt.NDArray[np.float32]:
    """Build a triangular mel filterbank matrix [n_mels, n_fft//2 + 1]."""
    n_freqs = n_fft // 2 + 1
    mel_low = _hz_to_mel(0)
    mel_high = _hz_to_mel(sr / 2)

    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filterbank = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        for j in range(left, center):
            if center > left:
                filterbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                filterbank[i, j] = (right - j) / (right - center)

    return filterbank


def load_audio(
    source: str | Path | bytes,
    *,
    sr: int = SAMPLE_RATE,
    max_seconds: int = MAX_AUDIO_SECONDS,
) -> npt.NDArray[np.float32]:
    """Load audio from WAV file path or bytes, return mono float32 [-1, 1].

    Stereo is averaged to mono. Audio is truncated to max_seconds.
    """
    if isinstance(source, bytes):
        f = io.BytesIO(source)
    else:
        f = open(source, "rb")  # noqa: SIM115

    try:
        with wave.open(f, "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    finally:
        if not isinstance(source, bytes):
            f.close()

    # Convert to float32
    if sampwidth == 2:
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        pcm = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        pcm = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        msg = f"Unsupported WAV sample width: {sampwidth}"
        raise ValueError(msg)

    # Stereo → mono
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)

    # Resample if needed (simple decimation/interpolation)
    if framerate != sr:
        num_samples = int(len(pcm) * sr / framerate)
        indices = np.linspace(0, len(pcm) - 1, num_samples)
        pcm = np.interp(indices, np.arange(len(pcm)), pcm).astype(np.float32)

    # Truncate
    max_samples = sr * max_seconds
    if len(pcm) > max_samples:
        pcm = pcm[:max_samples]

    return pcm


def mel_spectrogram(
    audio: npt.NDArray[np.float32],
    *,
    sr: int = SAMPLE_RATE,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
) -> npt.NDArray[np.float32]:
    """Compute log-mel spectrogram using numpy.fft only.

    Returns:
        [n_mels, num_frames] float32 array of log-mel energies.
    """
    # Hann window
    window = np.hanning(n_fft).astype(np.float32)

    # STFT via numpy.fft.rfft
    num_frames = (len(audio) - n_fft) // hop_length + 1
    if num_frames <= 0:
        # Pad short audio
        padded = np.zeros(n_fft, dtype=np.float32)
        padded[: len(audio)] = audio
        frame = padded * window
        spectrum = np.abs(np.fft.rfft(frame)) ** 2
        filterbank = _mel_filterbank(n_mels, n_fft, sr)
        mel = filterbank @ spectrum.astype(np.float32)
        return np.log(mel[:, np.newaxis] + 1e-6).astype(np.float32)

    # Build frames
    frames = np.zeros((num_frames, n_fft), dtype=np.float32)
    for i in range(num_frames):
        start = i * hop_length
        frames[i] = audio[start : start + n_fft] * window

    # FFT
    spectra = np.abs(np.fft.rfft(frames, axis=1)) ** 2  # [num_frames, n_fft//2+1]

    # Mel filterbank
    filterbank = _mel_filterbank(n_mels, n_fft, sr)
    mel = filterbank @ spectra.T  # [n_mels, num_frames]

    # Log-mel
    return np.log(mel + 1e-6).astype(np.float32)


def extract_audio_features(
    source: str | Path | bytes,
    *,
    max_frames: int = 750,
    sr: int = SAMPLE_RATE,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
    max_seconds: int = MAX_AUDIO_SECONDS,
) -> npt.NDArray[np.float32]:
    """Load audio and extract mel spectrogram features, padded/truncated to max_frames.

    Returns:
        [n_mels, max_frames] float32 array.
    """
    audio = load_audio(source, sr=sr, max_seconds=max_seconds)
    mel = mel_spectrogram(audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    # Pad or truncate to max_frames
    if mel.shape[1] < max_frames:
        padded = np.zeros((n_mels, max_frames), dtype=np.float32)
        padded[:, : mel.shape[1]] = mel
        return padded
    return mel[:, :max_frames]

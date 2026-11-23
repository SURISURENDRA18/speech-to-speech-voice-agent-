"""Audio processing utilities: format conversions, resampling, chunking, and buffering."""

import base64
from collections.abc import Generator

import numpy as np
import torch


def pcm16_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert raw 16-bit PCM little-endian bytes to float32 numpy array in [-1.0, 1.0]."""
    if not pcm_bytes:
        return np.empty(0, dtype=np.float32)
    # Ensure byte count is even
    truncated_len = len(pcm_bytes) - (len(pcm_bytes) % 2)
    audio_int16 = np.frombuffer(pcm_bytes[:truncated_len], dtype=np.int16)
    return (audio_int16.astype(np.float32) / 32768.0).copy()


def float32_to_pcm16(audio_float32: np.ndarray | torch.Tensor) -> bytes:
    """Convert float32 audio [-1.0, 1.0] to raw 16-bit PCM little-endian bytes."""
    if isinstance(audio_float32, torch.Tensor):
        audio_float32 = audio_float32.detach().cpu().numpy()
    if audio_float32.size == 0:
        return b""
    clipped = np.clip(audio_float32, -1.0, 1.0)
    audio_int16 = (clipped * 32767.0).astype(np.int16)
    return audio_int16.tobytes()


def base64_to_pcm_bytes(b64_str: str) -> bytes:
    """Decode base64 string into raw PCM bytes."""
    return base64.b64decode(b64_str)


def pcm_bytes_to_base64(pcm_bytes: bytes) -> str:
    """Encode raw PCM bytes into base64 string."""
    return base64.b64encode(pcm_bytes).decode("ascii")


def resample_audio(
    audio: np.ndarray, orig_sr: int, target_sr: int
) -> np.ndarray:
    """Resample 1D float32 audio array between sample rates.

    Uses high-quality linear interpolation if lengths differ.
    """
    if orig_sr == target_sr or len(audio) == 0:
        return audio.astype(np.float32, copy=False)

    num_target_samples = round(len(audio) * float(target_sr) / orig_sr)
    if num_target_samples <= 0:
        return np.empty(0, dtype=np.float32)

    orig_indices = np.linspace(0, len(audio) - 1, num=len(audio))
    target_indices = np.linspace(0, len(audio) - 1, num=num_target_samples)
    resampled = np.interp(target_indices, orig_indices, audio).astype(np.float32)
    return resampled


def slice_into_frames(
    samples: np.ndarray, frame_size: int = 512
) -> Generator[np.ndarray, None, None]:
    """Yield fixed-size frames from a 1D audio sample array."""
    total = len(samples)
    for i in range(0, total - (total % frame_size), frame_size):
        yield samples[i : i + frame_size]


def compute_rms_db(samples: np.ndarray) -> float:
    """Compute Root Mean Square (RMS) energy in decibels (dBFS)."""
    if len(samples) == 0:
        return -100.0
    mean_square = np.mean(np.square(samples))
    if mean_square <= 1e-10:
        return -100.0
    return float(10.0 * np.log10(mean_square))


class BoundedAudioAccumulator:
    """Accumulates incoming streaming audio samples up to a maximum duration.

    Prevents unbounded memory growth in long sessions.
    """

    def __init__(self, sample_rate: int = 16000, max_seconds: float = 30.0):
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_seconds)
        self._buffer: list[np.ndarray] = []
        self._current_samples = 0

    def append(self, samples: np.ndarray) -> bool:
        """Append samples. Returns True if appended within limit, False if truncated."""
        if len(samples) == 0:
            return True

        if self._current_samples + len(samples) > self.max_samples:
            remaining = self.max_samples - self._current_samples
            if remaining > 0:
                self._buffer.append(samples[:remaining])
                self._current_samples += remaining
            return False

        self._buffer.append(samples)
        self._current_samples += len(samples)
        return True

    def get_audio(self) -> np.ndarray:
        """Return all accumulated samples as a single continuous 1D float32 array."""
        if not self._buffer:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(self._buffer, axis=0)

    def duration_seconds(self) -> float:
        """Current duration in seconds."""
        return float(self._current_samples) / self.sample_rate

    def clear(self) -> None:
        """Reset the accumulator."""
        self._buffer.clear()
        self._current_samples = 0

    def __len__(self) -> int:
        return self._current_samples

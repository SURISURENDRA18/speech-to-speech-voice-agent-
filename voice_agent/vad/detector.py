"""Silero VAD v6 streaming speech detector and turn boundary handler."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from voice_agent.config import settings

logger = logging.getLogger(__name__)


@dataclass
class VADEvent:
    """Event emitted by the VAD detector."""

    event_type: str  # "speech_start", "speech_end", "speech_chunk", or "silence"
    timestamp_samples: int
    duration_samples: int = 0
    speech_audio: np.ndarray | None = None


def find_silero_model_path(custom_path: str | Path | None = None) -> Path:
    """Find the path to the silero_vad.jit model file."""
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p

    # Candidate 1: extracted silero-vad-master
    candidate1 = (
        settings.workspace_dir
        / "silero-vad-master"
        / "src"
        / "silero_vad"
        / "data"
        / "silero_vad.jit"
    )
    if candidate1.exists():
        return candidate1

    # Candidate 2: ~/.cache/torch/hub/snakers4_silero-vad_master/src/silero_vad/data/silero_vad.jit
    candidate2 = (
        Path.home()
        / ".cache"
        / "torch"
        / "hub"
        / "snakers4_silero-vad_master"
        / "src"
        / "silero_vad"
        / "data"
        / "silero_vad.jit"
    )
    if candidate2.exists():
        return candidate2

    # Candidate 3: ~/.cache/torch/hub/snakers4_silero-vad_master/files/silero_vad.jit
    candidate3 = (
        Path.home()
        / ".cache"
        / "torch"
        / "hub"
        / "snakers4_silero-vad_master"
        / "files"
        / "silero_vad.jit"
    )
    if candidate3.exists():
        return candidate3

    raise FileNotFoundError(
        f"Silero VAD model (.jit) not found in expected locations: {candidate1} or {candidate2}"
    )


class SileroVADDetector:
    """Streaming VAD detector using Silero VAD v6 TorchScript model.

    Processes 512-sample (32ms at 16kHz) audio chunks and detects speech turns.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        neg_threshold: float | None = None,
        min_silence_duration_ms: int = 300,
        speech_pad_ms: int = 30,
        device: str = "cpu",
    ):
        if sample_rate not in (8000, 16000):
            raise ValueError(f"Silero VAD supports 8000 or 16000 Hz, got {sample_rate}")

        self.sample_rate = sample_rate
        self.threshold = threshold
        self.neg_threshold = (
            neg_threshold if neg_threshold is not None else max(threshold - 0.15, 0.01)
        )
        self.min_silence_samples = int(sample_rate * min_silence_duration_ms / 1000)
        self.speech_pad_samples = int(sample_rate * speech_pad_ms / 1000)
        self.device = torch.device(device)

        resolved_path = find_silero_model_path(model_path)
        logger.info("Loading Silero VAD v6 model from %s", resolved_path)
        self.model = torch.jit.load(str(resolved_path), map_location=self.device)
        self.model.eval()

        # Turn tracking state
        self._triggered = False
        self._temp_end = 0
        self._current_sample = 0
        self._speech_buffer: list[np.ndarray] = []
        self._pre_speech_ring_buffer: list[np.ndarray] = []
        self._max_pre_speech_chunks = max(1, self.speech_pad_samples // 512)

        # Reset model internal recurrent state
        self.reset_states()

    def reset_states(self) -> None:
        """Reset internal recurrent neural network states in the TorchScript model."""
        if hasattr(self.model, "reset_states"):
            self.model.reset_states()

    def reset(self) -> None:
        """Reset all tracking and buffer states for a new turn or after interruption."""
        self.reset_states()
        self._triggered = False
        self._temp_end = 0
        self._current_sample = 0
        self._speech_buffer.clear()
        self._pre_speech_ring_buffer.clear()

    def process_chunk(self, chunk: np.ndarray) -> list[VADEvent]:
        """Process a single 512-sample chunk of 16kHz float32 audio.

        Returns a list of VADEvent objects (e.g. speech_start, speech_end).
        """
        if len(chunk) != 512:
            raise ValueError(f"Expected chunk size 512 samples, got {len(chunk)}")

        events: list[VADEvent] = []
        tensor_chunk = torch.from_numpy(chunk).to(self.device).unsqueeze(0)

        with torch.no_grad():
            speech_prob = float(self.model(tensor_chunk, self.sample_rate).item())

        self._current_sample += len(chunk)

        # Handle speech detection logic
        if speech_prob >= self.threshold:
            if self._temp_end != 0:
                self._temp_end = 0

            if not self._triggered:
                self._triggered = True
                speech_start_sample = max(0, self._current_sample - len(chunk) - self.speech_pad_samples)
                events.append(
                    VADEvent(
                        event_type="speech_start",
                        timestamp_samples=speech_start_sample,
                    )
                )
                # Seed speech buffer with pre-speech padding chunks
                for pre_chunk in self._pre_speech_ring_buffer:
                    self._speech_buffer.append(pre_chunk)

            self._speech_buffer.append(chunk)

        elif self._triggered and speech_prob < self.neg_threshold:
            # Inside speech, but currently quiet
            self._speech_buffer.append(chunk)
            if self._temp_end == 0:
                self._temp_end = self._current_sample

            if self._current_sample - self._temp_end >= self.min_silence_samples:
                # Speech turn ended!
                self._triggered = False
                speech_end_sample = self._temp_end
                duration = speech_end_sample - (
                    self._current_sample - len(self._speech_buffer) * 512
                )
                complete_audio = (
                    np.concatenate(self._speech_buffer, axis=0)
                    if self._speech_buffer
                    else np.empty(0, dtype=np.float32)
                )
                events.append(
                    VADEvent(
                        event_type="speech_end",
                        timestamp_samples=speech_end_sample,
                        duration_samples=max(0, duration),
                        speech_audio=complete_audio,
                    )
                )
                self._temp_end = 0
                self._speech_buffer.clear()
                self._pre_speech_ring_buffer.clear()
        elif self._triggered:
            # Continuing speech buffer while between neg_threshold and threshold
            self._speech_buffer.append(chunk)
        else:
            # Not in speech; maintain circular pre-speech padding buffer
            self._pre_speech_ring_buffer.append(chunk)
            if len(self._pre_speech_ring_buffer) > self._max_pre_speech_chunks:
                self._pre_speech_ring_buffer.pop(0)

        return events

    @property
    def is_speech_active(self) -> bool:
        """True if user speech is currently active."""
        return self._triggered

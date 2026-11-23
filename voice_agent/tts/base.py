"""Base interface and data models for Text-to-Speech (TTS)."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import numpy as np


@dataclass
class TTSAudioChunk:
    """A streaming chunk of synthesized audio."""

    audio_pcm16: bytes
    sample_rate: int = 24000
    channels: int = 1
    is_last: bool = False
    duration_seconds: float = 0.0
    audio_float32: np.ndarray | None = None


class TTSBase(ABC):
    """Abstract base class for streaming TTS engines."""

    @abstractmethod
    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        language: str | None = None,
    ) -> AsyncGenerator[TTSAudioChunk, None]:
        """Stream synthesized audio chunks asynchronously."""

    async def release_model(self) -> None:
        """Release cached inference weights when memory-constrained adapters need it."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Output sample rate in Hz."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier."""

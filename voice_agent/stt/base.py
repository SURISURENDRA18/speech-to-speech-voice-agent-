"""Base interface and data models for Speech-to-Text (STT)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class TranscriptionResult:
    """STT transcription output."""

    text: str
    language: str = "hi"
    is_final: bool = True
    duration_seconds: float = 0.0


class STTBase(ABC):
    """Abstract base class for STT engines."""

    @abstractmethod
    async def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe an audio buffer into text."""

    async def release_model(self) -> None:
        """Release cached inference weights when memory-constrained adapters need it."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name or identifier of the STT model."""

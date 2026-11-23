"""Whisper STT on NVIDIA GPUs via faster-whisper (CTranslate2).

Measured on an NVIDIA L4 with large-v3-turbo fp16 and a 3.3 s Hindi clip:
~235 ms, against ~490 ms for the transformers pipeline. Most of the gain is
lower per-token decoder overhead.
"""

import asyncio
import importlib.util
import logging
from typing import Any

import numpy as np

from voice_agent.audio import resample_audio
from voice_agent.config import settings
from voice_agent.stt.base import STTBase, TranscriptionResult

logger = logging.getLogger(__name__)


def faster_whisper_available() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


class FasterWhisperSTTAdapter(STTBase):
    """CTranslate2 Whisper, kept resident on the GPU between turns."""

    def __init__(
        self,
        model_name: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float16",
        default_language: str | None = None,
    ):
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self.default_language = default_language or settings.whisper_language
        self._model: Any = None
        self._lock = asyncio.Lock()

    @property
    def model_name(self) -> str:
        return f"Whisper {self._model_name} (faster-whisper, {self._device} {self._compute_type})"

    def load_model(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        logger.info("Loading faster-whisper '%s' on %s...", self._model_name, self._device)
        self._model = WhisperModel(
            self._model_name, device=self._device, compute_type=self._compute_type
        )

    async def preload(self) -> None:
        """Load and warm up so the first turn has no load/kernel-init cost."""
        await self.transcribe(np.zeros(16000, dtype=np.float32), 16000, "hi")

    async def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> TranscriptionResult:
        target_lang = language or self.default_language or "hi"
        if len(audio) == 0:
            return TranscriptionResult(text="", language=target_lang, duration_seconds=0.0)
        if sample_rate != 16000:
            audio = resample_audio(audio, sample_rate, 16000)
        audio = np.ascontiguousarray(audio, dtype=np.float32)

        async with self._lock:

            def _run() -> str:
                self.load_model()
                segments, _info = self._model.transcribe(
                    audio,
                    language=target_lang,
                    beam_size=1,
                    vad_filter=False,  # Silero VAD has already segmented the turn
                    condition_on_previous_text=False,
                    without_timestamps=True,
                )
                return "".join(segment.text for segment in segments).strip()

            text = await asyncio.to_thread(_run)

        return TranscriptionResult(
            text=text,
            language=target_lang,
            is_final=True,
            duration_seconds=len(audio) / 16000.0,
        )

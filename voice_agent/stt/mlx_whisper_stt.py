"""Whisper large-v3-turbo STT running natively on Apple Silicon via MLX.

Uses the quantized MLX port (~0.9 GB for 8-bit) so it can stay resident next to
the LLM and Svara TTS on an 8 GB Mac instead of being reloaded every turn.
"""

import asyncio
import importlib.util
import logging
import platform
from typing import Any

import numpy as np

from voice_agent.audio import resample_audio
from voice_agent.config import settings
from voice_agent.stt.base import STTBase, TranscriptionResult

logger = logging.getLogger(__name__)


def mlx_audio_available() -> bool:
    """True on Apple Silicon macOS with the mlx-audio runtime installed."""
    return (
        platform.system() == "Darwin"
        and platform.machine() in {"arm64", "aarch64"}
        and importlib.util.find_spec("mlx_audio") is not None
    )


class MLXWhisperSTTAdapter(STTBase):
    """Whisper transcription through mlx-audio, kept loaded between turns."""

    def __init__(self, model_id: str | None = None, default_language: str | None = None):
        self._model_id = model_id or settings.mlx_whisper_model
        self.default_language = default_language or settings.whisper_language
        self._model: Any = None
        self._lock = asyncio.Lock()

    @property
    def model_name(self) -> str:
        return f"Whisper large-v3-turbo MLX ({self._model_id})"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(self) -> None:
        if self._model is not None:
            return
        import mlx.core as mx
        from mlx_audio.stt import load

        logger.info("Loading MLX Whisper '%s'...", self._model_id)
        model = load(self._model_id)
        mx.eval(model.parameters())
        self._model = model
        logger.info("MLX Whisper loaded.")

    async def preload(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self.load_model)

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
        duration = float(len(audio)) / 16000.0

        async with self._lock:

            def _run() -> tuple[str, str]:
                import mlx.core as mx

                self.load_model()
                result = self._model.generate(
                    mx.array(audio),
                    language=target_lang,
                    return_timestamps=False,
                    temperature=0.0,
                    condition_on_previous_text=False,
                )
                text = (getattr(result, "text", "") or "").strip()
                return text, getattr(result, "language", None) or target_lang

            text, detected = await asyncio.to_thread(_run)

        return TranscriptionResult(
            text=text, language=detected, is_final=True, duration_seconds=duration
        )

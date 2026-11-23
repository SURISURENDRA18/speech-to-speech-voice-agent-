"""Whisper STT adapter supporting Whisper large-v3-turbo (809M parameters)."""

import asyncio
import gc
import logging
from typing import Any

import numpy as np
import torch

from voice_agent.audio import resample_audio
from voice_agent.config import settings
from voice_agent.devices import (
    empty_device_cache,
    keep_models_loaded,
    resolve_torch_device,
)
from voice_agent.stt.base import STTBase, TranscriptionResult

logger = logging.getLogger(__name__)

# Whisper model parameter counts:
# large-v3-turbo: 809M parameters (encoder: 32 layers, decoder: 4 layers)
# standard large-v3: ~1.55B parameters (encoder: 32 layers, decoder: 32 layers)
WHISPER_PARAM_COUNTS = {
    "large-v3-turbo": "809M",
    "turbo": "809M",
    "large-v3": "1.55B",
    "large": "1.55B",
    "medium": "769M",
    "small": "244M",
    "base": "74M",
    "tiny": "39M",
}


class WhisperSTTAdapter(STTBase):
    """Adapter for Whisper multilingual speech recognition.

    Defaults to `large-v3-turbo` (809M parameters) for fast multilingual ASR.
    """

    def __init__(
        self,
        model_name: str = "large-v3-turbo",
        device: str = "auto",
        default_language: str | None = None,
    ):
        self._model_name = model_name
        self.default_language = default_language or settings.whisper_language
        self._param_count = WHISPER_PARAM_COUNTS.get(model_name, "809M")

        self.device = resolve_torch_device(device)
        self._keep_loaded = keep_models_loaded(self.device)

        self._model: Any = None
        self._pipeline: Any = None
        self._lock = asyncio.Lock()

    @property
    def model_name(self) -> str:
        return f"{self._model_name} ({self._param_count} parameters)"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None or self._pipeline is not None

    def load_model(self) -> None:
        """Load the model weights into memory (supports MPS and CPU)."""
        if self.is_loaded:
            return

        logger.info(
            "Loading Whisper model '%s' (%s params) on device '%s'...",
            self._model_name,
            self._param_count,
            self.device,
        )

        try:
            # First attempt: Hugging Face transformers pipeline
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

            hf_model_id = f"openai/whisper-{self._model_name}"
            torch_dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32

            processor = AutoProcessor.from_pretrained(hf_model_id)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                hf_model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            ).to(self.device)

            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=self.device,
            )
            logger.info("Whisper model loaded successfully via Hugging Face pipeline.")
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            logger.info(
                "Hugging Face pipeline load for %s failed (%s). Attempting upstream whisper...",
                self._model_name,
                exc,
            )
            try:
                import sys

                whisper_dir = settings.workspace_dir / "whisper-main"
                if whisper_dir.exists() and str(whisper_dir) not in sys.path:
                    sys.path.insert(0, str(whisper_dir))

                import whisper

                self._model = whisper.load_model(self._model_name, device=self.device)
                logger.info("Whisper model loaded successfully via upstream whisper package.")
            except Exception as inner_exc:
                raise RuntimeError(
                    f"Failed to load Whisper '{self._model_name}'. "
                    f"Ensure weights are cached or network access is available: {inner_exc}"
                ) from inner_exc

    async def release_model(self) -> None:
        """Drop Whisper weights after each transcription (8 GB Macs only)."""
        if self._keep_loaded:
            return
        async with self._lock:
            self._pipeline = None
            self._model = None
            await asyncio.to_thread(gc.collect)
            empty_device_cache(self.device)

    async def preload(self) -> None:
        """Load and warm up Whisper so the first turn has no load/compile cost."""
        if not self._keep_loaded:
            return
        await self.transcribe(np.zeros(16000, dtype=np.float32), 16000, "hi")

    async def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe 1D float32 audio buffer into text asynchronously."""
        if len(audio) == 0:
            return TranscriptionResult(text="", language=language or "hi", duration_seconds=0.0)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            audio = resample_audio(audio, sample_rate, 16000)

        target_lang = language or self.default_language

        async with self._lock:
            # Lazy load model if not yet in memory
            if not self.is_loaded:
                # Run model loading in threadpool to avoid blocking event loop
                await asyncio.to_thread(self.load_model)

            duration = float(len(audio)) / 16000.0

            def _run_inference() -> tuple[str, str]:
                if self._pipeline is not None:
                    generate_kwargs = {}
                    if target_lang:
                        generate_kwargs["language"] = target_lang
                    res = self._pipeline(audio, generate_kwargs=generate_kwargs)
                    text = res.get("text", "").strip() if isinstance(res, dict) else str(res).strip()
                    detected_lang = target_lang or "auto"
                    return text, detected_lang
                elif self._model is not None:
                    decode_options: dict[str, Any] = {"fp16": self.device in ("cuda", "mps")}
                    if target_lang:
                        decode_options["language"] = target_lang
                    res = self._model.transcribe(audio, **decode_options)
                    return res.get("text", "").strip(), res.get("language", target_lang or "auto")
                raise RuntimeError("Whisper model is not loaded.")

            text, detected_lang = await asyncio.to_thread(_run_inference)
            return TranscriptionResult(
                text=text,
                language=detected_lang,
                is_final=True,
                duration_seconds=duration,
            )

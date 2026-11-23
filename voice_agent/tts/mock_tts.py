"""Mock TTS adapter for unit and integration testing."""

import asyncio
from collections.abc import AsyncGenerator

import numpy as np

from voice_agent.audio import float32_to_pcm16
from voice_agent.tts.base import TTSAudioChunk, TTSBase


class MockTTSAdapter(TTSBase):
    """Mock TTS adapter producing test audio chunks."""

    def __init__(self, sample_rate: int = 24000, chunk_duration_ms: int = 50):
        self._sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.call_count = 0

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def model_name(self) -> str:
        return "mock-tts (test adapter)"

    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        language: str | None = None,
    ) -> AsyncGenerator[TTSAudioChunk, None]:
        self.call_count += 1
        num_samples = int(self._sample_rate * (self.chunk_duration_ms / 1000.0))
        t = np.linspace(0, self.chunk_duration_ms / 1000.0, num_samples, endpoint=False, dtype=np.float32)
        sine = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        pcm_bytes = float32_to_pcm16(sine)

        # Yield two chunks
        yield TTSAudioChunk(
            audio_pcm16=pcm_bytes,
            sample_rate=self._sample_rate,
            channels=1,
            is_last=False,
            duration_seconds=self.chunk_duration_ms / 1000.0,
        )
        await asyncio.sleep(0.01)

        yield TTSAudioChunk(
            audio_pcm16=pcm_bytes,
            sample_rate=self._sample_rate,
            channels=1,
            is_last=True,
            duration_seconds=self.chunk_duration_ms / 1000.0,
        )

"""Mock STT adapter for testing and zero-download environments."""

import asyncio

import numpy as np

from voice_agent.stt.base import STTBase, TranscriptionResult


class MockSTTAdapter(STTBase):
    """Mock STT adapter providing deterministic transcription for testing."""

    def __init__(
        self,
        mock_response: str = "नमस्ते, आज मौसम कैसा है?",
        language: str = "hi",
        latency_seconds: float = 0.05,
    ):
        self.mock_response = mock_response
        self.language = language
        self.latency_seconds = latency_seconds
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return "mock-stt (test adapter)"

    async def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> TranscriptionResult:
        self.call_count += 1
        if self.latency_seconds > 0:
            await asyncio.sleep(self.latency_seconds)

        duration = float(len(audio)) / sample_rate if sample_rate > 0 else 0.0
        return TranscriptionResult(
            text=self.mock_response,
            language=language or self.language,
            is_final=True,
            duration_seconds=duration,
        )

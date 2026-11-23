"""Mock LLM adapter for deterministic fast tests and zero-memory test environments."""

import asyncio
from collections.abc import AsyncGenerator

from voice_agent.llm.base import LLMBase


class MockLLMAdapter(LLMBase):
    """Mock LLM adapter that streams tokens asynchronously with simulated latency."""

    def __init__(
        self,
        canned_response: str = "नमस्ते! मैं आपकी क्या सहायता कर सकता हूँ?",
        token_delay_seconds: float = 0.02,
    ):
        self.canned_response = canned_response
        self.token_delay_seconds = token_delay_seconds
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return "mock-llm (test adapter)"

    async def generate_stream(
        self,
        prompt: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        self.call_count += 1
        # Tokenize by space preserving spaces
        words = self.canned_response.split(" ")
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            if self.token_delay_seconds > 0:
                await asyncio.sleep(self.token_delay_seconds)
            yield token

"""Base interface for Language Model (LLM) engines."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from voice_agent.config import settings


def build_chat_messages(
    prompt: str,
    history: list[dict[str, str]] | None,
    system_prompt: str | None,
) -> list[dict[str, str]]:
    """Assemble ``[system, *history, user]`` chat messages for any adapter.

    History length is owned by the caller (``ConversationMemory``); adapters
    must not truncate it again.
    """
    messages: list[dict[str, str]] = []
    sys_p = system_prompt or settings.system_prompt
    if sys_p:
        messages.append({"role": "system", "content": sys_p})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    return messages


class LLMBase(ABC):
    """Abstract base class for streaming LLM engines."""

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream token or text deltas asynchronously from the LLM."""

    async def release_model(self) -> None:
        """Release cached inference weights when memory-constrained adapters need it."""

    @property
    def swaps_weights(self) -> bool:
        """True if release_model() really unloads weights (8 GB Mac mode).

        The pipeline then avoids running TTS while the LLM is still loaded.
        """
        return False

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier."""

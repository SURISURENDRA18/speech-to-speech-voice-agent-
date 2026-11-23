"""Base interface for Language Model (LLM) engines."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator


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

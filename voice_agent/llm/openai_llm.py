"""Streaming LLM adapter for OpenAI-compatible servers (e.g. vLLM serving Qwen3).

On a GPU, vLLM gives ~20 ms to first token and 150+ tokens/s for Qwen3-1.7B,
far faster than HF ``generate`` in-process, so the first sentence is ready in
roughly 100-200 ms.
"""

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from voice_agent.llm.base import LLMBase, build_chat_messages

logger = logging.getLogger(__name__)


class OpenAICompatibleLLMAdapter(LLMBase):
    """Streams chat completions from ``{base_url}/v1/chat/completions``."""

    def __init__(
        self,
        base_url: str,
        model: str = "Qwen/Qwen3-1.7B",
        api_key: str = "EMPTY",
        max_new_tokens: int = 48,
        temperature: float = 0.4,
        timeout_seconds: float = 30.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._base_url = self._base_url.removesuffix("/v1")
        self._model = model
        self._api_key = api_key
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._timeout = timeout_seconds
        self._client: httpx.AsyncClient | None = None

    @property
    def model_name(self) -> str:
        return f"{self._model} (remote {self._base_url})"

    def _http(self) -> httpx.AsyncClient:
        # One pooled keep-alive client avoids a TCP/TLS handshake per turn.
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout, connect=5.0),
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        return self._client

    async def preload(self) -> None:
        """Open the connection and warm the server's kernels/prefix cache."""
        async for _ in self.generate_stream("नमस्ते"):
            pass

    async def generate_stream(
        self,
        prompt: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": build_chat_messages(prompt, history, system_prompt),
            "max_tokens": self._max_new_tokens,
            "temperature": self._temperature,
            "top_p": 0.9,
            "stream": True,
            # Qwen3: answer directly, no <think> block (vLLM chat template arg).
            "chat_template_kwargs": {"enable_thinking": False},
        }
        url = f"{self._base_url}/v1/chat/completions"
        async with self._http().stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "replace")[:500]
                raise RuntimeError(f"LLM server error {resp.status_code}: {body}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for choice in event.get("choices", []):
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        yield delta

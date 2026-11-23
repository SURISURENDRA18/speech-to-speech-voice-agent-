"""Qwen3 1.7B LLM adapter using Hugging Face Transformers and Apple Silicon MPS/CPU."""

import asyncio
import gc
import logging
import threading
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, TextIteratorStreamer

from voice_agent.config import settings
from voice_agent.devices import (
    empty_device_cache,
    keep_models_loaded,
    resolve_torch_device,
)
from voice_agent.llm.base import LLMBase

logger = logging.getLogger(__name__)


def find_local_qwen3_path() -> Path | None:
    """Find local cached snapshot of Qwen3 1.7B if present."""
    hub_path = Path.home() / ".cache" / "huggingface" / "hub" / "models--unsloth--Qwen3-1.7B"
    snapshots = hub_path / "snapshots"
    if snapshots.exists():
        for snap in snapshots.iterdir():
            if snap.is_dir() and (snap / "model.safetensors").exists():
                return snap
    return None


class Qwen3LLMAdapter(LLMBase):
    """Adapter for Qwen3 1.7B streaming conversational generation."""

    def __init__(
        self,
        model_name_or_path: str = "unsloth/Qwen3-1.7B",
        device: str = "auto",
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ):
        self._model_identifier = model_name_or_path
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature

        self.device = resolve_torch_device(device)
        self._keep_loaded = keep_models_loaded(self.device)

        # Resolve torch dtype
        if torch_dtype == "bfloat16":
            self._dtype = torch.bfloat16
        elif torch_dtype == "float16":
            self._dtype = torch.float16
        else:
            self._dtype = torch.float32

        self._tokenizer: Any = None
        self._model: Any = None
        self._lock = asyncio.Lock()

    @property
    def model_name(self) -> str:
        return f"Qwen3-1.7B ({self._model_identifier})"

    @property
    def swaps_weights(self) -> bool:
        return not self._keep_loaded

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def load_model(self) -> None:
        """Load Qwen3 tokenizer and weights into memory."""
        if self.is_loaded:
            return

        # Check local snapshot first
        local_path = find_local_qwen3_path()
        model_target = str(local_path) if local_path else self._model_identifier

        logger.info(
            "Loading Qwen3 1.7B from '%s' on device '%s' (dtype: %s)...",
            model_target,
            self.device,
            self._dtype,
        )

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_target)

            # Try loading via Qwen3ForCausalLM or AutoModelForCausalLM
            try:
                from transformers import Qwen3ForCausalLM

                self._model = Qwen3ForCausalLM.from_pretrained(
                    model_target,
                    torch_dtype=self._dtype,
                    low_cpu_mem_usage=True,
                ).to(self.device)
            except (ImportError, AttributeError, ValueError, RuntimeError):
                from transformers import AutoModelForCausalLM

                self._model = AutoModelForCausalLM.from_pretrained(
                    model_target,
                    torch_dtype=self._dtype,
                    low_cpu_mem_usage=True,
                ).to(self.device)

            self._model.eval()
            logger.info("Qwen3 1.7B loaded successfully.")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Qwen3 1.7B from '{model_target}'. "
                f"Ensure weights are cached or network access is available: {exc}"
            ) from exc

    async def release_model(self) -> None:
        """Drop Qwen weights before local Svara synthesis (8 GB Macs only)."""
        if self._keep_loaded:
            return
        async with self._lock:
            self._model = None
            self._tokenizer = None
            await asyncio.to_thread(gc.collect)
            empty_device_cache(self.device)

    async def preload(self) -> None:
        """Load and warm up Qwen so the first reply has no load cost."""
        if not self._keep_loaded:
            return
        async for _ in self.generate_stream("नमस्ते"):
            pass

    async def generate_stream(
        self,
        prompt: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream generated text tokens asynchronously."""
        async with self._lock:
            if not self.is_loaded:
                await asyncio.to_thread(self.load_model)

        # Build conversation messages
        messages: list[dict[str, str]] = []
        sys_p = system_prompt or settings.system_prompt
        if sys_p:
            messages.append({"role": "system", "content": sys_p})

        if history:
            # Keep only last 4 turns to bound context memory
            messages.extend(history[-4:])

        messages.append({"role": "user", "content": prompt})

        # Apply chat template
        formatted_prompt = self._format_chat_prompt(messages)

        inputs = self._tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": self._max_new_tokens,
            "temperature": self._temperature,
            "do_sample": (self._temperature > 0.0),
        }

        # Run model.generate in a background thread
        thread = threading.Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        loop = asyncio.get_running_loop()

        def _get_next_token() -> str | None:
            try:
                return next(streamer)
            except StopIteration:
                return None

        while True:
            token = await loop.run_in_executor(None, _get_next_token)
            if token is None:
                break
            if token:
                yield token

        thread.join()

    def _format_chat_prompt(self, messages: list[dict[str, str]]) -> str:
        """Format a concise voice-agent prompt without exposing reasoning tokens."""
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

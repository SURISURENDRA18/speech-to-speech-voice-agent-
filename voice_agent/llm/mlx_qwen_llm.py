"""Qwen3 1.7B (4-bit) running natively on Apple Silicon via MLX.

Uses the mlx-lm machinery vendored inside mlx-audio, so no extra package is
needed. The 4-bit weights take ~1 GB and stay resident between turns, which
avoids the multi-minute swap storms the 3.4 GB bf16 PyTorch model causes on
8 GB Macs.
"""

import asyncio
import logging
import threading
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from voice_agent.config import settings
from voice_agent.llm.base import LLMBase, build_chat_messages

logger = logging.getLogger(__name__)


class MLXQwen3LLMAdapter(LLMBase):
    """Streaming Qwen3 generation through MLX, kept loaded between turns."""

    def __init__(
        self,
        model_id: str | None = None,
        max_new_tokens: int = 48,
        temperature: float = 0.4,
        repetition_penalty: float = 1.15,
    ):
        self._model_id = model_id or settings.mlx_llm_model
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._repetition_penalty = repetition_penalty
        self._model: Any = None
        self._tokenizer: Any = None
        self._lock = asyncio.Lock()

    @property
    def model_name(self) -> str:
        return f"Qwen3-1.7B MLX ({self._model_id})"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def load_model(self) -> None:
        if self.is_loaded:
            return
        import mlx.core as mx
        from mlx_audio.lm.models import qwen3
        from mlx_audio.utils import (
            apply_quantization,
            get_model_path,
            load_config,
            load_weights,
        )
        from transformers import AutoTokenizer

        logger.info("Loading MLX Qwen3 '%s'...", self._model_id)
        model_path = get_model_path(self._model_id)
        config = load_config(model_path)
        model = qwen3.Model(qwen3.ModelArgs.from_dict(config))
        weights = load_weights(Path(model_path))
        apply_quantization(model, config, weights)
        model.load_weights(list(model.sanitize(weights).items()))
        mx.eval(model.parameters())
        model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = model
        logger.info("MLX Qwen3 loaded.")

    async def preload(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self.load_model)

    def _build_prompt(
        self,
        prompt: str,
        history: list[dict[str, str]] | None,
        system_prompt: str | None,
    ) -> str:
        return self._tokenizer.apply_chat_template(
            build_chat_messages(prompt, history, system_prompt),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    async def generate_stream(
        self,
        prompt: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        async with self._lock:
            if not self.is_loaded:
                await asyncio.to_thread(self.load_model)

            formatted = self._build_prompt(prompt, history, system_prompt)
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()
            stop_event = threading.Event()

            def publish(item: str | Exception | None) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, item)

            def worker() -> None:
                try:
                    from mlx_audio.lm.generate import stream_generate
                    from mlx_audio.lm.sample_utils import (
                        make_logits_processors,
                        make_sampler,
                    )

                    for response in stream_generate(
                        self._model,
                        self._tokenizer,
                        formatted,
                        max_tokens=self._max_new_tokens,
                        sampler=make_sampler(temp=self._temperature, top_p=0.9),
                        logits_processors=make_logits_processors(
                            repetition_penalty=self._repetition_penalty,
                            repetition_context_size=32,
                        ),
                    ):
                        if stop_event.is_set():
                            break
                        if response.text:
                            publish(response.text)
                except Exception as exc:  # noqa: BLE001
                    publish(exc)
                finally:
                    publish(None)

            thread = threading.Thread(target=worker, name="qwen-mlx-generate", daemon=True)
            thread.start()
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield item
            finally:
                stop_event.set()
                if thread.is_alive():
                    await asyncio.to_thread(thread.join)

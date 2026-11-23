"""Svara TTS adapter with remote, Apple Silicon MLX, and macOS speech backends."""

import asyncio
import gc
import importlib.util
import logging
import platform
import shutil
import subprocess
import tempfile
import threading
import wave
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from voice_agent.audio import float32_to_pcm16, pcm16_to_float32, resample_audio
from voice_agent.config import settings
from voice_agent.tts.base import TTSAudioChunk, TTSBase

logger = logging.getLogger(__name__)


def load_svara_voices() -> list[dict[str, Any]]:
    """Load Svara voice metadata from extracted assets."""
    yaml_path = (
        settings.workspace_dir
        / "svara-tts-inference-main"
        / "assets"
        / "voices"
        / "svara-tts-v1.yaml"
    )
    if not yaml_path.exists():
        return []

    try:
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, list) else []
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Failed to load Svara voice profiles from %s: %s", yaml_path, exc)
        return []


class SvaraTTSAdapter(TTSBase):
    """Indic speech synthesis via remote Svara, local MLX Svara, or macOS speech fallback."""

    def __init__(
        self,
        model_name: str = "svara-tts-v1",
        local_model_name: str = "mlx-community/svara-tts-v1-4bit",
        default_voice: str = "hi_female",
        output_sample_rate: int = 24000,
        api_url: str | None = None,
        device: str = "auto",
        keep_loaded: bool = False,
    ):
        # keep_loaded=True keeps Svara resident between turns (safe when the
        # STT/LLM are the small MLX models); False reloads it every turn.
        self._keep_loaded = keep_loaded
        self._model_name = model_name
        self._local_model_name = local_model_name
        self.default_voice = default_voice
        self._sample_rate = output_sample_rate
        self.api_url = api_url or settings.svara_api_url
        self.device = device
        self._voices = load_svara_voices()
        self._mlx_model: Any = None
        self._mlx_lock = asyncio.Lock()
        self._mlx_error: str | None = None
        self._client: Any = None

    @property
    def _mlx_available(self) -> bool:
        """Whether the Apple Silicon MLX TTS runtime is installed."""
        return (
            platform.system() == "Darwin"
            and platform.machine() in {"arm64", "aarch64"}
            and importlib.util.find_spec("mlx_audio") is not None
        )

    @property
    def uses_local_mlx(self) -> bool:
        """Local MLX Svara shares the Mac GPU with the MLX LLM; don't overlap them."""
        return not self.api_url and self._mlx_available

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def model_name(self) -> str:
        if self.api_url:
            mode = "remote"
        elif self._mlx_error:
            mode = "macOS spoken fallback after MLX synthesis failure"
        elif self._mlx_available:
            mode = f"local MLX Svara ({self._local_model_name})"
        else:
            mode = "macOS spoken fallback (not Svara)"
        return f"{self._model_name} ({mode})"

    def get_voices(self) -> list[dict[str, Any]]:
        """Return list of supported voice profiles."""
        return self._voices

    async def release_model(self) -> None:
        """Drop MLX weights before loading Whisper for the next conversation turn."""
        if self._keep_loaded:
            return
        async with self._mlx_lock:
            model = self._mlx_model
            self._mlx_model = None
            if model is not None:
                del model
                await asyncio.to_thread(gc.collect)
                if self._mlx_available:
                    try:
                        await asyncio.to_thread(self._clear_mlx_cache)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("Unable to clear the MLX cache: %s", exc)

    async def preload(self) -> None:
        """Warm up Svara (remote server or local MLX) ahead of the first reply."""
        if self.api_url:
            # First request on a fresh vLLM/SNAC server compiles kernels.
            async for _ in self._stream_remote("नमस्ते।", self.default_voice, "hi"):
                pass
            return
        if not self._mlx_available or self._mlx_model is not None:
            return
        # Warm up through the normal synthesis path: loading alone can leave
        # lazy arrays bound to a thread-local GPU stream that later synthesis
        # threads can't see, and a short utterance also compiles the kernels.
        async for _ in self._stream_mlx("नमस्ते।", self.default_voice, "hi"):
            pass

    @staticmethod
    def _clear_mlx_cache() -> None:
        """Clear Metal's MLX cache on a thread with its own GPU stream."""
        import mlx.core as mx

        with mx.stream(mx.new_stream(mx.gpu)):
            mx.clear_cache()

    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        language: str | None = None,
    ) -> AsyncGenerator[TTSAudioChunk, None]:
        """Stream synthesized audio chunks for the given text."""
        if not text.strip():
            return

        target_voice = voice or self.default_voice

        # Mode 1: Remote Svara inference server (via Docker/GPU)
        if self.api_url:
            async for chunk in self._stream_remote(text, target_voice, language):
                yield chunk
            return

        # Mode 2: Stream local Svara chunks as they are decoded.
        if self._mlx_available:
            emitted_audio = False
            try:
                async for chunk in self._stream_mlx(text, target_voice, language):
                    emitted_audio = True
                    yield chunk
                self._mlx_error = None
                return
            except Exception as exc:
                self._mlx_error = str(exc)
                if emitted_audio:
                    raise
                logger.warning("Local Svara MLX synthesis failed; using spoken fallback: %s", exc)

        # Mode 3: Spoken macOS fallback if MLX is unavailable or fails before audio.
        async for chunk in self._stream_local(text, target_voice, language):
            yield chunk

    def _load_mlx_model(self) -> Any:
        """Load the quantized Svara model through MLX Audio on Apple Silicon."""
        from mlx_audio.tts.utils import load_model

        return load_model(self._local_model_name)

    async def _stream_mlx(
        self, text: str, voice: str, language: str | None
    ) -> AsyncGenerator[TTSAudioChunk, None]:
        """Bridge MLX's synchronous streamed decoder into async WebSocket audio."""
        async with self._mlx_lock:
            import mlx.core as mx

            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[TTSAudioChunk | Exception | None] = asyncio.Queue(
                maxsize=64
            )
            stop_event = threading.Event()
            source_rate = self._sample_rate

            def publish(item: TTSAudioChunk | Exception | None) -> None:
                if stop_event.is_set():
                    return
                future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
                try:
                    future.result()
                except RuntimeError:
                    stop_event.set()

            def generate() -> None:
                nonlocal source_rate
                try:
                    voice_name = self._mlx_voice_name(voice, language)
                    # MLX streams are thread-local; create the GPU stream here,
                    # and load/evaluate the model in that same worker.
                    with mx.stream(mx.new_stream(mx.gpu)):
                        if self._mlx_model is None:
                            self._mlx_model = self._load_mlx_model()
                        model = self._mlx_model
                        source_rate = int(getattr(model, "sample_rate", self._sample_rate))
                        for result in model.generate(
                            text=text,
                            voice=voice_name,
                            temperature=0.75,
                            top_p=0.9,
                            top_k=40,
                            repetition_penalty=1.1,
                            max_tokens=1200,
                            stream=True,
                            streaming_interval=0.5,
                        ):
                            if stop_event.is_set():
                                break
                            if result.audio is None:
                                continue
                            mx.eval(result.audio)
                            audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
                            if audio.size == 0:
                                continue
                            if source_rate != self._sample_rate:
                                audio = resample_audio(audio, source_rate, self._sample_rate)
                            pcm = float32_to_pcm16(audio)
                            if pcm:
                                publish(
                                    TTSAudioChunk(
                                        audio_pcm16=pcm,
                                        sample_rate=self._sample_rate,
                                        channels=1,
                                        duration_seconds=len(pcm) / (self._sample_rate * 2),
                                    )
                                )
                except Exception as exc:  # noqa: BLE001
                    publish(exc)
                finally:
                    publish(None)

            worker = threading.Thread(
                target=generate,
                name="svara-mlx-stream",
                daemon=True,
            )
            worker.start()
            emitted = False
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise item
                    # Forward each chunk immediately (no one-chunk holdback) so
                    # playback starts as soon as the first audio is decoded.
                    emitted = True
                    yield item
                if not emitted:
                    raise RuntimeError("Local Svara model returned no audio chunks")
                # Empty terminal marker keeps the is_last contract.
                yield TTSAudioChunk(
                    audio_pcm16=b"",
                    sample_rate=self._sample_rate,
                    channels=1,
                    is_last=True,
                    duration_seconds=0.0,
                )
            finally:
                stop_event.set()
                if worker.is_alive():
                    await asyncio.to_thread(worker.join)

    @staticmethod
    def _mlx_voice_name(voice: str, language: str | None) -> str:
        """Convert project voice IDs to Svara's language and gender voice labels."""
        lang = (language or voice.split("_", 1)[0]).lower().split("-", 1)[0]
        gender = voice.rsplit("_", 1)[-1].capitalize()
        language_names = {
            "as": "Assamese", "bn": "Bengali", "bh": "Bhojpuri",
            "brx": "Bodo", "doi": "Dogri", "en": "English (Indian)",
            "gu": "Gujarati", "hi": "Hindi", "hne": "Chhattisgarhi",
            "kn": "Kannada", "mag": "Magahi", "mai": "Maithili",
            "ml": "Malayalam", "mr": "Marathi", "ne": "Nepali",
            "pa": "Punjabi", "sa": "Sanskrit", "ta": "Tamil", "te": "Telugu",
        }
        return f"{language_names.get(lang, 'Hindi')} ({gender if gender in {'Male', 'Female'} else 'Female'})"

    def _http(self) -> Any:
        import httpx

        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.svara_api_timeout_seconds, connect=5.0)
            )
        return self._client

    async def fetch_remote_voices(self) -> list[dict[str, Any]]:
        """Voice list from the remote Svara server (empty if unavailable)."""
        if not self.api_url:
            return []
        try:
            resp = await self._http().get(f"{self.api_url.rstrip('/')}/v1/voices")
            resp.raise_for_status()
            voices = resp.json().get("voices", [])
            return voices if isinstance(voices, list) else []
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch voices from Svara server: %s", exc)
            return []

    async def _stream_remote(
        self, text: str, voice: str, language: str | None
    ) -> AsyncGenerator[TTSAudioChunk, None]:
        """Stream raw PCM16 24 kHz audio from a Svara inference server.

        Targets Kenpath/svara-tts-inference: ``POST /v1/audio/speech`` with
        ``stream: true`` and ``response_format: pcm`` returns chunked PCM as it
        is generated. Without ``stream: true`` the server buffers the whole clip.
        """
        url = f"{self.api_url.rstrip('/')}/v1/audio/speech"
        payload = {
            "model": self._model_name,
            "input": text,
            "voice": voice,
            "response_format": "pcm",
            "stream": True,
        }
        if settings.svara_buffer_ms is not None:
            payload["buffer_ms"] = settings.svara_buffer_ms
        remainder = b""
        emitted = False
        try:
            async with self._http().stream("POST", url, json=payload) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", "replace")[:500]
                    raise RuntimeError(f"Svara server error {resp.status_code}: {body}")
                rate = int(resp.headers.get("x-sample-rate", self._sample_rate))
                async for data in resp.aiter_bytes():
                    data = remainder + data
                    # Never split a 16-bit sample across chunks.
                    usable = len(data) - (len(data) % 2)
                    remainder = data[usable:]
                    if usable == 0:
                        continue
                    pcm = data[:usable]
                    if rate != self._sample_rate:
                        pcm = float32_to_pcm16(
                            resample_audio(pcm16_to_float32(pcm), rate, self._sample_rate)
                        )
                    emitted = True
                    yield TTSAudioChunk(
                        audio_pcm16=pcm,
                        sample_rate=self._sample_rate,
                        channels=1,
                        is_last=False,
                        duration_seconds=len(pcm) / (self._sample_rate * 2),
                    )
        except Exception as exc:
            # Fall back to macOS speech only before any audio went out and
            # only where it exists; on a Linux GPU server, surface the error.
            if emitted or shutil.which("say") is None:
                raise
            logger.warning("Remote Svara TTS failed (%s); using macOS speech fallback.", exc)
            async for chunk in self._stream_local(text, voice, language):
                yield chunk
            return

        yield TTSAudioChunk(
            audio_pcm16=b"",
            sample_rate=self._sample_rate,
            channels=1,
            is_last=True,
            duration_seconds=0.0,
        )

    async def _stream_local(
        self, text: str, voice: str, language: str | None
    ) -> AsyncGenerator[TTSAudioChunk, None]:
        """Speak locally with macOS ``say`` when the Svara server is unavailable.

        This is a real speech fallback, not Svara inference. Configure
        ``SVARA_API_URL`` to use the Svara model and voices.
        """
        if not text.strip():
            return

        pcm_bytes = await asyncio.to_thread(
            self._synthesize_with_macos_say, text, voice, language
        )
        # Send short PCM frames to keep WebSocket playback responsive. The
        # browser schedules these frames consecutively on its audio clock.
        bytes_per_chunk = max(2, int(self._sample_rate * 2 * 0.1))
        bytes_per_chunk -= bytes_per_chunk % 2
        chunks = [
            pcm_bytes[offset : offset + bytes_per_chunk]
            for offset in range(0, len(pcm_bytes), bytes_per_chunk)
        ]
        for idx, chunk in enumerate(chunks):
            yield TTSAudioChunk(
                audio_pcm16=chunk,
                sample_rate=self._sample_rate,
                channels=1,
                is_last=idx == len(chunks) - 1,
                duration_seconds=len(chunk) / (self._sample_rate * 2),
            )

    def _synthesize_with_macos_say(
        self, text: str, voice: str, language: str | None
    ) -> bytes:
        """Render speech into mono PCM16 using the installed macOS speech voice."""
        say = shutil.which("say")
        if say is None:
            raise RuntimeError(
                "No local speech engine is available. Configure SVARA_API_URL "
                "or run this backend on macOS with the 'say' command installed."
            )

        mac_voice = self._macos_voice(voice, language)
        with tempfile.TemporaryDirectory(prefix="voice-agent-tts-") as temp_dir:
            output_path = Path(temp_dir) / "speech.wav"
            command = [
                say,
                "-o",
                str(output_path),
                "--file-format=WAVE",
                f"--data-format=LEI16@{self._sample_rate}",
            ]
            if mac_voice:
                command.extend(["-v", mac_voice])
            command.append(text)

            try:
                subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                with wave.open(str(output_path), "rb") as audio_file:
                    if (
                        audio_file.getnchannels() != 1
                        or audio_file.getsampwidth() != 2
                        or audio_file.getframerate() != self._sample_rate
                    ):
                        raise RuntimeError("macOS speech engine returned an unexpected WAV format")
                    pcm_bytes = audio_file.readframes(audio_file.getnframes())
            except (OSError, subprocess.SubprocessError, wave.Error) as exc:
                raise RuntimeError(f"macOS speech synthesis failed: {exc}") from exc

        if not pcm_bytes:
            raise RuntimeError("macOS speech engine returned empty audio")
        return pcm_bytes

    @staticmethod
    def _macos_voice(voice: str, language: str | None) -> str | None:
        """Map common Indic voice selections to installed macOS voices."""
        lang = (language or voice.split("_", 1)[0]).lower().split("-", 1)[0]
        return {"hi": "Lekha", "kn": "Soumya", "en": "Aman"}.get(lang)

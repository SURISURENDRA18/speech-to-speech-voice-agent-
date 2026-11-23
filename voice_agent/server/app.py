"""FastAPI application factory with lifespan model initialization and REST/WebSocket routes."""

import asyncio
import logging
import platform
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from voice_agent.config import settings
from voice_agent.llm import (
    MLXQwen3LLMAdapter,
    MockLLMAdapter,
    OpenAICompatibleLLMAdapter,
    Qwen3LLMAdapter,
)
from voice_agent.normalization import IndicTextNormalizer
from voice_agent.server.websocket import voice_websocket_handler
from voice_agent.stt import (
    FasterWhisperSTTAdapter,
    MLXWhisperSTTAdapter,
    MockSTTAdapter,
    WhisperSTTAdapter,
    faster_whisper_available,
    mlx_audio_available,
)
from voice_agent.tts import MockTTSAdapter, SvaraTTSAdapter, load_svara_voices
from voice_agent.vad import SileroVADDetector

logger = logging.getLogger(__name__)


def _use_faster_whisper() -> bool:
    if settings.stt_backend == "transformers":
        return False
    if settings.stt_backend == "faster_whisper":
        return True
    return torch.cuda.is_available() and faster_whisper_available()


async def _preload_models(*adapters: Any) -> None:
    """Load models in the background so the first reply doesn't pay load time."""
    for adapter in adapters:
        preload = getattr(adapter, "preload", None)
        if preload is None:
            continue
        try:
            await preload()
            logger.info("Preloaded %s", getattr(adapter, "model_name", adapter))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Preloading %s failed (will load on first use): %s", adapter, exc)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager: initialize and load models once at application startup."""
    logger.info("Initializing Speech-to-Speech pipeline models...")

    # 1. Silero VAD v6 (TorchScript JIT)
    logger.info("Initializing Silero VAD v6...")
    vad = SileroVADDetector(
        threshold=settings.vad_threshold,
        neg_threshold=settings.vad_neg_threshold,
        min_silence_duration_ms=settings.vad_min_silence_duration_ms,
        speech_pad_ms=settings.vad_speech_pad_ms,
    )
    app.state.vad = vad

    # MLX keeps quantized STT + LLM + Svara resident together (~3.7 GB), so no
    # per-turn model swapping is needed. PyTorch adapters remain the fallback.
    use_mlx = settings.use_mlx and mlx_audio_available()

    # 2. STT: Whisper large-v3-turbo (809M) or Mock
    if settings.use_mock_stt:
        logger.info("Using Mock STT adapter.")
        stt = MockSTTAdapter()
    elif _use_faster_whisper():
        logger.info("Initializing faster-whisper STT adapter (%s)...", settings.whisper_model_name)
        stt = FasterWhisperSTTAdapter(
            model_name=settings.whisper_model_name,
            device="cuda",
            compute_type=settings.faster_whisper_compute_type,
        )
    elif use_mlx:
        logger.info("Initializing MLX Whisper STT adapter (%s)...", settings.mlx_whisper_model)
        stt = MLXWhisperSTTAdapter(model_id=settings.mlx_whisper_model)
    else:
        logger.info(
            "Initializing Whisper STT adapter (%s, 809M params)...",
            settings.whisper_model_name,
        )
        stt = WhisperSTTAdapter(
            model_name=settings.whisper_model_name,
            device=settings.whisper_device,
        )
    app.state.stt = stt

    # 3. LLM: Qwen3 1.7B or Mock
    if settings.use_mock_llm:
        logger.info("Using Mock LLM adapter.")
        llm = MockLLMAdapter()
    elif settings.llm_api_url:
        logger.info("Using remote LLM server %s (%s)", settings.llm_api_url, settings.llm_api_model)
        llm = OpenAICompatibleLLMAdapter(
            base_url=settings.llm_api_url,
            model=settings.llm_api_model,
            api_key=settings.llm_api_key,
            max_new_tokens=settings.llm_max_new_tokens,
            temperature=settings.llm_temperature,
        )
    elif use_mlx:
        logger.info("Initializing MLX Qwen3 LLM adapter (%s)...", settings.mlx_llm_model)
        llm = MLXQwen3LLMAdapter(
            model_id=settings.mlx_llm_model,
            max_new_tokens=settings.llm_max_new_tokens,
            temperature=settings.llm_temperature,
        )
    else:
        logger.info("Initializing Qwen3 1.7B LLM adapter...")
        llm = Qwen3LLMAdapter(
            model_name_or_path=settings.llm_model_name_or_path,
            device=settings.llm_device,
            torch_dtype=settings.llm_torch_dtype,
            max_new_tokens=settings.llm_max_new_tokens,
            temperature=settings.llm_temperature,
        )
    app.state.llm = llm

    # 4. Indic Text Normalization
    logger.info("Initializing Indic Text Normalizer...")
    normalizer = IndicTextNormalizer(
        default_lang=settings.normalization_default_lang,
        use_wfst=True,
    )
    app.state.normalizer = normalizer

    # 5. TTS: Svara TTS or Mock
    if settings.use_mock_tts:
        logger.info("Using Mock TTS adapter.")
        tts = MockTTSAdapter(sample_rate=settings.output_sample_rate)
    else:
        logger.info("Initializing Svara TTS adapter (%s)...", settings.svara_model_name)
        tts = SvaraTTSAdapter(
            model_name=settings.svara_model_name,
            default_voice=settings.svara_voice,
            output_sample_rate=settings.output_sample_rate,
            api_url=settings.svara_api_url,
            keep_loaded=use_mlx,
        )
    app.state.tts = tts

    preload_task: asyncio.Task | None = None
    if settings.preload_models:
        # Each adapter's preload() is a no-op when it swaps weights per turn.
        preload_task = asyncio.create_task(_preload_models(stt, llm, tts))

    logger.info(
        "Speech-to-Speech pipeline ready (MLX: %s, CUDA: %s)", use_mlx, torch.cuda.is_available()
    )
    yield

    if preload_task and not preload_task.done():
        preload_task.cancel()

    logger.info("Shutting down voice agent and releasing resources...")
    vad.reset()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Local Speech-to-Speech Voice Agent",
        description="Multilingual Indian speech-to-speech pipeline (Apple Silicon MLX or CUDA GPU).",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        """Serve the interactive web test client."""
        static_html = settings.workspace_dir / "voice_agent" / "server" / "static" / "index.html"
        if static_html.exists():
            return HTMLResponse(content=static_html.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>Speech-to-Speech Voice Agent</h1><p>Static index.html not found.</p>")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "components": {
                "vad": "silero-v6",
                "stt": app.state.stt.model_name,
                "llm": app.state.llm.model_name,
                "normalizer": "indic-normalization (wfst="
                + str(app.state.normalizer.is_wfst_available)
                + ")",
                "tts": app.state.tts.model_name,
            },
        }

    @app.get("/v1/voices")
    async def voices() -> dict[str, Any]:
        """List available Svara Indic voice profiles."""
        voice_list = load_svara_voices()
        tts = app.state.tts
        if not voice_list and hasattr(tts, "fetch_remote_voices"):
            voice_list = await tts.fetch_remote_voices()
        return {"total": len(voice_list), "voices": voice_list}

    @app.get("/v1/client-config")
    async def client_config() -> dict[str, Any]:
        """Defaults for the browser client."""
        mode = settings.client_playback_mode
        if mode == "auto":
            # A GPU Svara server generates faster than real time, so chunks can
            # play as they arrive; slow local Svara needs per-sentence buffering.
            mode = "fast" if settings.svara_api_url or settings.use_mock_tts else "smooth"
        return {"playback_mode": mode}

    @app.get("/info")
    async def info() -> dict[str, Any]:
        """System hardware, accelerator, and model specifications."""
        mps_available = torch.backends.mps.is_available()
        return {
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "os_version": platform.mac_ver()[0],
            },
            "accelerator": {
                "mps_available": mps_available,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
                "mps_built": torch.backends.mps.is_built(),
                "torch_version": torch.__version__,
            },
            "models": {
                "vad": "Silero VAD v6 (TorchScript JIT)",
                "stt": "Whisper large-v3-turbo (809M parameters)",
                "llm": "Qwen3 1.7B (unsloth/Qwen3-1.7B)",
                "tts": "Svara TTS (SNAC 24kHz)",
                "normalizer": "Indic Text Normalization (19 languages)",
            },
            "memory_optimization": {
                "target_device": "CUDA GPU" if torch.cuda.is_available() else "Apple Silicon Mac",
                "lazy_loading": True,
                "bounded_queues": settings.input_queue_maxsize,
                "short_context_tokens": settings.llm_max_new_tokens,
            },
        }

    # Static assets (sample speech WAV used by the web client's test button)
    static_dir = settings.workspace_dir / "voice_agent" / "server" / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # Register WebSocket endpoint
    app.websocket("/ws/voice")(voice_websocket_handler)

    return app


app = create_app()

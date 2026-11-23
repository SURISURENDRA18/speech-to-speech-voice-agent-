"""Configuration management for the local speech-to-speech voice agent."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Voice agent application settings with environment variable overrides."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server settings
    host: str = Field(default="0.0.0.0", description="Host to bind the server")
    port: int = Field(default=8000, description="Port to bind the server")
    log_level: str = Field(default="INFO", description="Logging level")
    debug: bool = Field(default=False, description="Enable debug mode")

    # Audio & Streaming bounds
    input_sample_rate: int = Field(
        default=16000, description="Default client input audio sample rate (Hz)"
    )
    output_sample_rate: int = Field(
        default=24000, description="TTS output audio sample rate (Hz)"
    )
    audio_channels: int = Field(default=1, description="Number of audio channels (mono)")
    input_queue_maxsize: int = Field(
        default=100, description="Max input audio frames queued before dropping"
    )
    output_queue_maxsize: int = Field(
        default=100, description="Max output audio frames queued before dropping"
    )
    max_speech_duration_seconds: float = Field(
        default=30.0, description="Max duration of a single speech utterance"
    )

    # Silero VAD settings
    vad_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Speech detection probability threshold"
    )
    vad_neg_threshold: float | None = Field(
        default=0.35, ge=0.0, le=1.0, description="Silence detection threshold"
    )
    vad_min_silence_duration_ms: int = Field(
        default=300, description="Silence duration in ms to trigger end of turn"
    )
    vad_speech_pad_ms: int = Field(
        default=30, description="Speech padding in ms around detected utterance"
    )
    vad_chunk_size: int = Field(
        default=512, description="VAD frame sample count (512 for 16kHz)"
    )

    # Apple Silicon MLX backend: quantized Whisper + Qwen3 that stay resident
    # alongside Svara (~3.7 GB total) instead of reloading PyTorch models
    # every turn. Falls back to the PyTorch adapters when MLX is unavailable.
    use_mlx: bool = Field(
        default=True, description="Use MLX STT/LLM on Apple Silicon when available"
    )
    mlx_whisper_model: str = Field(
        default="mlx-community/whisper-large-v3-turbo-asr-8bit",
        description="MLX Whisper checkpoint",
    )
    mlx_llm_model: str = Field(
        default="mlx-community/Qwen3-1.7B-4bit", description="MLX Qwen3 checkpoint"
    )
    preload_models: bool = Field(
        default=True, description="Load models in the background at server startup"
    )
    keep_models_loaded: bool | None = Field(
        default=None,
        description=(
            "Keep PyTorch STT/LLM resident between turns. None = auto "
            "(True on CUDA GPUs, False on 8 GB Macs)"
        ),
    )
    overlap_llm_tts: bool = Field(
        default=True,
        description="Start speaking the first sentence while the LLM is still generating",
    )
    client_playback_mode: Literal["auto", "smooth", "fast"] = Field(
        default="auto",
        description=(
            "Browser playback default. auto = fast when TTS is a GPU server "
            "(faster than real time), smooth for slow local Svara"
        ),
    )

    # Whisper STT settings
    whisper_model_name: str = Field(
        default="large-v3-turbo",
        description="Whisper model (large-v3-turbo has 809M parameters)",
    )
    whisper_device: Literal["cuda", "mps", "cpu", "auto"] = Field(
        default="auto", description="Device for Whisper inference"
    )
    stt_backend: Literal["auto", "faster_whisper", "transformers"] = Field(
        default="auto",
        description="auto = faster-whisper on CUDA when installed, else transformers",
    )
    faster_whisper_compute_type: str = Field(
        default="float16", description="CTranslate2 compute type on GPU"
    )
    whisper_language: str | None = Field(
        default=None, description="STT language code (None for auto-detection)"
    )
    use_mock_stt: bool = Field(
        default=False,
        description="Whether to use mock STT adapter (useful in testing or low memory)",
    )

    # Qwen3 1.7B LLM settings
    llm_model_name_or_path: str = Field(
        default="unsloth/Qwen3-1.7B",
        description="Path or HuggingFace ID for Qwen3 1.7B",
    )
    llm_device: Literal["cuda", "mps", "cpu", "auto"] = Field(
        default="auto", description="Device for LLM inference"
    )
    llm_torch_dtype: Literal["bfloat16", "float16", "float32"] = Field(
        default="bfloat16", description="Torch dtype for LLM weights"
    )
    llm_max_new_tokens: int = Field(
        default=128,
        description=(
            "Max tokens per reply. Indic scripts need ~3 tokens per word, so "
            "values below ~100 cut replies off mid-sentence"
        ),
    )
    llm_temperature: float = Field(
        default=0.5, description="Generation temperature for natural spoken replies"
    )
    conversation_max_turns: int = Field(
        default=6,
        ge=1,
        description="User/assistant exchanges remembered per session and sent to the LLM",
    )
    system_prompt: str = Field(
        default=(
            "You are a warm, attentive multilingual Indian voice assistant having a "
            "spoken conversation. Listen to exactly what the user said and respond "
            "to that, using what they told you earlier in the conversation (their "
            "name, plans, questions). Match their tone and formality. Answer in one "
            "or two short spoken sentences; when it helps the conversation, end "
            "with a brief follow-up question. Never use lists, markdown, emojis or "
            "symbols, because your words are spoken aloud. If you do not know "
            "something or lack live information, say so briefly instead of guessing."
        ),
        description="System prompt for the conversation",
    )
    llm_api_url: str | None = Field(
        default=None,
        description="OpenAI-compatible LLM server (e.g. vLLM serving Qwen3) base URL",
    )
    llm_api_model: str = Field(
        default="Qwen/Qwen3-1.7B", description="Model name sent to the LLM server"
    )
    llm_api_key: str = Field(default="EMPTY", description="API key for the LLM server")
    use_mock_llm: bool = Field(
        default=False,
        description="Whether to use mock LLM adapter (useful in testing or low memory)",
    )

    # Indic Text Normalization settings
    normalization_enabled: bool = Field(
        default=True, description="Enable Indic text normalization before TTS"
    )
    normalization_default_lang: str = Field(
        default="hi", description="Default language code for normalization"
    )

    # Svara TTS settings
    svara_model_name: str = Field(
        default="svara-tts-v1", description="Svara model identifier"
    )
    svara_voice: str = Field(
        default="hi_female", description="Default voice profile ID"
    )
    svara_device: Literal["cuda", "mps", "cpu", "auto"] = Field(
        default="auto", description="Device for SNAC audio decoder"
    )
    svara_api_url: str | None = Field(
        default=None,
        description="Optional remote Svara API URL (OpenAI compatible) when running vLLM externally",
    )
    svara_buffer_ms: int | None = Field(
        default=None,
        description=(
            "Audio the remote Svara server buffers before streaming starts "
            "(server default 500 ms). Lower = faster first audio."
        ),
    )
    tts_min_frame_ms: int = Field(
        default=160,
        ge=20,
        description="Smallest audio frame sent to the client; tiny TTS pieces are merged",
    )
    tts_max_chars_per_request: int = Field(
        default=220,
        ge=40,
        description=(
            "When the full reply is known before speaking (no LLM/TTS overlap), "
            "sentences are merged into TTS requests up to this length"
        ),
    )
    svara_api_timeout_seconds: float = Field(
        default=60.0, description="Timeout for one remote Svara synthesis request"
    )
    use_mock_tts: bool = Field(
        default=False,
        description="Whether to use mock TTS adapter (useful in testing or without GPU)",
    )

    # Path to local upstream packages (extracted from workspace zips)
    workspace_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent,
        description="Root workspace directory",
    )


settings = Settings()

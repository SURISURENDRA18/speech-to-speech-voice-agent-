"""WebSocket Protocol Schemas for Speech-to-Speech communication."""

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

# ============================================================================
# Client -> Server Message Schemas
# ============================================================================


class SessionStartConfig(BaseModel):
    """Configuration sent by the client when starting a voice session."""

    sample_rate: int = Field(
        default=16000, description="Input audio sample rate (Hz)"
    )
    channels: int = Field(default=1, description="Number of audio channels")
    language: str = Field(default="hi", description="Conversation language (e.g. 'hi', 'en')")
    voice: str = Field(default="hi_female", description="Voice profile ID")
    vad_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="VAD speech probability threshold"
    )
    min_silence_duration_ms: int = Field(
        default=300, description="Silence duration in ms to trigger end of turn"
    )
    system_prompt: str | None = Field(
        default=None, description="Optional custom system prompt"
    )


class ClientStartMessage(BaseModel):
    """Client start message."""

    type: Literal["start"] = "start"
    config: SessionStartConfig = Field(default_factory=SessionStartConfig)


class ClientAudioMessage(BaseModel):
    """Client audio chunk message (JSON base64 format)."""

    type: Literal["audio"] = "audio"
    data: str = Field(description="Base64-encoded PCM audio bytes")


class ClientInterruptMessage(BaseModel):
    """Client explicit barge-in / interrupt message."""

    type: Literal["interrupt"] = "interrupt"


class ClientStopMessage(BaseModel):
    """Client stop session message."""

    type: Literal["stop"] = "stop"


# ============================================================================
# Server -> Client Message Schemas
# ============================================================================


class ServerConnectedMessage(BaseModel):
    """Sent when WebSocket connection is accepted."""

    type: Literal["connected"] = "connected"
    session_id: str
    server_time: float = Field(default_factory=time.time)


class ServerStartedMessage(BaseModel):
    """Sent when voice session is started."""

    type: Literal["session_started"] = "session_started"
    session_id: str
    config: dict[str, Any]


class ServerVADSpeechStartMessage(BaseModel):
    """Sent when VAD detects beginning of user speech."""

    type: Literal["vad_speech_start"] = "vad_speech_start"
    timestamp: float = Field(default_factory=time.time)


class ServerVADSpeechEndMessage(BaseModel):
    """Sent when VAD detects end of user speech (end of turn)."""

    type: Literal["vad_speech_end"] = "vad_speech_end"
    timestamp: float = Field(default_factory=time.time)
    duration_ms: int = 0


class ServerTranscriptMessage(BaseModel):
    """Sent when STT transcribes user speech."""

    type: Literal["transcript"] = "transcript"
    text: str
    language: str = "hi"
    is_final: bool = True


class ServerLLMChunkMessage(BaseModel):
    """Sent as LLM generates token or text deltas."""

    type: Literal["llm_chunk"] = "llm_chunk"
    delta: str
    accumulated: str


class ServerTTSChunkMessage(BaseModel):
    """Sent as TTS synthesizes audio chunks."""

    type: Literal["tts_chunk"] = "tts_chunk"
    audio: str = Field(description="Base64-encoded 16-bit PCM bytes")
    sample_rate: int = 24000
    channels: int = 1
    format: str = "pcm16"
    is_last: bool = False


class ServerInterruptedMessage(BaseModel):
    """Sent when barge-in cancels assistant playback."""

    type: Literal["interrupted"] = "interrupted"
    reason: str = "user_speech"


class ServerTurnCompleteMessage(BaseModel):
    """Sent when the entire turn (STT + LLM + TTS) completes."""

    type: Literal["turn_complete"] = "turn_complete"
    user_text: str
    assistant_text: str


class ServerErrorMessage(BaseModel):
    """Sent when an error occurs during session processing."""

    type: Literal["error"] = "error"
    code: str
    message: str


class ServerStoppedMessage(BaseModel):
    """Sent when session is ended."""

    type: Literal["session_stopped"] = "session_stopped"

"""Speech-to-Text module."""

from .base import STTBase, TranscriptionResult
from .faster_whisper_stt import FasterWhisperSTTAdapter, faster_whisper_available
from .mlx_whisper_stt import MLXWhisperSTTAdapter, mlx_audio_available
from .mock_stt import MockSTTAdapter
from .whisper_stt import WhisperSTTAdapter

__all__ = [
    "FasterWhisperSTTAdapter",
    "MLXWhisperSTTAdapter",
    "MockSTTAdapter",
    "STTBase",
    "TranscriptionResult",
    "WhisperSTTAdapter",
    "faster_whisper_available",
    "mlx_audio_available",
]

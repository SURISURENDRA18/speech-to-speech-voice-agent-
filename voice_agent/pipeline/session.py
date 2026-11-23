"""Voice session state machine and speech-to-speech turn coordinator."""

import asyncio
import contextlib
import logging
from collections.abc import Callable, Coroutine
from enum import Enum
from typing import Any

import numpy as np

from voice_agent.audio import (
    BoundedAudioAccumulator,
    pcm16_to_float32,
    pcm_bytes_to_base64,
    resample_audio,
    slice_into_frames,
)
from voice_agent.config import settings
from voice_agent.llm.base import LLMBase
from voice_agent.normalization.indic_norm import IndicTextNormalizer
from voice_agent.protocol import (
    ServerErrorMessage,
    ServerInterruptedMessage,
    ServerLLMChunkMessage,
    ServerTranscriptMessage,
    ServerTTSChunkMessage,
    ServerTurnCompleteMessage,
    ServerVADSpeechEndMessage,
    ServerVADSpeechStartMessage,
    SessionStartConfig,
)
from voice_agent.stt.base import STTBase
from voice_agent.tts.base import TTSBase
from voice_agent.vad.detector import SileroVADDetector

logger = logging.getLogger(__name__)


class SessionState(str, Enum):
    """Voice session state."""

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    CLOSED = "closed"


class VoiceSession:
    """Manages full-duplex conversational audio pipeline for a single WebSocket connection.

    Coordinates VAD, STT, LLM, Normalization, and TTS with support for low latency,
    streaming, cancellation, and barge-in / interruption.
    """

    def __init__(
        self,
        session_id: str,
        vad: SileroVADDetector,
        stt: STTBase,
        llm: LLMBase,
        normalizer: IndicTextNormalizer,
        tts: TTSBase,
        send_callback: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
    ):
        self.session_id = session_id
        self.vad = vad
        self.stt = stt
        self.llm = llm
        self.normalizer = normalizer
        self.tts = tts
        self._send_callback = send_callback

        self.state = SessionState.IDLE
        self.config = SessionStartConfig()
        self.conversation_history: list[dict[str, str]] = []

        # Audio buffer for handling arbitrary incoming sample rates and slicing into 512-sample frames
        self._input_sample_accumulator = BoundedAudioAccumulator(
            sample_rate=16000,
            max_seconds=settings.max_speech_duration_seconds,
        )
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(
            maxsize=settings.input_queue_maxsize
        )

        self._active_turn_task: asyncio.Task | None = None
        self._processor_task: asyncio.Task | None = None
        self._is_running = False

    async def start(self, config: SessionStartConfig | None = None) -> None:
        """Start the session and begin processing audio stream."""
        if config:
            self.config = config

        self.state = SessionState.LISTENING
        self._is_running = True
        self.vad.reset()

        # Start audio consumer worker
        self._processor_task = asyncio.create_task(self._process_audio_stream())
        logger.info("Session %s started with language: %s, voice: %s", self.session_id, self.config.language, self.config.voice)

    async def push_audio(self, pcm_bytes: bytes) -> None:
        """Push raw incoming audio bytes into bounded queue."""
        if not self._is_running or self.state == SessionState.CLOSED:
            return

        try:
            self._audio_queue.put_nowait(pcm_bytes)
        except asyncio.QueueFull:
            logger.warning("Session %s input queue full; dropping frame", self.session_id)

    async def interrupt(self, reason: str = "user_speech") -> None:
        """Cancel ongoing generation and playback immediately (Barge-in)."""
        if self.state in (SessionState.PROCESSING, SessionState.SPEAKING):
            logger.info("Barge-in / Interruption triggered on session %s (reason: %s)", self.session_id, reason)
            if self._active_turn_task and not self._active_turn_task.done():
                self._active_turn_task.cancel()
                try:
                    await self._active_turn_task
                except asyncio.CancelledError:
                    pass
                self._active_turn_task = None

            self.state = SessionState.LISTENING
            await self._send_event(ServerInterruptedMessage(reason=reason).model_dump())

    async def stop(self) -> None:
        """Stop active session and cleanup all tasks and buffers."""
        self._is_running = False
        self.state = SessionState.CLOSED

        if self._active_turn_task and not self._active_turn_task.done():
            self._active_turn_task.cancel()

        if self._processor_task and not self._processor_task.done():
            self._processor_task.cancel()

        self._input_sample_accumulator.clear()
        self.vad.reset()
        logger.info("Session %s stopped.", self.session_id)

    async def _send_event(self, event_data: dict[str, Any]) -> None:
        """Send event over WebSocket."""
        try:
            await self._send_callback(event_data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to send WebSocket event on session %s: %s", self.session_id, exc)

    async def _process_audio_stream(self) -> None:
        """Background worker consuming audio chunks and feeding VAD."""
        frame_buffer = np.empty(0, dtype=np.float32)

        try:
            while self._is_running:
                pcm_bytes = await self._audio_queue.get()
                if not pcm_bytes:
                    continue

                samples = pcm16_to_float32(pcm_bytes)
                if self.config.sample_rate != 16000:
                    samples = resample_audio(samples, self.config.sample_rate, 16000)

                # Concatenate with unaligned remainder
                if len(frame_buffer) > 0:
                    samples = np.concatenate([frame_buffer, samples])

                # Process all complete 512-sample frames
                frames = list(slice_into_frames(samples, frame_size=512))
                num_processed = len(frames) * 512
                frame_buffer = samples[num_processed:]

                for frame in frames:
                    events = self.vad.process_chunk(frame)
                    for event in events:
                        # While the reply is still being computed (STT/LLM/model
                        # swaps take a long time on 8 GB Macs), background noise
                        # must not cancel the turn or start a second concurrent
                        # one. Voice barge-in stays enabled once audio is playing;
                        # the explicit client "interrupt" message always works.
                        if self.state == SessionState.PROCESSING:
                            logger.debug(
                                "Ignoring VAD %s while turn is processing", event.event_type
                            )
                            continue

                        if event.event_type == "speech_start":
                            logger.debug("Speech start detected")
                            # If assistant is currently speaking, trigger barge-in!
                            if self.state == SessionState.SPEAKING:
                                await self.interrupt(reason="user_speech")

                            await self._send_event(ServerVADSpeechStartMessage().model_dump())

                        elif event.event_type == "speech_end":
                            logger.debug("Speech end detected (duration samples: %s)", event.duration_samples)
                            duration_ms = int(event.duration_samples * 1000 / 16000)
                            await self._send_event(
                                ServerVADSpeechEndMessage(duration_ms=duration_ms).model_dump()
                            )

                            if self._active_turn_task and not self._active_turn_task.done():
                                logger.debug("Previous turn still running; skipping new turn")
                                continue

                            if event.speech_audio is not None and len(event.speech_audio) > 0:
                                # Mark busy immediately so following frames are not
                                # treated as barge-in before the task starts running.
                                self.state = SessionState.PROCESSING
                                # Dispatch turn pipeline in background task
                                self._active_turn_task = asyncio.create_task(
                                    self._run_turn(event.speech_audio)
                                )

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("Error in audio stream processing loop")
            await self._send_event(
                ServerErrorMessage(code="AUDIO_PROCESS_ERROR", message=str(exc)).model_dump()
            )

    async def _run_turn(self, speech_audio: np.ndarray) -> None:
        """Executes one conversational turn: STT -> LLM -> Normalization -> TTS."""
        try:
            self.state = SessionState.PROCESSING

            # Keep only one large neural model resident at a time on 8 GB Macs.
            await self.tts.release_model()

            # 1. STT: Transcribe user speech
            try:
                stt_result = await self.stt.transcribe(
                    speech_audio, sample_rate=16000, language=self.config.language
                )
            finally:
                await self.stt.release_model()
            user_text = stt_result.text.strip()
            detected_lang = stt_result.language or self.config.language

            await self._send_event(
                ServerTranscriptMessage(
                    text=user_text,
                    language=detected_lang,
                    is_final=True,
                ).model_dump()
            )

            # If no speech was recognized, return to listening
            if not user_text:
                self.state = SessionState.LISTENING
                return

            # 2. LLM: Stream assistant response. When the LLM keeps its weights
            # resident, each finished sentence goes straight to TTS while the
            # LLM keeps generating, so the first audio doesn't wait for the
            # whole reply. On 8 GB Macs that swap models, TTS waits instead.
            accumulated_response = ""
            sentence_buffer = ""
            response_sentences: list[str] = []
            dispatched = 0
            overlap = (
                settings.overlap_llm_tts
                and not self.llm.swaps_weights
                and not getattr(self.tts, "uses_local_mlx", False)
            )
            sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
            tts_task: asyncio.Task | None = (
                asyncio.create_task(self._tts_worker(sentence_queue, detected_lang))
                if overlap
                else None
            )

            def dispatch(sentence: str) -> None:
                nonlocal dispatched
                dispatched += 1
                if tts_task is not None:
                    sentence_queue.put_nowait(sentence)
                else:
                    response_sentences.append(sentence)

            try:
                try:
                    async for token in self.llm.generate_stream(
                        prompt=user_text,
                        history=self.conversation_history,
                        system_prompt=self.config.system_prompt,
                    ):
                        accumulated_response += token
                        sentence_buffer += token

                        await self._send_event(
                            ServerLLMChunkMessage(
                                delta=token,
                                accumulated=accumulated_response,
                            ).model_dump()
                        )

                        if any(p in token for p in (".", "!", "?", "।", "\n")) and sentence_buffer.strip():
                            dispatch(sentence_buffer.strip())
                            sentence_buffer = ""
                        if tts_task is not None and tts_task.done():
                            break  # TTS failed; stop generating and surface it below
                finally:
                    await self.llm.release_model()

                # Synthesize the trailing text only when it is the whole reply. A
                # fragment after a complete sentence is almost always the LLM being
                # cut off by max_new_tokens; speaking it ends the reply mid-word.
                if sentence_buffer.strip() and dispatched == 0:
                    dispatch(sentence_buffer.strip())
                elif sentence_buffer.strip():
                    logger.debug("Skipping truncated trailing fragment: %r", sentence_buffer)

                if tts_task is not None:
                    sentence_queue.put_nowait(None)
                    await tts_task
                else:
                    for sentence in response_sentences:
                        await self._synthesize_and_stream(sentence, detected_lang)
            finally:
                if tts_task is not None and not tts_task.done():
                    tts_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await tts_task

            # 3. Turn complete: record conversation history
            self.conversation_history.append({"role": "user", "content": user_text})
            self.conversation_history.append({"role": "assistant", "content": accumulated_response.strip()})

            await self._send_event(
                ServerTurnCompleteMessage(
                    user_text=user_text,
                    assistant_text=accumulated_response.strip(),
                ).model_dump()
            )

        except asyncio.CancelledError:
            logger.info("Session %s turn cancelled.", self.session_id)
            raise
        except Exception as exc:
            logger.exception("Error executing turn on session %s", self.session_id)
            await self._send_event(
                ServerErrorMessage(code="TURN_ERROR", message=str(exc)).model_dump()
            )
        finally:
            if self.state != SessionState.CLOSED:
                self.state = SessionState.LISTENING

    async def _tts_worker(self, queue: "asyncio.Queue[str | None]", language: str) -> None:
        """Speak sentences in order as the LLM produces them."""
        while True:
            sentence = await queue.get()
            if sentence is None:
                return
            await self._synthesize_and_stream(sentence, language)

    async def _synthesize_and_stream(self, text: str, language: str) -> None:
        """Normalize text and stream TTS chunks."""
        # 1. Indic Text Normalization
        normalized_text = (
            self.normalizer.normalize(text, lang=language)
            if settings.normalization_enabled
            else text
        )

        # 2. TTS audio chunk streaming
        self.state = SessionState.SPEAKING
        async for chunk in self.tts.synthesize_stream(
            text=normalized_text,
            voice=self.config.voice,
            language=language,
        ):
            b64_audio = pcm_bytes_to_base64(chunk.audio_pcm16)
            await self._send_event(
                ServerTTSChunkMessage(
                    audio=b64_audio,
                    sample_rate=chunk.sample_rate,
                    channels=chunk.channels,
                    format="pcm16",
                    is_last=chunk.is_last,
                ).model_dump()
            )

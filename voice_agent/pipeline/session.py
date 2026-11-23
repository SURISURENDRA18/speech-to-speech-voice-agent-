"""Voice session state machine and speech-to-speech turn coordinator.

One ``VoiceSession`` exists per WebSocket connection. It owns that
connection's VAD state, conversation memory and in-flight turn task. Model
adapters (STT/LLM/TTS/normalizer) are shared, stateless-per-call services.

Turn flow::

    VAD speech_end -> _run_turn
        _transcribe        STT -> "transcript" event
        _generate_and_speak
            LLM tokens -> "llm_chunk" events -> SentenceSegmenter
            sentences -> _speak -> normalizer -> TTS -> PCM16Coalescer -> "tts_chunk"
        ConversationMemory.record_turn (only what was actually spoken)
"""

import asyncio
import contextlib
import logging
from collections.abc import Callable, Coroutine
from enum import Enum
from typing import Any

import numpy as np

from voice_agent.audio import (
    PCM16Coalescer,
    pcm16_to_float32,
    pcm_bytes_to_base64,
    resample_audio,
    slice_into_frames,
)
from voice_agent.config import settings
from voice_agent.llm.base import LLMBase
from voice_agent.normalization.indic_norm import IndicTextNormalizer
from voice_agent.pipeline.conversation import ConversationMemory, build_system_prompt
from voice_agent.pipeline.segmenter import (
    SentenceSegmenter,
    ends_with_terminator,
    group_for_synthesis,
)
from voice_agent.pipeline.synthesis import (
    GUARD_POLL_SECONDS,
    PlayoutClock,
    SynthesisJob,
    SynthesisRateTracker,
    reply_can_play_through,
)
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
from voice_agent.tts.base import TTSAudioChunk, TTSBase
from voice_agent.vad.detector import SileroVADDetector, VADEvent

logger = logging.getLogger(__name__)

VAD_SAMPLE_RATE = 16000
VAD_FRAME_SAMPLES = 512

SendCallback = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class SessionState(str, Enum):
    """Voice session state."""

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    CLOSED = "closed"


class VoiceSession:
    """Manages the full-duplex conversational audio pipeline for one connection.

    Coordinates VAD, STT, LLM, normalization and TTS with streaming,
    cancellation and barge-in. The ``vad`` instance must not be shared with
    another session: it carries per-stream recurrent state.
    """

    def __init__(
        self,
        session_id: str,
        vad: SileroVADDetector,
        stt: STTBase,
        llm: LLMBase,
        normalizer: IndicTextNormalizer,
        tts: TTSBase,
        send_callback: SendCallback,
        memory: ConversationMemory | None = None,
        tts_rate: SynthesisRateTracker | None = None,
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
        # `is not None`: an empty ConversationMemory is falsy (it defines __len__).
        self.memory = (
            memory
            if memory is not None
            else ConversationMemory(max_turns=settings.conversation_max_turns)
        )
        # Measures the shared TTS backend, so the server passes one shared tracker.
        self.tts_rate = tts_rate or SynthesisRateTracker(settings.tts_assumed_realtime_factor)
        self._tts_slots = asyncio.Semaphore(settings.tts_parallel_requests)

        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(
            maxsize=settings.input_queue_maxsize
        )
        self._active_turn_task: asyncio.Task | None = None
        self._processor_task: asyncio.Task | None = None
        self._is_running = False

    # ------------------------------------------------------------------
    # Public lifecycle API (called by the WebSocket handler)
    # ------------------------------------------------------------------

    async def start(self, config: SessionStartConfig | None = None) -> None:
        """Start the session and begin processing the audio stream."""
        if config:
            self.config = config

        self.state = SessionState.LISTENING
        self._is_running = True
        self.vad.reset()

        if self._processor_task is None or self._processor_task.done():
            self._processor_task = asyncio.create_task(self._process_audio_stream())
        logger.info(
            "Session %s started (language=%s, voice=%s)",
            self.session_id,
            self.config.language,
            self.config.voice,
        )

    async def push_audio(self, pcm_bytes: bytes) -> None:
        """Queue raw client PCM16 audio; drops frames if the consumer is behind."""
        if not self._is_running or self.state == SessionState.CLOSED:
            return
        try:
            self._audio_queue.put_nowait(pcm_bytes)
        except asyncio.QueueFull:
            logger.warning("Session %s input queue full; dropping frame", self.session_id)

    async def interrupt(self, reason: str = "user_speech") -> None:
        """Cancel ongoing generation and playback immediately (barge-in)."""
        if self.state not in (SessionState.PROCESSING, SessionState.SPEAKING):
            return
        logger.info("Barge-in on session %s (reason: %s)", self.session_id, reason)
        await self._cancel_active_turn()
        self.state = SessionState.LISTENING
        await self._send_event(ServerInterruptedMessage(reason=reason).model_dump())

    async def stop(self) -> None:
        """Stop the session and release its tasks and buffers."""
        self._is_running = False
        self.state = SessionState.CLOSED
        for task in (self._active_turn_task, self._processor_task):
            if task and not task.done():
                task.cancel()
        self.vad.reset()
        logger.info("Session %s stopped.", self.session_id)

    # ------------------------------------------------------------------
    # Audio input -> VAD
    # ------------------------------------------------------------------

    async def _process_audio_stream(self) -> None:
        """Background worker: decode client audio, frame it, and feed the VAD."""
        remainder = np.empty(0, dtype=np.float32)
        try:
            while self._is_running:
                pcm_bytes = await self._audio_queue.get()
                if not pcm_bytes:
                    continue

                samples = pcm16_to_float32(pcm_bytes)
                if self.config.sample_rate != VAD_SAMPLE_RATE:
                    samples = resample_audio(samples, self.config.sample_rate, VAD_SAMPLE_RATE)
                if len(remainder):
                    samples = np.concatenate([remainder, samples])

                frames = list(slice_into_frames(samples, frame_size=VAD_FRAME_SAMPLES))
                remainder = samples[len(frames) * VAD_FRAME_SAMPLES :]

                for frame in frames:
                    for event in self.vad.process_chunk(frame):
                        await self._handle_vad_event(event)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("Error in audio stream processing loop")
            await self._send_event(
                ServerErrorMessage(code="AUDIO_PROCESS_ERROR", message=str(exc)).model_dump()
            )

    async def _handle_vad_event(self, event: VADEvent) -> None:
        # While a reply is being computed (STT/LLM can take seconds on 8 GB
        # Macs), background noise must not cancel the turn or start a second
        # concurrent one. Voice barge-in stays enabled once audio is playing;
        # the explicit client "interrupt" message always works.
        if self.state == SessionState.PROCESSING:
            logger.debug("Ignoring VAD %s while turn is processing", event.event_type)
            return

        if event.event_type == "speech_start":
            if self.state == SessionState.SPEAKING:
                await self.interrupt(reason="user_speech")
            await self._send_event(ServerVADSpeechStartMessage().model_dump())
            return

        if event.event_type != "speech_end":
            return

        duration_ms = int(event.duration_samples * 1000 / VAD_SAMPLE_RATE)
        await self._send_event(ServerVADSpeechEndMessage(duration_ms=duration_ms).model_dump())

        if self._active_turn_task and not self._active_turn_task.done():
            logger.debug("Previous turn still running; skipping new turn")
            return
        if event.speech_audio is None or len(event.speech_audio) == 0:
            return

        # Mark busy before the task runs so following frames are not treated
        # as barge-in.
        self.state = SessionState.PROCESSING
        self._active_turn_task = asyncio.create_task(self._run_turn(event.speech_audio))

    # ------------------------------------------------------------------
    # One conversational turn
    # ------------------------------------------------------------------

    async def _run_turn(self, speech_audio: np.ndarray) -> None:
        """Run STT -> LLM -> normalization -> TTS for one user utterance."""
        user_message = ""
        spoken: list[str] = []  # sentences whose audio reached the client
        try:
            self.state = SessionState.PROCESSING
            # Keep only one large neural model resident at a time on 8 GB Macs.
            await self.tts.release_model()

            user_text, language = await self._transcribe(speech_audio)
            if not user_text:
                return

            user_message = self.memory.compose_user_message(user_text)
            reply = await self._generate_and_speak(user_message, language, spoken)

            self.memory.record_turn(user_message, " ".join(spoken))
            await self._send_event(
                ServerTurnCompleteMessage(user_text=user_text, assistant_text=reply).model_dump()
            )
        except asyncio.CancelledError:
            # Barge-in: remember what the user said and what they actually heard.
            self.memory.record_turn(user_message, " ".join(spoken))
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

    async def _transcribe(self, speech_audio: np.ndarray) -> tuple[str, str]:
        """Transcribe the utterance and emit the ``transcript`` event."""
        try:
            result = await self.stt.transcribe(
                speech_audio, sample_rate=VAD_SAMPLE_RATE, language=self.config.language
            )
        finally:
            await self.stt.release_model()

        text = result.text.strip()
        language = result.language or self.config.language
        await self._send_event(
            ServerTranscriptMessage(text=text, language=language, is_final=True).model_dump()
        )
        return text, language

    def _overlap_llm_and_tts(self) -> bool:
        """Speak sentence 1 while the LLM still generates, when memory allows."""
        return (
            settings.overlap_llm_tts
            and not self.llm.swaps_weights
            and not getattr(self.tts, "uses_local_mlx", False)
        )

    async def _generate_and_speak(
        self, user_message: str, language: str, spoken: list[str]
    ) -> str:
        """Stream the LLM reply, speak it sentence by sentence, return the full text."""
        system_prompt = build_system_prompt(
            self.config.system_prompt or settings.system_prompt, language
        )
        segmenter = SentenceSegmenter()
        sentences: list[str] = []
        reply = ""
        playout = PlayoutClock()
        jobs: list[SynthesisJob] = []

        # Overlap: every sentence starts synthesizing the moment the LLM emits
        # it (up to tts_parallel_requests at once); _speak_from_queue plays them
        # strictly in order.
        queue: asyncio.Queue[SynthesisJob | None] | None = (
            asyncio.Queue() if self._overlap_llm_and_tts() else None
        )
        tts_task = (
            asyncio.create_task(self._speak_from_queue(queue, playout, spoken, jobs))
            if queue is not None
            else None
        )

        def accept(sentence: str) -> None:
            sentences.append(sentence)
            if queue is not None:
                jobs.append(self._start_synthesis(sentence, language))
                queue.put_nowait(jobs[-1])

        try:
            try:
                async for delta in self.llm.generate_stream(
                    prompt=user_message,
                    history=self.memory.history(),
                    system_prompt=system_prompt,
                ):
                    reply += delta
                    await self._send_event(
                        ServerLLMChunkMessage(delta=delta, accumulated=reply).model_dump()
                    )
                    for sentence in segmenter.push(delta):
                        accept(sentence)
                    if tts_task is not None and tts_task.done():
                        break  # TTS failed; stop generating and surface it below
            finally:
                await self.llm.release_model()

            # A terminated tail is the reply's last sentence. An unterminated
            # tail after earlier sentences is almost always the LLM being cut by
            # max_new_tokens; speaking it would end the reply mid-word.
            tail = segmenter.flush()
            if tail and (ends_with_terminator(tail) or not sentences):
                accept(tail)
            elif tail:
                logger.debug("Skipping truncated trailing fragment: %r", tail)

            if tts_task is not None and queue is not None:
                queue.put_nowait(None)
                await tts_task
            else:
                for text in group_for_synthesis(sentences, settings.tts_max_chars_per_request):
                    jobs.append(self._start_synthesis(text, language))
                    await self._speak(jobs[-1], playout, spoken, jobs)
        finally:
            if tts_task is not None and not tts_task.done():
                tts_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await tts_task
            for job in jobs:
                await job.cancel()

        return reply.strip()

    async def _speak_from_queue(
        self,
        queue: "asyncio.Queue[SynthesisJob | None]",
        playout: PlayoutClock,
        spoken: list[str],
        reply_jobs: list[SynthesisJob],
    ) -> None:
        """Play sentences in order as their synthesis jobs are queued."""
        while (job := await queue.get()) is not None:
            await self._speak(job, playout, spoken, reply_jobs)

    def _start_synthesis(self, text: str, language: str) -> SynthesisJob:
        """Normalize ``text`` and start synthesizing it in the background."""
        tts_text = (
            self.normalizer.normalize(text, lang=language)
            if settings.normalization_enabled
            else text
        )
        return SynthesisJob(
            self.tts,
            text=text,
            tts_text=tts_text,
            voice=self.config.voice,
            language=language,
            slots=self._tts_slots,
            tracker=self.tts_rate,
        )

    async def _speak(
        self,
        job: SynthesisJob,
        playout: PlayoutClock,
        spoken: list[str],
        reply_jobs: list[SynthesisJob],
    ) -> None:
        """Stream one job's audio once the rest of the reply can play without gaps.

        ``reply_jobs`` is every job of this reply in playing order; it may still
        grow while the LLM is generating.
        """
        self.state = SessionState.SPEAKING
        while not reply_can_play_through(
            reply_jobs[reply_jobs.index(job) :],
            queued_seconds=playout.queued(),
            tracker=self.tts_rate,
            margin=settings.tts_underrun_margin,
            holdback_seconds=settings.tts_min_frame_ms / 1000,
        ):
            await asyncio.sleep(GUARD_POLL_SECONDS)

        out = _AudioOut(self._send_audio, playout, self.tts.sample_rate)
        backlog_sent = False
        while (item := await job.chunks.get()) is not None:
            if isinstance(item, Exception):
                raise item
            await out.write(item)
            if not backlog_sent and job.chunks.empty():
                await out.flush()  # the guard counted the whole backlog as sent
                backlog_sent = True

        # Always close the request with is_last=True ("smooth" playback flushes on it).
        await out.close()
        if out.heard:
            spoken.append(job.text)

    async def _send_audio(
        self, pcm: bytes, sample_rate: int, channels: int, *, is_last: bool
    ) -> bool:
        """Send one ``tts_chunk``; returns True when it carried audio."""
        if not pcm and not is_last:
            return False
        await self._send_event(
            ServerTTSChunkMessage(
                audio=pcm_bytes_to_base64(pcm),
                sample_rate=sample_rate,
                channels=channels,
                format="pcm16",
                is_last=is_last,
            ).model_dump()
        )
        return bool(pcm)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _cancel_active_turn(self) -> None:
        task, self._active_turn_task = self._active_turn_task, None
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _send_event(self, event_data: dict[str, Any]) -> None:
        """Send an event to the client; a dead socket must not kill the turn."""
        try:
            await self._send_callback(event_data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to send WebSocket event on session %s: %s", self.session_id, exc)


class _AudioOut:
    """Re-frames one TTS request's audio, sends it, and tracks client playout."""

    def __init__(
        self,
        send: Callable[..., Coroutine[Any, Any, bool]],
        playout: PlayoutClock,
        sample_rate: int,
    ):
        self._send = send
        self._playout = playout
        self._rate, self._channels = sample_rate, 1
        self._coalescer = PCM16Coalescer(sample_rate, min_ms=settings.tts_min_frame_ms)
        self.heard = False

    async def write(self, chunk: TTSAudioChunk) -> None:
        if chunk.sample_rate != self._rate or chunk.channels != self._channels:
            # Format change mid-stream: flush old-format audio first.
            await self._emit(self._coalescer.flush(), is_last=False)
            self._rate, self._channels = chunk.sample_rate, chunk.channels
            self._coalescer = PCM16Coalescer(
                self._rate, settings.tts_min_frame_ms, self._channels
            )
        frame = self._coalescer.push(chunk.audio_pcm16) if chunk.audio_pcm16 else None
        if frame:
            await self._emit(frame, is_last=False)

    async def flush(self) -> None:
        """Send whatever the coalescer is holding, mid-request."""
        await self._emit(self._coalescer.flush(), is_last=False)

    async def close(self) -> None:
        await self._emit(self._coalescer.flush(), is_last=True)

    async def _emit(self, pcm: bytes, *, is_last: bool) -> None:
        if await self._send(pcm, self._rate, self._channels, is_last=is_last):
            self.heard = True
            self._playout.sent(len(pcm) / (2 * self._channels * self._rate))

"""Keep streamed speech continuous when TTS is slower than real time.

Measured on one NVIDIA L4: Svara streams a single voice at ~0.6x real time,
but vLLM batches concurrent requests almost for free (four at once deliver
~1.9x real time in total). Playing audio as it arrives therefore runs the
client dry every second or two. Two tools fix that:

* ``SynthesisJob`` synthesizes one sentence in the background, so later
  sentences are generated while earlier ones are still playing.
* ``reply_can_play_through`` decides when the next sentence may start: only
  once it and every later sentence already synthesizing can play back to back
  without the client running dry. It uses rates ``SynthesisRateTracker``
  measured on real requests. Any unavoidable wait therefore happens before a
  sentence, never mid-word. On a backend faster than real time it never waits.
"""

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable, Sequence

from voice_agent.languages import base_language
from voice_agent.tts.base import TTSAudioChunk, TTSBase

logger = logging.getLogger(__name__)

# How often a held sentence re-checks whether it may start playing.
GUARD_POLL_SECONDS = 0.02
# Starting estimate of audio length per character, replaced by measurements.
# Svara speaks Hindi at ~0.08-0.09 s per character.
DEFAULT_SECONDS_PER_CHAR = 0.09
# Slowest synthesis rate the guard reasons about (avoids dividing by ~0).
_MIN_REALTIME_FACTOR = 0.1
# Streaming speed is only trusted over at least this much time and audio.
# Svara sends a short sentence's audio in a burst when it finishes, so a
# shorter window measures that burst, not how fast audio was generated.
_MIN_STREAM_SECONDS = 0.25
# Floor for the whole-request duration (avoids dividing by ~0).
_MIN_REQUEST_SECONDS = 0.01
_MAX_REALTIME_FACTOR = 10.0
_SMOOTHING = 0.3


def pcm_seconds(chunk: TTSAudioChunk) -> float:
    """Duration of a chunk's PCM16 audio."""
    frame_bytes = 2 * max(1, chunk.channels)
    return len(chunk.audio_pcm16) / (frame_bytes * chunk.sample_rate)


class SynthesisRateTracker:
    """Learns how fast the TTS backend produces audio.

    It measures the backend, not a user, so one instance is shared by every
    session. Speed is tracked globally; audio length per character depends on
    the script, so it is tracked per language.
    """

    def __init__(
        self,
        realtime_factor: float,
        seconds_per_char: float = DEFAULT_SECONDS_PER_CHAR,
    ):
        self._realtime_factor = realtime_factor
        self._measured_rate = False
        self._default_seconds_per_char = seconds_per_char
        self._seconds_per_char: dict[str, float] = {}

    @property
    def realtime_factor(self) -> float:
        """Audio seconds produced per wall-clock second by one request."""
        return self._realtime_factor

    def seconds_per_char(self, language: str | None) -> float:
        """Measured audio seconds per character of text in ``language``."""
        return self._seconds_per_char.get(
            base_language(language), self._default_seconds_per_char
        )

    def expected_seconds(self, text: str, language: str | None) -> float:
        """Estimated audio duration of ``text``."""
        return len(text) * self.seconds_per_char(language)

    def observe(
        self,
        text: str,
        language: str | None,
        audio_seconds: float,
        streaming_seconds: float,
        first_chunk_seconds: float,
        request_seconds: float,
    ) -> None:
        """Record one finished request.

        ``streaming_seconds`` runs from the first audio chunk to the last, so the
        request's start-up latency does not count as slow streaming; the first
        chunk's audio arrived at the start of that window and is excluded. When
        that window is too short to measure (audio arrived in a burst), the rate
        falls back to the whole request (``request_seconds``, from sending the
        request to the last chunk): bursty delivery says nothing about how fast
        the audio was generated, and the whole request errs on the slow side.
        """
        if not text or audio_seconds <= 0:
            return
        key = base_language(language)
        spc = audio_seconds / len(text)
        old_spc = self._seconds_per_char.get(key)
        self._seconds_per_char[key] = (
            spc if old_spc is None else old_spc + _SMOOTHING * (spc - old_spc)
        )

        streamed = audio_seconds - first_chunk_seconds
        if streaming_seconds >= _MIN_STREAM_SECONDS and streamed >= _MIN_STREAM_SECONDS:
            rate = streamed / streaming_seconds
        else:
            rate = audio_seconds / max(request_seconds, _MIN_REQUEST_SECONDS)
        rate = min(_MAX_REALTIME_FACTOR, rate)
        if self._measured_rate:
            self._realtime_factor += _SMOOTHING * (rate - self._realtime_factor)
        else:
            self._realtime_factor, self._measured_rate = rate, True


def safe_to_start(
    buffered_seconds: float,
    expected_seconds: float,
    queued_seconds: float,
    realtime_factor: float,
    margin: float,
    holdback_seconds: float = 0.0,
) -> bool:
    """Whether a sentence still being synthesized can start playing gap-free.

    While the remaining ``R`` seconds are generated at rate ``r``, the client
    plays ``R / r`` seconds but only ``R`` arrive, so ``R * (1/r - 1)`` seconds
    must already be queued (``queued_seconds`` from earlier sentences plus
    ``buffered_seconds`` of this one). ``margin`` scales the length estimate,
    which is the uncertain part. ``holdback_seconds`` is audio that is
    generated but held back before sending (frame coalescing), which delays
    every later piece by up to that much.
    """
    if realtime_factor >= 1.0:
        return True
    rate = max(realtime_factor, _MIN_REALTIME_FACTOR)
    remaining = max(0.0, expected_seconds * margin - buffered_seconds)
    needed = remaining * (1.0 / rate - 1.0) + holdback_seconds
    return queued_seconds + buffered_seconds >= needed


def reply_can_play_through(
    jobs: Sequence["SynthesisJob"],
    queued_seconds: float,
    tracker: SynthesisRateTracker,
    margin: float,
    holdback_seconds: float = 0.0,
) -> bool:
    """Whether ``jobs[0]`` can start now and the reply continue without a gap.

    Each job already synthesizing must pass ``safe_to_start`` with everything
    ahead of it (client queue plus earlier jobs) counted as queued. A later job
    still waiting for a synthesis slot ends the look-ahead; it is judged again
    when its own turn comes. The first job must have started.
    """
    for index, job in enumerate(jobs):
        if not job.started:
            return index > 0
        expected = tracker.expected_seconds(job.tts_text, job.language)
        if not job.finished and not safe_to_start(
            job.generated_seconds,
            expected,
            queued_seconds,
            tracker.realtime_factor,
            margin,
            holdback_seconds,
        ):
            return False
        # Audio this job adds ahead of the next one; a short guess is the safe side.
        queued_seconds += (
            job.generated_seconds
            if job.finished
            else max(job.generated_seconds, expected / margin)
        )
    return True


class PlayoutClock:
    """Estimates how much already-sent audio the client has not played yet.

    The client schedules chunks back to back, starting "now" when its queue
    has run dry. Network latency and the client's jitter lead only make the
    real queue longer, so this estimate errs on the safe side.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._end = 0.0

    def sent(self, seconds: float) -> None:
        """Record that ``seconds`` of audio were sent to the client."""
        self._end = max(self._end, self._clock()) + seconds

    def queued(self) -> float:
        """Seconds of sent audio the client has not played yet."""
        return max(0.0, self._end - self._clock())


class SynthesisJob:
    """Synthesizes one text in the background and buffers its chunks in order.

    ``slots`` limits how many jobs synthesize at once; asyncio semaphores are
    FIFO, so jobs start in the order they were created. A job holds at most one
    sentence of audio, so its queue is naturally bounded. The owner must call
    ``cancel()`` (it is safe after completion).
    """

    def __init__(
        self,
        tts: TTSBase,
        text: str,
        tts_text: str,
        voice: str | None,
        language: str | None,
        slots: asyncio.Semaphore,
        tracker: SynthesisRateTracker,
    ):
        self.text = text  # what memory records once it is heard
        self.tts_text = tts_text  # what the TTS engine reads
        self.language = language
        self.generated_seconds = 0.0
        self.started = False  # holds a synthesis slot
        self.chunks: asyncio.Queue[TTSAudioChunk | Exception | None] = asyncio.Queue()
        self._task = asyncio.create_task(self._run(tts, voice, slots, tracker))

    @property
    def finished(self) -> bool:
        """Whether synthesis has completed, failed, or been cancelled."""
        return self._task.done()

    async def cancel(self) -> None:
        """Stop the background synthesis; safe to call after it has finished."""
        if not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _run(
        self,
        tts: TTSBase,
        voice: str | None,
        slots: asyncio.Semaphore,
        tracker: SynthesisRateTracker,
    ) -> None:
        try:
            async with slots:
                self.started = True
                await self._synthesize(tts, voice, tracker)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - handed to the player, which re-raises it
            # Logged here too: if the turn ends before this job plays, the
            # player never sees the error.
            logger.warning("TTS synthesis failed for %r: %s", self.tts_text, exc)
            self.chunks.put_nowait(exc)
        finally:
            self.chunks.put_nowait(None)

    async def _synthesize(
        self, tts: TTSBase, voice: str | None, tracker: SynthesisRateTracker
    ) -> None:
        requested_at = time.monotonic()
        first_at: float | None = None
        first_seconds = 0.0
        last_at = 0.0
        async for chunk in tts.synthesize_stream(
            text=self.tts_text, voice=voice, language=self.language
        ):
            seconds = pcm_seconds(chunk)
            if seconds > 0:
                last_at = time.monotonic()
                if first_at is None:
                    first_at, first_seconds = last_at, seconds
                self.generated_seconds += seconds
            self.chunks.put_nowait(chunk)
        if first_at is not None:
            tracker.observe(
                self.tts_text,
                self.language,
                audio_seconds=self.generated_seconds,
                streaming_seconds=last_at - first_at,
                first_chunk_seconds=first_seconds,
                request_seconds=last_at - requested_at,
            )


# A typical one-sentence Hindi reply, used to measure the backend at startup.
CALIBRATION_TEXT = "नमस्ते! मैं आपकी किस तरह मदद कर सकती हूँ?"


async def calibrate(tts: TTSBase, tracker: SynthesisRateTracker, language: str = "hi") -> None:
    """Synthesize one sentence so even the first reply is paced on measured rates."""
    job = SynthesisJob(
        tts,
        text=CALIBRATION_TEXT,
        tts_text=CALIBRATION_TEXT,
        voice=None,
        language=language,
        slots=asyncio.Semaphore(1),
        tracker=tracker,
    )
    try:
        while (item := await job.chunks.get()) is not None:
            if isinstance(item, Exception):
                raise item
    finally:
        await job.cancel()

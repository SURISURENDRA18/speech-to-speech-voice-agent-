"""Unit tests for VoiceSession pipeline, conversation memory and barge-in."""

import asyncio
import base64
import time
from collections.abc import AsyncGenerator

import numpy as np
import pytest

from voice_agent.llm.base import LLMBase
from voice_agent.llm.mock_llm import MockLLMAdapter
from voice_agent.normalization.indic_norm import IndicTextNormalizer
from voice_agent.pipeline.conversation import ConversationMemory
from voice_agent.pipeline.session import SessionState, VoiceSession
from voice_agent.pipeline.synthesis import SynthesisRateTracker
from voice_agent.server.protocol import SessionStartConfig
from voice_agent.stt.mock_stt import MockSTTAdapter
from voice_agent.tts.base import TTSAudioChunk, TTSBase
from voice_agent.tts.mock_tts import MockTTSAdapter
from voice_agent.vad.detector import VADEvent


class ScriptedVAD:
    """Deterministic VAD: any non-silent frame is speech; N silent frames end it.

    Keeps pipeline tests independent of which Silero checkpoint is installed.
    """

    def __init__(self, silence_frames_to_end: int = 3):
        self._silence_to_end = silence_frames_to_end
        self.reset()

    def reset(self) -> None:
        self._active = False
        self._silent = 0
        self._buffer: list[np.ndarray] = []

    def process_chunk(self, chunk: np.ndarray) -> list[VADEvent]:
        loud = float(np.abs(chunk).max()) > 0.01
        if loud and not self._active:
            self._active, self._silent, self._buffer = True, 0, [chunk]
            return [VADEvent(event_type="speech_start", timestamp_samples=0)]
        if not self._active:
            return []
        self._buffer.append(chunk)
        self._silent = 0 if loud else self._silent + 1
        if self._silent < self._silence_to_end:
            return []
        audio = np.concatenate(self._buffer)
        self.reset()
        return [
            VADEvent(
                event_type="speech_end",
                timestamp_samples=len(audio),
                duration_samples=len(audio),
                speech_audio=audio,
            )
        ]


class RecordingLLM(LLMBase):
    """Replies with scripted text and records every prompt it receives."""

    def __init__(self, replies: list[str], token_delay: float = 0.0):
        self._replies = list(replies)
        self._delay = token_delay
        self.calls: list[dict] = []

    @property
    def model_name(self) -> str:
        return "recording-llm"

    async def generate_stream(
        self, prompt, history=None, system_prompt=None
    ) -> AsyncGenerator[str, None]:
        self.calls.append(
            {"prompt": prompt, "history": list(history or []), "system_prompt": system_prompt}
        )
        for word in self._replies.pop(0).split(" "):
            if self._delay:
                await asyncio.sleep(self._delay)
            yield word + " "


class TinyChunkTTS(TTSBase):
    """Emits audio in 5 ms network-sized pieces, like a remote HTTP stream."""

    def __init__(self, pieces: int = 40):
        self.pieces = pieces
        self.texts: list[str] = []

    @property
    def sample_rate(self) -> int:
        return 24000

    @property
    def model_name(self) -> str:
        return "tiny-chunk-tts"

    async def synthesize_stream(self, text, voice=None, language=None):
        self.texts.append(text)
        for _ in range(self.pieces):
            yield TTSAudioChunk(audio_pcm16=b"\x01\x00" * 120, sample_rate=24000)
        yield TTSAudioChunk(audio_pcm16=b"", sample_rate=24000, is_last=True)


def make_session(stt=None, llm=None, tts=None, vad=None):
    events: list[dict] = []

    async def sender(event: dict) -> None:
        events.append(event)

    session = VoiceSession(
        session_id="test-session",
        vad=vad or ScriptedVAD(),
        stt=stt or MockSTTAdapter(mock_response="नमस्ते", latency_seconds=0.0),
        llm=llm or MockLLMAdapter(canned_response="नमस्ते!", token_delay_seconds=0.0),
        normalizer=IndicTextNormalizer(default_lang="hi", use_wfst=False),
        tts=tts or MockTTSAdapter(sample_rate=24000, chunk_duration_ms=20),
        send_callback=sender,
    )
    return session, events


@pytest.mark.asyncio
async def test_session_lifecycle_and_turn():
    session, events = make_session(
        stt=MockSTTAdapter(mock_response="नमस्ते, आज का दिन कैसा है?", latency_seconds=0.01),
        llm=MockLLMAdapter(canned_response="बहुत अच्छा है! धन्यवाद।", token_delay_seconds=0.01),
    )
    await session.start(SessionStartConfig(language="hi", voice="hi_female"))
    assert session.state == SessionState.LISTENING

    t = np.linspace(0, 0.4, 6400, endpoint=False, dtype=np.float32)
    speech_pcm = (0.5 * np.sin(2 * np.pi * 300 * t) * 32767).astype(np.int16).tobytes()
    silence_pcm = np.zeros(3200, dtype=np.int16).tobytes()

    await session.push_audio(speech_pcm)
    for _ in range(3):
        await session.push_audio(silence_pcm)
    for _ in range(50):
        await asyncio.sleep(0.02)
        if any(e["type"] == "turn_complete" for e in events):
            break

    event_types = [e["type"] for e in events]
    for expected in (
        "vad_speech_start",
        "vad_speech_end",
        "transcript",
        "llm_chunk",
        "tts_chunk",
        "turn_complete",
    ):
        assert expected in event_types
    assert len(session.memory) == 1

    await session.stop()
    assert session.state == SessionState.CLOSED


@pytest.mark.asyncio
async def test_follow_up_turn_sees_history_and_reply_language():
    llm = RecordingLLM(["नमस्ते रवि जी! आप कैसे हैं?", "आपका नाम रवि है।"])
    stt = MockSTTAdapter(mock_response="मेरा नाम रवि है", latency_seconds=0.0)
    session, _ = make_session(stt=stt, llm=llm)
    await session.start(SessionStartConfig(language="hi"))

    await session._run_turn(np.ones(1600, dtype=np.float32))
    stt.mock_response = "मेरा नाम क्या है?"
    await session._run_turn(np.ones(1600, dtype=np.float32))

    second = llm.calls[1]
    assert second["prompt"] == "मेरा नाम क्या है?"
    assert second["history"] == [
        {"role": "user", "content": "मेरा नाम रवि है"},
        {"role": "assistant", "content": "नमस्ते रवि जी! आप कैसे हैं?"},
    ]
    assert "Reply only in Hindi" in second["system_prompt"]
    await session.stop()


@pytest.mark.asyncio
async def test_truncated_trailing_fragment_is_not_spoken_or_remembered():
    llm = RecordingLLM(["पहला वाक्य पूरा है। दूसरा अधूरा"])
    tts = TinyChunkTTS(pieces=1)
    session, _ = make_session(llm=llm, tts=tts)
    await session.start()

    await session._run_turn(np.ones(1600, dtype=np.float32))

    assert tts.texts == ["पहला वाक्य पूरा है।"]
    assert session.memory.turns[0].assistant == "पहला वाक्य पूरा है।"
    await session.stop()


@pytest.mark.asyncio
async def test_tiny_tts_pieces_are_coalesced_into_larger_frames():
    tts = TinyChunkTTS(pieces=40)  # 40 x 5 ms = 200 ms of audio
    session, events = make_session(tts=tts)
    await session.start()

    await session._run_turn(np.ones(1600, dtype=np.float32))

    frames = [e for e in events if e["type"] == "tts_chunk"]
    total = sum(len(base64.b64decode(f["audio"])) for f in frames)
    assert total == 40 * 240
    assert len(frames) <= 3  # instead of 41 tiny messages
    assert frames[-1]["is_last"] is True
    await session.stop()


def test_injected_empty_memory_is_used_not_replaced():
    memory = ConversationMemory(max_turns=2)
    session = VoiceSession(
        session_id="mem",
        vad=ScriptedVAD(),
        stt=MockSTTAdapter(),
        llm=MockLLMAdapter(),
        normalizer=IndicTextNormalizer(default_lang="hi", use_wfst=False),
        tts=MockTTSAdapter(),
        send_callback=lambda _event: asyncio.sleep(0),
        memory=memory,
    )
    assert session.memory is memory


@pytest.mark.asyncio
async def test_session_barge_in_interruption():
    llm = MockLLMAdapter(canned_response="A very long response " * 10, token_delay_seconds=0.1)
    session, events = make_session(
        stt=MockSTTAdapter(mock_response="Test prompt", latency_seconds=0.01), llm=llm
    )
    await session.start()

    session._active_turn_task = asyncio.create_task(
        session._run_turn(np.ones(16000, dtype=np.float32))
    )
    await asyncio.sleep(0.05)
    assert session.state in (SessionState.PROCESSING, SessionState.SPEAKING)

    await session.interrupt(reason="user_speech")

    assert session.state == SessionState.LISTENING
    interrupted = [e for e in events if e["type"] == "interrupted"]
    assert len(interrupted) == 1
    assert interrupted[0]["reason"] == "user_speech"
    # Nothing was spoken yet, so the user's words carry into the next turn.
    assert session.memory.compose_user_message("again") == "Test prompt again"
    await session.stop()


class StampedTTS(TTSBase):
    """Each request's audio carries its own sample value; listed texts are slow."""

    def __init__(self, slow_texts: set[str], pieces: int = 4, slow_delay: float = 0.05):
        self.slow_texts = slow_texts
        self.pieces = pieces
        self.slow_delay = slow_delay
        self.log: list[tuple[str, str]] = []  # (event, text)
        self.values: dict[str, int] = {}

    @property
    def sample_rate(self) -> int:
        return 24000

    @property
    def model_name(self) -> str:
        return "stamped-tts"

    async def synthesize_stream(self, text, voice=None, language=None):
        value = self.values.setdefault(text, len(self.values) + 1)
        self.log.append(("start", text))
        for _ in range(self.pieces):
            await asyncio.sleep(self.slow_delay if text in self.slow_texts else 0)
            yield TTSAudioChunk(
                audio_pcm16=np.full(2400, value, dtype=np.int16).tobytes(), sample_rate=24000
            )
        self.log.append(("end", text))
        yield TTSAudioChunk(audio_pcm16=b"", sample_rate=24000, is_last=True)


@pytest.mark.asyncio
async def test_sentences_synthesize_in_parallel_but_play_in_order():
    llm = RecordingLLM(["पहला वाक्य। दूसरा वाक्य।"])
    tts = StampedTTS(slow_texts={"पहला वाक्य।"})
    session, events = make_session(llm=llm, tts=tts)
    session.tts_rate = SynthesisRateTracker(realtime_factor=2.0)  # never hold back
    await session.start()

    await session._run_turn(np.ones(1600, dtype=np.float32))

    # The second sentence started before the slow first one finished...
    assert tts.log.index(("start", "दूसरा वाक्य।")) < tts.log.index(("end", "पहला वाक्य।"))
    # ...but the client still hears the first sentence completely, then the second.
    audio = np.concatenate(
        [
            np.frombuffer(base64.b64decode(e["audio"]), dtype=np.int16)
            for e in events
            if e["type"] == "tts_chunk"
        ]
    )
    first, second = tts.values["पहला वाक्य।"], tts.values["दूसरा वाक्य।"]
    assert list(dict.fromkeys(audio.tolist())) == [first, second]
    assert sum(1 for e in events if e["type"] == "tts_chunk" and e["is_last"]) == 2
    assert session.memory.turns[0].assistant == "पहला वाक्य। दूसरा वाक्य।"
    await session.stop()


class HalfRealtimeTTS(TTSBase):
    """Streams 40 ms of audio every 80 ms: 0.5x real time, like Svara on an L4."""

    def __init__(self, pieces: int = 10):
        self.pieces = pieces

    @property
    def sample_rate(self) -> int:
        return 24000

    @property
    def model_name(self) -> str:
        return "half-realtime-tts"

    async def synthesize_stream(self, text, voice=None, language=None):
        for _ in range(self.pieces):
            await asyncio.sleep(0.08)
            yield TTSAudioChunk(audio_pcm16=b"\x01\x00" * 960, sample_rate=24000)
        yield TTSAudioChunk(audio_pcm16=b"", sample_rate=24000, is_last=True)


@pytest.mark.asyncio
async def test_slow_tts_is_held_back_until_it_can_play_without_gaps():
    text = "यह एक वाक्य है।"
    tts = HalfRealtimeTTS(pieces=10)  # 0.4 s of audio over 0.8 s
    sent: list[tuple[float, float]] = []  # (send time, audio seconds)

    async def sender(event: dict) -> None:
        if event["type"] == "tts_chunk" and event["audio"]:
            sent.append((time.monotonic(), len(base64.b64decode(event["audio"])) / 48000))

    normalizer = IndicTextNormalizer(default_lang="hi", use_wfst=False)
    session = VoiceSession(
        session_id="slow",
        vad=ScriptedVAD(),
        stt=MockSTTAdapter(mock_response="नमस्ते", latency_seconds=0.0),
        llm=RecordingLLM([text]),
        normalizer=normalizer,
        tts=tts,
        send_callback=sender,
        tts_rate=SynthesisRateTracker(
            realtime_factor=0.5, seconds_per_char=0.4 / len(normalizer.normalize(text, lang="hi"))
        ),
    )
    await session.start(SessionStartConfig(language="hi"))

    await session._run_turn(np.ones(1600, dtype=np.float32))

    # Replay the client's back-to-back scheduling: after playback starts it
    # must never wait for audio (a few ms of event-loop jitter allowed).
    assert sum(seconds for _, seconds in sent) == pytest.approx(0.4)
    play_end = sent[0][0] + sent[0][1]
    for arrived, seconds in sent[1:]:
        assert arrived <= play_end + 0.03, f"underrun of {arrived - play_end:.3f}s"
        play_end = max(play_end, arrived) + seconds
    await session.stop()

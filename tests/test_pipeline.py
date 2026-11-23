"""Unit tests for VoiceSession pipeline and interruption handling."""

import asyncio

import numpy as np
import pytest

from voice_agent.llm.mock_llm import MockLLMAdapter
from voice_agent.normalization.indic_norm import IndicTextNormalizer
from voice_agent.pipeline.session import SessionState, VoiceSession
from voice_agent.server.protocol import SessionStartConfig
from voice_agent.stt.mock_stt import MockSTTAdapter
from voice_agent.tts.mock_tts import MockTTSAdapter
from voice_agent.vad.detector import SileroVADDetector


@pytest.mark.asyncio
async def test_session_lifecycle_and_turn():
    events_received: list[dict] = []

    async def mock_sender(event: dict):
        events_received.append(event)

    vad = SileroVADDetector(threshold=0.3, min_silence_duration_ms=50)
    stt = MockSTTAdapter(mock_response="नमस्ते, आज का दिन कैसा है?", latency_seconds=0.01)
    llm = MockLLMAdapter(canned_response="बहुत अच्छा है! धन्यवाद।", token_delay_seconds=0.01)
    normalizer = IndicTextNormalizer(default_lang="hi", use_wfst=False)
    tts = MockTTSAdapter(sample_rate=24000, chunk_duration_ms=20)

    session = VoiceSession(
        session_id="test-session-1",
        vad=vad,
        stt=stt,
        llm=llm,
        normalizer=normalizer,
        tts=tts,
        send_callback=mock_sender,
    )

    await session.start(SessionStartConfig(language="hi", voice="hi_female"))
    assert session.state == SessionState.LISTENING

    # Feed synthetic speech (300Hz sine wave)
    t = np.linspace(0, 0.4, 6400, endpoint=False, dtype=np.float32)
    speech_pcm = (0.5 * np.sin(2 * np.pi * 300 * t) * 32767).astype(np.int16).tobytes()
    silence_pcm = np.zeros(3200, dtype=np.int16).tobytes()

    await session.push_audio(speech_pcm)
    # Wait briefly for VAD to detect speech
    await asyncio.sleep(0.1)

    # Push trailing silence to trigger speech end
    for _ in range(5):
        await session.push_audio(silence_pcm)
        await asyncio.sleep(0.05)

    # Wait for turn pipeline to execute
    await asyncio.sleep(0.5)

    event_types = [e["type"] for e in events_received]
    assert "vad_speech_start" in event_types
    assert "vad_speech_end" in event_types
    assert "transcript" in event_types
    assert "llm_chunk" in event_types
    assert "tts_chunk" in event_types
    assert "turn_complete" in event_types

    await session.stop()
    assert session.state == SessionState.CLOSED


@pytest.mark.asyncio
async def test_session_barge_in_interruption():
    events_received: list[dict] = []

    async def mock_sender(event: dict):
        events_received.append(event)

    vad = SileroVADDetector()
    stt = MockSTTAdapter(mock_response="Test prompt", latency_seconds=0.01)
    # Slow LLM to allow testing barge-in mid-generation
    llm = MockLLMAdapter(canned_response="A very long response " * 10, token_delay_seconds=0.1)
    normalizer = IndicTextNormalizer(default_lang="hi", use_wfst=False)
    tts = MockTTSAdapter(sample_rate=24000)

    session = VoiceSession(
        session_id="test-session-interrupt",
        vad=vad,
        stt=stt,
        llm=llm,
        normalizer=normalizer,
        tts=tts,
        send_callback=mock_sender,
    )

    await session.start()

    # Manually trigger a turn
    speech_audio = np.ones(16000, dtype=np.float32)
    turn_task = asyncio.create_task(session._run_turn(speech_audio))
    session._active_turn_task = turn_task

    # Wait until processing starts
    await asyncio.sleep(0.05)
    assert session.state in (SessionState.PROCESSING, SessionState.SPEAKING)

    # Trigger interruption (barge-in)
    await session.interrupt(reason="user_speech")

    assert session.state == SessionState.LISTENING
    interrupted_events = [e for e in events_received if e["type"] == "interrupted"]
    assert len(interrupted_events) == 1
    assert interrupted_events[0]["reason"] == "user_speech"

    await session.stop()

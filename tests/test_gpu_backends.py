"""Tests for the GCP/GPU code paths: remote LLM (vLLM), remote Svara, LLM->TTS overlap."""

import asyncio
import json

import httpx
import numpy as np
import pytest

from voice_agent.llm.mock_llm import MockLLMAdapter
from voice_agent.llm.openai_llm import OpenAICompatibleLLMAdapter
from voice_agent.normalization.indic_norm import IndicTextNormalizer
from voice_agent.pipeline.session import VoiceSession
from voice_agent.server.protocol import SessionStartConfig
from voice_agent.stt.mock_stt import MockSTTAdapter
from voice_agent.tts.mock_tts import MockTTSAdapter
from voice_agent.tts.svara_tts import SvaraTTSAdapter
from voice_agent.vad.detector import SileroVADDetector


class _ChunkedStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


@pytest.mark.asyncio
async def test_openai_llm_streams_sse_deltas_without_thinking():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        events = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "नमस्ते"}}]},
            {"choices": [{"delta": {"content": "!"}}]},
        ]
        body = [f"data: {json.dumps(e, ensure_ascii=False)}\n\n".encode() for e in events]
        body.append(b"data: [DONE]\n\n")
        return httpx.Response(200, stream=_ChunkedStream(body))

    llm = OpenAICompatibleLLMAdapter(base_url="http://qwen:8001/v1", max_new_tokens=32)
    llm._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tokens = [t async for t in llm.generate_stream("hi", history=[], system_prompt="sys")]

    assert tokens == ["नमस्ते", "!"]
    assert seen["stream"] is True
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen["max_tokens"] == 32
    assert seen["messages"][0] == {"role": "system", "content": "sys"}


@pytest.mark.asyncio
async def test_remote_svara_requests_streaming_and_keeps_samples_aligned():
    pcm = (np.arange(3000, dtype=np.int16) * 7).tobytes()
    # Odd-sized network chunks split 16-bit samples across reads.
    pieces = [pcm[i : i + 1001] for i in range(0, len(pcm), 1001)]
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200, headers={"X-Sample-Rate": "24000"}, stream=_ChunkedStream(pieces)
        )

    tts = SvaraTTSAdapter(api_url="http://svara:8080", output_sample_rate=24000)
    tts._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    chunks = [c async for c in tts.synthesize_stream("नमस्ते", voice="hi_female")]

    assert seen["stream"] is True and seen["response_format"] == "pcm"
    assert all(len(c.audio_pcm16) % 2 == 0 for c in chunks)
    assert b"".join(c.audio_pcm16 for c in chunks) == pcm
    assert chunks[-1].is_last and not any(c.is_last for c in chunks[:-1])


@pytest.mark.asyncio
async def test_session_overlaps_tts_with_llm_and_skips_truncated_fragment():
    events: list[dict] = []

    async def sender(event: dict) -> None:
        events.append(event)

    llm = MockLLMAdapter(
        canned_response="पहला वाक्य। दूसरा वाक्य! अधूरा टुकड़ा", token_delay_seconds=0.02
    )
    session = VoiceSession(
        session_id="overlap",
        vad=SileroVADDetector(),
        stt=MockSTTAdapter(mock_response="नमस्ते", latency_seconds=0.0),
        llm=llm,
        normalizer=IndicTextNormalizer(default_lang="hi", use_wfst=False),
        tts=MockTTSAdapter(sample_rate=24000, chunk_duration_ms=20),
        send_callback=sender,
    )
    await session.start(SessionStartConfig(language="hi", voice="hi_female"))
    await session._run_turn(np.zeros(16000, dtype=np.float32))
    await session.stop()

    types = [e["type"] for e in events]
    assert "turn_complete" in types
    first_tts = types.index("tts_chunk")
    last_llm = len(types) - 1 - types[::-1].index("llm_chunk")
    # Audio for sentence 1 starts before the LLM has finished the reply.
    assert first_tts < last_llm
    # Two complete sentences are spoken; the cut-off fragment is not.
    assert sum(1 for e in events if e["type"] == "tts_chunk" and e["is_last"]) == 2


@pytest.mark.asyncio
async def test_session_interrupt_cancels_overlapped_turn_cleanly():
    events: list[dict] = []

    async def sender(event: dict) -> None:
        events.append(event)

    session = VoiceSession(
        session_id="interrupt",
        vad=SileroVADDetector(),
        stt=MockSTTAdapter(mock_response="नमस्ते", latency_seconds=0.0),
        llm=MockLLMAdapter(canned_response="एक। दो। तीन।", token_delay_seconds=0.05),
        normalizer=IndicTextNormalizer(default_lang="hi", use_wfst=False),
        tts=MockTTSAdapter(sample_rate=24000, chunk_duration_ms=20),
        send_callback=sender,
    )
    await session.start(SessionStartConfig(language="hi", voice="hi_female"))
    session._active_turn_task = asyncio.create_task(
        session._run_turn(np.zeros(16000, dtype=np.float32))
    )
    await asyncio.sleep(0.15)
    await session.interrupt(reason="client_interrupt")
    await asyncio.sleep(0.2)
    await session.stop()

    types = [e["type"] for e in events]
    assert "interrupted" in types
    assert "turn_complete" not in types
    assert "error" not in types

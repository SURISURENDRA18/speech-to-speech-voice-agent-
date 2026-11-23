"""Unit tests for Text-to-Speech (TTS) adapters."""

import pytest

from voice_agent.tts.mock_tts import MockTTSAdapter
from voice_agent.tts.svara_tts import SvaraTTSAdapter, load_svara_voices


@pytest.mark.asyncio
async def test_mock_tts_stream():
    tts = MockTTSAdapter(sample_rate=24000, chunk_duration_ms=50)
    chunks = []
    async for chunk in tts.synthesize_stream("नमस्ते"):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0].sample_rate == 24000
    assert chunks[0].is_last is False
    assert chunks[1].is_last is True
    assert len(chunks[0].audio_pcm16) > 0


def test_load_svara_voices():
    voices = load_svara_voices()
    assert len(voices) > 0
    voice_ids = [v.get("voice_id") for v in voices]
    assert "hi_female" in voice_ids
    assert "hi_male" in voice_ids


def test_mlx_voice_mapping_uses_requested_language_and_gender():
    assert SvaraTTSAdapter._mlx_voice_name("hi_female", "hi-IN") == "Hindi (Female)"
    assert SvaraTTSAdapter._mlx_voice_name("ta_male", "ta") == "Tamil (Male)"
    assert SvaraTTSAdapter._mlx_voice_name("hi", None) == "Hindi (Female)"


@pytest.mark.asyncio
async def test_svara_tts_local_stream(monkeypatch):
    # Exercise the macOS ``say`` path even on Macs that have mlx-audio installed.
    monkeypatch.setattr(SvaraTTSAdapter, "_mlx_available", property(lambda _self: False))
    tts = SvaraTTSAdapter(default_voice="hi_female", output_sample_rate=24000)
    # A 200 ms non-empty PCM utterance, standing in for macOS ``say`` output.
    pcm = b"\x10\x00" * (24000 // 5)
    monkeypatch.setattr(tts, "_synthesize_with_macos_say", lambda *_: pcm)
    chunks = []
    async for chunk in tts.synthesize_stream("नमस्ते दुनिया"):
        chunks.append(chunk)

    # Local PCM is split into consecutive 100 ms playback frames.
    assert len(chunks) == 2
    assert chunks[0].sample_rate == 24000
    assert chunks[0].is_last is False
    assert chunks[1].is_last is True
    assert all(chunk.audio_pcm16 for chunk in chunks)
    assert b"".join(chunk.audio_pcm16 for chunk in chunks) == pcm

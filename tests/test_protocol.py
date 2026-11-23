"""Unit tests for WebSocket protocol schemas."""

from voice_agent.server.protocol import (
    ClientAudioMessage,
    ClientInterruptMessage,
    ClientStartMessage,
    ClientStopMessage,
    ServerConnectedMessage,
    ServerErrorMessage,
    ServerInterruptedMessage,
    ServerLLMChunkMessage,
    ServerStartedMessage,
    ServerStoppedMessage,
    ServerTranscriptMessage,
    ServerTTSChunkMessage,
    ServerTurnCompleteMessage,
    ServerVADSpeechEndMessage,
    ServerVADSpeechStartMessage,
    SessionStartConfig,
)


def test_client_messages():
    # Start message
    msg = ClientStartMessage(
        config=SessionStartConfig(language="hi", voice="hi_female")
    )
    dumped = msg.model_dump()
    assert dumped["type"] == "start"
    assert dumped["config"]["language"] == "hi"

    # Audio message
    audio_msg = ClientAudioMessage(data="AAAA")
    assert audio_msg.model_dump()["type"] == "audio"

    # Interrupt message
    assert ClientInterruptMessage().model_dump()["type"] == "interrupt"

    # Stop message
    assert ClientStopMessage().model_dump()["type"] == "stop"


def test_server_messages():
    # Connected
    conn = ServerConnectedMessage(session_id="test-123")
    assert conn.type == "connected"
    assert conn.session_id == "test-123"

    # Started
    started = ServerStartedMessage(session_id="test-123", config={"sample_rate": 16000})
    assert started.type == "session_started"

    # VAD events
    start_vad = ServerVADSpeechStartMessage()
    assert start_vad.type == "vad_speech_start"

    end_vad = ServerVADSpeechEndMessage(duration_ms=1200)
    assert end_vad.type == "vad_speech_end"
    assert end_vad.duration_ms == 1200

    # Transcript
    trans = ServerTranscriptMessage(text="नमस्ते", language="hi")
    assert trans.text == "नमस्ते"

    # LLM Chunk
    llm = ServerLLMChunkMessage(delta="हाँ ", accumulated="हाँ ")
    assert llm.delta == "हाँ "

    # TTS Chunk
    tts = ServerTTSChunkMessage(audio="AQID", sample_rate=24000, is_last=True)
    assert tts.is_last is True

    # Interrupted
    interrupted = ServerInterruptedMessage()
    assert interrupted.reason == "user_speech"

    # Turn complete
    complete = ServerTurnCompleteMessage(user_text="User", assistant_text="Bot")
    assert complete.user_text == "User"

    # Error
    err = ServerErrorMessage(code="ERR_TEST", message="Test error")
    assert err.code == "ERR_TEST"

    # Stopped
    stopped = ServerStoppedMessage()
    assert stopped.type == "session_stopped"

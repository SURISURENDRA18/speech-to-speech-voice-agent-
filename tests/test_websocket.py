"""Integration tests for FastAPI REST endpoints and WebSocket /ws/voice."""

import pytest
from fastapi.testclient import TestClient

from voice_agent.config import settings
from voice_agent.server.app import create_app


@pytest.fixture
def client():
    # Configure mock adapters for fast integration testing
    settings.use_mock_stt = True
    settings.use_mock_llm = True
    settings.use_mock_tts = True

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_rest_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "vad" in data["components"]
    assert "stt" in data["components"]
    assert "llm" in data["components"]
    assert "tts" in data["components"]


def test_rest_voices(client):
    response = client.get("/v1/voices")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] > 0
    voices = data["voices"]
    assert any(v["voice_id"] == "hi_female" for v in voices)


def test_rest_info(client):
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "accelerator" in data
    assert "models" in data
    assert "memory_optimization" in data
    assert data["models"]["stt"] == "Whisper large-v3-turbo (809M parameters)"


def test_websocket_full_session(client):
    with client.websocket_connect("/ws/voice") as ws:
        # 1. Check connected message
        connected = ws.receive_json()
        assert connected["type"] == "connected"
        assert "session_id" in connected

        # 2. Send start configuration
        ws.send_json({
            "type": "start",
            "config": {
                "sample_rate": 16000,
                "language": "hi",
                "voice": "hi_female",
            },
        })
        started = ws.receive_json()
        assert started["type"] == "session_started"

        # 3. Test explicit manual interruption
        ws.send_json({"type": "interrupt"})

        # 4. Send stop message
        ws.send_json({"type": "stop"})
        stopped = ws.receive_json()
        assert stopped["type"] == "session_stopped"

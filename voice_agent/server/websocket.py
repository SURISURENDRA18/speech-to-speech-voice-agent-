"""WebSocket endpoint handler for full-duplex speech-to-speech communication."""

import json
import logging
import uuid

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from voice_agent.audio import base64_to_pcm_bytes
from voice_agent.pipeline.session import VoiceSession
from voice_agent.server.protocol import (
    ClientAudioMessage,
    ClientStartMessage,
    ServerConnectedMessage,
    ServerErrorMessage,
    ServerStartedMessage,
    ServerStoppedMessage,
)

logger = logging.getLogger(__name__)


async def voice_websocket_handler(websocket: WebSocket) -> None:
    """Full-duplex WebSocket handler for /ws/voice."""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info("New WebSocket connection accepted (session %s)", session_id)

    # Send initial connection acknowledgment
    await websocket.send_json(ServerConnectedMessage(session_id=session_id).model_dump())

    # Obtain model instances from application state
    app_state = websocket.app.state
    vad = app_state.vad
    stt = app_state.stt
    llm = app_state.llm
    normalizer = app_state.normalizer
    tts = app_state.tts

    async def _send_event_to_client(event: dict) -> None:
        try:
            await websocket.send_json(event)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to send json to client: %s", exc)

    session = VoiceSession(
        session_id=session_id,
        vad=vad,
        stt=stt,
        llm=llm,
        normalizer=normalizer,
        tts=tts,
        send_callback=_send_event_to_client,
    )

    try:
        while True:
            message = await websocket.receive()
            msg_type = message.get("type")

            if msg_type == "websocket.disconnect":
                break

            # Handle binary audio frame
            if message.get("bytes"):
                pcm_bytes = message["bytes"]
                await session.push_audio(pcm_bytes)
                continue

            # Handle JSON text frame
            if message.get("text"):
                raw_text = message["text"].strip()
                if not raw_text:
                    continue

                try:
                    payload = json.loads(raw_text)
                except json.JSONDecodeError:
                    await websocket.send_json(
                        ServerErrorMessage(
                            code="JSON_DECODE_ERROR",
                            message="Invalid JSON message received.",
                        ).model_dump()
                    )
                    continue

                event_type = payload.get("type")

                if event_type == "start":
                    try:
                        start_msg = ClientStartMessage.model_validate(payload)
                        await session.start(start_msg.config)
                        await websocket.send_json(
                            ServerStartedMessage(
                                session_id=session_id,
                                config=start_msg.config.model_dump(),
                            ).model_dump()
                        )
                    except ValidationError as e:
                        await websocket.send_json(
                            ServerErrorMessage(
                                code="VALIDATION_ERROR",
                                message=str(e),
                            ).model_dump()
                        )

                elif event_type == "audio":
                    try:
                        audio_msg = ClientAudioMessage.model_validate(payload)
                        pcm_bytes = base64_to_pcm_bytes(audio_msg.data)
                        await session.push_audio(pcm_bytes)
                    except Exception as e:  # noqa: BLE001
                        await websocket.send_json(
                            ServerErrorMessage(
                                code="AUDIO_DECODE_ERROR",
                                message=f"Failed to decode base64 audio: {e}",
                            ).model_dump()
                        )

                elif event_type == "interrupt":
                    await session.interrupt(reason="client_interrupt")

                elif event_type == "stop":
                    await session.stop()
                    await websocket.send_json(ServerStoppedMessage().model_dump())

                else:
                    await websocket.send_json(
                        ServerErrorMessage(
                            code="UNKNOWN_MESSAGE_TYPE",
                            message=f"Unrecognized message type '{event_type}'.",
                        ).model_dump()
                    )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected normally for session %s", session_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("Unhandled exception in WebSocket handler for session %s: %s", session_id, exc)
    finally:
        await session.stop()
        logger.info("Cleaned up session %s on disconnect", session_id)

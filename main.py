"""Main application entry point for the Speech-to-Speech Voice Agent."""

import argparse
import logging

import uvicorn

from voice_agent.config import settings


def main() -> None:
    """Parse CLI arguments and run the Uvicorn server."""
    parser = argparse.ArgumentParser(
        description="Run the Speech-to-Speech Voice Agent on Apple Silicon Mac."
    )
    parser.add_argument("--host", default=settings.host, help="Bind host")
    parser.add_argument("--port", type=int, default=settings.port, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable reload")
    parser.add_argument("--mock-stt", action="store_true", help="Use mock STT adapter")
    parser.add_argument("--mock-llm", action="store_true", help="Use mock LLM adapter")
    parser.add_argument("--mock-tts", action="store_true", help="Use mock TTS adapter")
    parser.add_argument(
        "--mock-all", action="store_true", help="Use mock adapters for STT, LLM, and TTS"
    )

    args = parser.parse_args()

    if args.mock_all:
        settings.use_mock_stt = True
        settings.use_mock_llm = True
        settings.use_mock_tts = True
    else:
        if args.mock_stt:
            settings.use_mock_stt = True
        if args.mock_llm:
            settings.use_mock_llm = True
        if args.mock_tts:
            settings.use_mock_tts = True

    settings.host = args.host
    settings.port = args.port

    # Show the app's own logs (model preload, TTS fallbacks) next to uvicorn's.
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "urllib3", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    print("=" * 70)
    print(" Speech-to-Speech Voice Agent")
    print(f" Binding to: http://{args.host}:{args.port}")
    print(f" WebSocket endpoint: ws://{args.host}:{args.port}/ws/voice")
    print(f" Voices endpoint: http://{args.host}:{args.port}/v1/voices")
    print(f" Health endpoint: http://{args.host}:{args.port}/health")
    print(f" Info endpoint: http://{args.host}:{args.port}/info")
    print("=" * 70)

    uvicorn.run(
        "voice_agent.server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()

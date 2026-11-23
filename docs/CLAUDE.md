# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository. Read this first, then follow [CODE.md](CODE.md) for coding standards.

## What this project is

A real-time, full-duplex **speech-to-speech voice agent for Indian languages**. One FastAPI WebSocket (`/ws/voice`) runs a streaming pipeline:

```text
client PCM16 ─► Silero VAD ─► Whisper STT ─► Qwen3 1.7B LLM ─► SentenceSegmenter
            ─► Indic normalizer ─► Svara TTS ─► PCM16Coalescer ─► client playback
```

It runs on an 8 GB Apple Silicon Mac (MLX) or on one NVIDIA L4 (vLLM + faster-whisper). Barge-in cancels the LLM and TTS mid-reply.

## Commands

```bash
source .venv/bin/activate

python main.py                       # real models (first run downloads weights)
python main.py --mock-all            # no models: canned STT/LLM, tone TTS
python main.py --mock-stt --mock-llm # canned text, real spoken audio
python terminal_client.py            # mic client; --wav file.wav to send a file

python -m ruff check voice_agent/ tests/                              # lint (CI gate)
USE_MLX=false python -m pytest tests/ -q --ignore=tests/test_real_smoke.py   # CI test set
python -m pytest tests/test_pipeline.py -q                            # one file
```

Tests that need downloaded assets: `test_vad.py`, `test_tts.py::test_load_svara_voices` and `test_websocket.py::test_rest_voices` need `silero-vad-master/…/silero_vad.jit` and `svara-tts-inference-main/assets/voices/svara-tts-v1.yaml` under `WORKSPACE_DIR` (CI downloads them; see `.github/workflows/ci.yml`). `test_real_smoke.py` needs cached Qwen3 weights on Apple Silicon. Everything else runs offline.

## Layout

| Path | Responsibility |
| :-- | :-- |
| `main.py` | CLI entry: flags → `settings`, starts uvicorn |
| `voice_agent/config.py` | `Settings` (pydantic-settings, `.env`). **Every tunable lives here** |
| `voice_agent/languages.py` | Language codes/names. Single source of truth |
| `voice_agent/audio.py` | Pure PCM helpers: convert, resample, frame, `PCM16Coalescer` |
| `voice_agent/protocol.py` | Pydantic WebSocket message schemas (`server/protocol.py` re-exports) |
| `voice_agent/pipeline/session.py` | `VoiceSession`: per-connection state machine and turn orchestration |
| `voice_agent/pipeline/conversation.py` | `ConversationMemory`, `build_system_prompt` |
| `voice_agent/pipeline/segmenter.py` | `SentenceSegmenter`, `clean_for_speech`, `group_for_synthesis` |
| `voice_agent/pipeline/synthesis.py` | `SynthesisJob` (background TTS per sentence), underrun guard, `SynthesisRateTracker` |
| `voice_agent/{vad,stt,llm,tts,normalization}/` | Adapters. Each has a `base.py` ABC plus MLX / PyTorch / remote / mock implementations |
| `voice_agent/server/app.py` | Lifespan: picks adapters from settings and stores them on `app.state` |
| `voice_agent/server/websocket.py` | `/ws/voice` handler: decodes messages, owns one `VoiceSession` |
| `voice_agent/server/static/index.html` | Browser client: mic capture, jitter-buffered playback |
| `deploy/gcp/` | Docker images, compose (svara + qwen + agent + caddy), VM scripts |
| `docs/` | `ARCHITECTURE.md`, `PROTOCOL.md`, setup and deploy guides |

## Architecture rules (do not break these)

1. **Adapters behind ABCs.** The pipeline depends only on `STTBase`, `LLMBase` and `TTSBase`. To add a backend, write a new adapter class and select it in `server/app.py` `lifespan`. Never add `if backend == ...` inside `session.py`.
2. **Shared vs per-session state.** STT, LLM, TTS and the normalizer are loaded once and shared by all sessions. Anything with per-stream state is created per session: the VAD (via `app.state.vad_factory`), `ConversationMemory`, and the audio queues. Do not put per-user state on a shared adapter.
3. **Blocking inference never runs on the event loop.** Use `asyncio.to_thread` or a worker thread that bridges through an `asyncio.Queue`, as in `mlx_qwen_llm.py` and `svara_tts.py`. Guard shared models with the adapter's `asyncio.Lock`.
4. **Cancellation is the barge-in mechanism.** Turn code must let `asyncio.CancelledError` propagate. Clean up in `finally` blocks (stop worker threads, release models). Never use bare `except:` or `except BaseException`, because they swallow cancellation.
5. **Memory records what was spoken.** `ConversationMemory.record_turn` gets the text whose audio reached the client, not the raw LLM output.
6. **8 GB Mac memory budget.** `release_model()` and `swaps_weights` exist so PyTorch models can be swapped per turn. Do not keep extra models resident on the MPS path. LLM/TTS overlap is disabled when it would put two large models on the GPU at once (`VoiceSession._overlap_llm_and_tts`).
7. **Protocol changes are contract changes.** Edit `protocol.py`, `docs/PROTOCOL.md`, `index.html` and `terminal_client.py` together. Each TTS request ends with exactly one `tts_chunk` that has `is_last: true` ("smooth" playback flushes on it).

## Conversation quality: where to look

- The persona and length rules are `settings.system_prompt` (`config.py`). The reply-language rule is appended per turn by `build_system_prompt`.
- History depth is `CONVERSATION_MAX_TURNS`. Adapters receive history already trimmed and must not trim it again (`llm/base.py::build_chat_messages`).
- `LLM_MAX_NEW_TOKENS` below ~100 truncates Indic replies. Qwen3 runs with `enable_thinking=False` in every adapter. Keep it that way, because `<think>` text would be spoken.
- `MockLLMAdapter` always returns the same canned reply. If "the agent ignores what I say", first check that it isn't running with `--mock-llm` or `USE_MOCK_LLM=true`.

## Audio quality: where to look

- Choppy or broken voice usually comes from TTS slower than real time (Svara on an L4 streams ~0.6x), too many tiny frames, scheduling gaps, or a TTS restart per sentence. Measure first: time each `tts_chunk` against its audio length. Parallel synthesis and the underrun guard live in `pipeline/synthesis.py` (`TTS_PARALLEL_REQUESTS`, `TTS_UNDERRUN_MARGIN`). The fixes live in `PCM16Coalescer` (`TTS_MIN_FRAME_MS`), `group_for_synthesis` (`TTS_MAX_CHARS_PER_REQUEST`), and the jitter buffer in `index.html` (`JITTER` constants, `playTTSChunk`).
- Bad pronunciation of numbers or symbols: check `normalization/indic_norm.py` and `segmenter.clean_for_speech`.
- Local Svara on base Macs is slower than real time. The client then defaults to "smooth" playback (`/v1/client-config`).

## Workflow expectations

- Run `ruff check` and the CI test set before you say a change is done. Add or adjust tests in `tests/` for every behaviour change. Pipeline tests use the fakes in `tests/test_pipeline.py` (`ScriptedVAD`, `RecordingLLM`, `TinyChunkTTS`), not real models.
- New settings go in `config.py` with a `description`, in `.env.example`, and in the README config table when they are user-facing.
- Keep docs in step: architectural changes go in `docs/ARCHITECTURE.md`, message changes in `docs/PROTOCOL.md`.
- Never commit `.env`, model weights, or downloaded `silero-vad-master/` / `svara-tts-inference-main/` trees.
- CI (`.github/workflows/ci.yml`) runs ruff 0.16.3 and pytest on Python 3.11 with CPU torch. Deploys are manual (`deploy.yml`).

# CODE.md: Engineering Standards

How code is written in this repository. Every rule comes from a real need of a streaming, low-latency, memory-constrained voice pipeline. When a rule and a measured result disagree, change the rule in this file, in the same PR.

---

## 1. Low-level design principles

### 1.1 Single Responsibility: one reason to change

Every module and class owns exactly one concern.

| Unit | Its one job | It must NOT |
| :-- | :-- | :-- |
| `SentenceSegmenter` | Turn a token stream into speakable sentences | Know about TTS, sockets or the LLM |
| `ConversationMemory` | Remember spoken exchanges, carry over interrupted input | Build HTTP payloads or call models |
| `PCM16Coalescer` | Re-frame PCM bytes | Encode base64 or send events |
| `VoiceSession` | Orchestrate one turn and own session state | Parse text, resample audio inline, or pick backends |
| `server/app.py` | Composition root: choose and build adapters | Hold per-session state |

If a method needs a comment like `# 1. do X  # 2. do Y  # 3. do Z`, split it into `_do_x`, `_do_y` and `_do_z`. `VoiceSession._run_turn` is the model: it calls `_transcribe`, `_generate_and_speak` and `_speak`, and each of those is short enough to read on one screen.

### 1.2 Open/Closed: extend by adding, not by editing

To add a new STT/LLM/TTS backend, **add a class** that implements the ABC and select it in `lifespan()`. The pipeline does not change.

```python
# Good: new backend, zero pipeline edits
class GroqLLMAdapter(LLMBase):
    async def generate_stream(self, prompt, history=None, system_prompt=None): ...
    @property
    def model_name(self) -> str: return "groq/qwen3"

# Bad: backend knowledge leaking into the orchestrator
if settings.llm_backend == "groq":   # inside session.py
    ...
```

### 1.3 Liskov Substitution: every adapter is interchangeable

Every implementation must honour the base-class contract, including the parts that are only implied:

- `TTSBase.synthesize_stream` yields `TTSAudioChunk`s of **whole PCM16 samples**, and the last one has `is_last=True`. Empty text yields nothing.
- `LLMBase.generate_stream` yields text deltas. It must stop cleanly when the consumer stops iterating (cancellation), which means worker threads are stopped in `finally`.
- `release_model()` is always safe to call, and is a no-op when nothing needs releasing.
- Mocks follow the same contract. A mock that breaks it hides bugs.

### 1.4 Interface Segregation: small, focused interfaces

The ABCs only contain what the pipeline calls. Optional capabilities are discovered, not forced on every adapter:

- `preload()` is optional. `_preload_models` uses `getattr(adapter, "preload", None)`.
- `uses_local_mlx` exists only on `SvaraTTSAdapter`, and is read with `getattr(..., False)`.

Do not add an abstract method that half the adapters implement as `pass`.

### 1.5 Dependency Inversion: depend on abstractions, inject concretes

- `VoiceSession.__init__` receives its collaborators (`stt`, `llm`, `tts`, `vad`, `normalizer`, `send_callback`, optional `memory`). It never constructs models.
- Tests inject fakes through the same constructor. No monkeypatching of globals is needed.
- `send_callback` decouples the session from FastAPI. The session has no idea a WebSocket exists.

### 1.6 DRY, with a single source of truth

| Fact | Lives only in |
| :-- | :-- |
| Tunables (thresholds, sizes, model IDs) | `config.py` `Settings` |
| Language codes → names | `languages.py` |
| Chat message assembly | `llm/base.py::build_chat_messages` |
| Wire message shapes | `protocol.py` |
| VAD frame geometry | `VAD_SAMPLE_RATE` / `VAD_FRAME_SAMPLES` in `session.py` |

If you copy a dict, a regex or a magic number into a second file, extract it instead.

### 1.7 KISS and YAGNI

Build for the current requirement. A plain function beats a class with one method. A dict lookup beats a plugin registry. Add a seam (interface, setting) only when there is a second real use, or when a test needs it.

### 1.8 Composition over inheritance

Inheritance is used only for the adapter ABCs, one level deep. Behaviour is shared by composing small objects: `VoiceSession` *has* a `ConversationMemory`, a `SentenceSegmenter` and a `PCM16Coalescer`. It does not inherit them.

---

## 2. Design patterns in use (use them the same way)

| Pattern | Where | Rule |
| :-- | :-- | :-- |
| **Adapter** | `stt/`, `llm/`, `tts/` | One class per backend, all behind the ABC |
| **Factory / composition root** | `server/app.py::lifespan`, `app.state.vad_factory` | The only place that reads backend-selection settings |
| **Strategy** | Overlapped vs sequential speaking in `_generate_and_speak` | The choice is made once, by a named predicate (`_overlap_llm_and_tts`) |
| **State machine** | `SessionState` | Every transition is visible in `session.py`. No other module sets `session.state` |
| **Producer/consumer** | LLM → `asyncio.Queue` → `_speak_from_queue`; worker thread → queue → async generator | Bounded where producers can outrun consumers. Use a `None` sentinel to end a queue |
| **Template (ABC)** | `STTBase`, `LLMBase`, `TTSBase` | Defaults for optional hooks (`release_model`) live on the base |

---

## 3. Async and concurrency rules

1. **Never block the event loop.** Model inference, file I/O and `subprocess` calls go through `asyncio.to_thread` or a dedicated thread. Anything over ~5 ms of CPU work counts as blocking.
2. **Thread → async handoff** goes through `loop.call_soon_threadsafe` or `asyncio.run_coroutine_threadsafe`. Never touch an `asyncio.Queue` directly from a thread.
3. **Every thread needs a stop signal** (`threading.Event`), checked inside its loop, and is joined in `finally`.
4. **Cancellation must propagate.** Catch `CancelledError` only to clean up or record state, then `raise`. Never use bare `except:` or `except BaseException`. Use `contextlib.suppress(asyncio.CancelledError)` only when awaiting a task *you* just cancelled.
5. **Every task you create has an owner** that cancels it (`_cancel_active_turn`, `stop()`). Fire-and-forget tasks are forbidden.
6. **Shared models are guarded** by the adapter's `asyncio.Lock`. Per-session state is never shared.
7. **Queues are bounded** wherever an external party (the client, the network) controls the producer rate (`input_queue_maxsize`).

---

## 4. Error handling

- **Fail fast at startup, degrade gracefully at runtime.** Missing VAD weights raise during `lifespan`. A TTS failure mid-turn sends an `error` event and the session goes back to `LISTENING`.
- Catch the **narrowest** exception you can handle. `except Exception` is allowed only at a boundary (WebSocket handler, turn runner, background worker), must log with `logger.exception` or `logger.warning`, and must carry `# noqa: BLE001` when it is deliberate.
- Fall back only **before** any output is produced. Once audio has reached the client, re-raise instead of mixing two voices (see `SvaraTTSAdapter.synthesize_stream`).
- Error messages say what failed **and** what to do: `"Configure SVARA_API_URL or run on macOS with 'say' installed"`.
- Never swallow an error silently. At minimum, `logger.debug` it with context.

---

## 5. Python style

- Python 3.11+, full type hints on every public function and method. Use `X | None`, not `Optional[X]`, and use `collections.abc` types.
- `ruff check` (ruff defaults, pinned in CI) must pass. Prefer readable over clever.
- Naming: `snake_case` functions and modules, `PascalCase` classes, `UPPER_SNAKE` module constants, and a leading `_` for anything not part of the public API.
- Docstrings: one line saying *what* for every public symbol. Add a paragraph for *why* when the reason is not obvious (latency, memory, a library quirk). Comments explain why, never restate the code.
- Functions stay short: aim for ≤ 40 lines and ≤ 3 levels of nesting. Use early returns instead of nested `if`s.
- No magic numbers. Name them (`VAD_FRAME_SAMPLES = 512`) or make them settings.
- Use `dataclass(frozen=True)` for value objects (`Turn`) and Pydantic models for anything that crosses the wire.
- Imports: stdlib, then third-party, then `voice_agent`, sorted. Heavy optional backends (`mlx`, `faster_whisper`) are imported **inside** the function that needs them, so other platforms can still import the module.
- Logging: use `logger = logging.getLogger(__name__)` with `%s` formatting (lazy), never f-strings in log calls. `info` for lifecycle, `debug` for per-frame or per-token detail, `warning` for degraded behaviour.

---

## 6. Configuration

- Every tunable is a `Settings` field with a type, a default and a `description`, and validation bounds (`ge=`, `le=`) where they make sense.
- Read `settings` at the composition root or at call time. Do not cache settings in module-level globals.
- A user-facing setting must also appear in `.env.example` and the README config table.

---

## 7. Testing standards

- **Every behaviour change ships with a test.** Bug fix: a test that fails without the fix.
- Unit-test pure logic directly (`test_segmenter.py`, `test_conversation.py`, coalescer tests in `test_audio.py`).
- Test the pipeline through `VoiceSession` with fakes (`ScriptedVAD`, `RecordingLLM`, `TinyChunkTTS`, `Mock*Adapter`). No network, no model weights, no `sleep`-based timing beyond short polling.
- Tests assert **behaviour** (events sent, what the LLM was asked, what was spoken), not private implementation details.
- Tests that need real hardware or downloaded assets live in `test_real_smoke.py` or are clearly marked. They are excluded from CI.
- Test names describe the expectation: `test_truncated_trailing_fragment_is_not_spoken_or_remembered`.

---

## 8. Performance budget (voice-specific)

- Target: **≤ 1.2 s** from end of user speech to first audio on GPU. Do not add work to the hot path (per-frame VAD loop, per-token LLM loop, per-chunk TTS loop) without measuring it.
- Per-token and per-chunk code must be O(1) or close to it. No logging at `info` level there.
- Audio leaves the server in frames of at least `TTS_MIN_FRAME_MS`. Do not send one WebSocket message per network read.
- Stay within the 8 GB Mac memory budget in `docs/ARCHITECTURE.md`. A new resident model needs a memory note there.

---

## 9. Review checklist

Before opening a PR, confirm each item:

- [ ] Each new or changed class has one responsibility. No backend `if`s in the pipeline.
- [ ] No blocking call on the event loop. Every task and thread has an owner and a stop path.
- [ ] `CancelledError` propagates. Cleanup happens in `finally`.
- [ ] New tunables are in `config.py`, `.env.example` and the README.
- [ ] Protocol changes are mirrored in `protocol.py`, `docs/PROTOCOL.md`, `index.html` and `terminal_client.py`.
- [ ] Tests are added or updated. `ruff check` and the CI test set pass locally.
- [ ] Docs (`ARCHITECTURE.md`, `CLAUDE.md`) are updated if the architecture or a workflow changed.

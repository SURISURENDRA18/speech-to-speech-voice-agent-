# Speech-to-Speech Voice Agent Architecture

## 1. System Overview

This system implements a full-duplex, low-latency, speech-to-speech conversational pipeline built on Python and FastAPI. The architecture takes inspiration from [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech), but is tailored for multilingual Indian languages and optimized for an Apple Silicon Mac M2 Pro with 8 GB unified memory.

```
                              ┌────────────────────────────────────────────────────────┐
                              │                 FastAPI WebSocket Session              │
                              └────────────────────────────────────────────────────────┘
                                            │                            ▲
                 [Audio Input Chunks]       │                            │   [TTS Audio Chunks]
                                            ▼                            │
                              ┌───────────────────────────┐              │
                              │     Silero VAD v6         │              │
                              │ (TorchScript Streaming)   │              │
                              └─────────────┬─────────────┘              │
                                            │ (Turn Complete)            │
                                            ▼                            │
                              ┌───────────────────────────┐              │
                              │ Whisper large-v3-turbo    │              │
                              │ (809M Multilingual STT)   │              │
                              └─────────────┬─────────────┘              │
                                            │ (User Transcript)          │
                                            ▼                            │
                              ┌───────────────────────────┐              │
                              │      Qwen3 1.7B LLM       │              │
                              │   (Streaming Generator)   │              │
                              └─────────────┬─────────────┘              │
                                            │ (Streaming Text Tokens)    │
                                            ▼                            │
                              ┌───────────────────────────┐              │
                              │ Indic Text Normalization  │              │
                              │ (WFST / Rule-Based)       │              │
                              └─────────────┬─────────────┘              │
                                            │ (Normalized Text Tokens)   │
                                            ▼                            │
                              ┌───────────────────────────┐              │
                              │         Svara TTS         │──────────────┘
                              │   (SNAC 24kHz Decoder)    │
                              └───────────────────────────┘
```

---

## 2. Pipeline Stages

### 2.1 Voice Activity Detection (VAD)
- **Model**: Silero VAD v6 (`silero_vad.jit`).
- **Operation**: Operates frame-by-frame on 512-sample (32 ms @ 16 kHz) chunks.
- **State Machine**: Uses `VADIterator` logic. When speech probability exceeds threshold (default 0.5), it emits `speech_start`. When speech ceases for `min_silence_duration_ms` (default 300 ms), it triggers `speech_end` and dispatches the accumulated speech buffer to the STT stage.
- **Barge-in / Interruption**: If speech starts while TTS playback or LLM generation is active, the session immediately signals cancellation to cancel the LLM and TTS tasks, clears queues, and emits an `interrupted` event to the client.

### 2.2 Speech-to-Text (STT)
- **Model**: Whisper `large-v3-turbo` (809M parameters).
- **Audio Prep**: Speech audio is resampled to 16 kHz mono, normalized to float32 [-1.0, 1.0], padded/trimmed to log-mel spectrogram features, and decoded.
- **Languages**: Multilingual support across Indic languages (Hindi, Bengali, Marathi, Tamil, Telugu, etc.) and English.
- **Distinction**: `large-v3-turbo` uses 4 decoder layers (~809M params) compared to standard `large-v3` (~1.55B params), cutting memory by almost 50% and improving decoding latency.

### 2.3 Conversational LLM
- **Model**: Qwen3 1.7B (`Qwen3ForCausalLM`).
- **Generation**: Asynchronous token streaming. Emits sentence/phrase deltas to minimize Time-to-First-Audio (TTFA).
- **Context Management**: Bounded conversational history (keeps recent turns) and limited output tokens (`max_tokens=128` by default) to keep memory usage low and prevent runaway latency.

### 2.4 Indic Text Normalization
- **Purpose**: Normalizes numbers, dates, fractions, currency symbols, and abbreviations into their spoken forms before reaching TTS (e.g., "₹500" -> "पांच सौ रुपये" / "five hundred rupees").
- **Implementation**: Dual-tier adapter:
  - High fidelity: NVIDIA NeMo / Kenpath WFST `indic_text_normalization` (when Pynini/OpenFST is installed).
  - Resilient fallback: High-speed rule-based Indic text normalizer covering cardinals, ordinals, decimals, currency, and dates for all 19 Indic languages and Indian English.

### 2.5 Text-to-Speech (TTS)
- **Model**: Svara-TTS (`svara-tts-v1`) with SNAC neural audio codec.
- **Local Mac behavior**: Without `SVARA_API_URL`, the adapter uses macOS `say` to produce spoken audio for playback. This is a local fallback, not Svara inference; use a remote Svara server for Svara voices.
- **Architecture**: Text tokens -> Codec token indices -> `SvaraMapper` -> `SNACCodec` -> 24 kHz 16-bit PCM.
- **Runtime paths**:
  - Remote/Docker Svara API endpoint (for production GPU vLLM deployments).
  - Local MPS/CPU SNAC decoder with streaming chunks.
  - Mock audio generator for fast local test suites without downloading multi-gigabyte TTS weights.

---

## 3. Concurrency, Queues, and Lifespan

1. **Lifespan Management**:
   Models are loaded once during FastAPI startup (`@asynccontextmanager` lifespan) and stored on `app.state`. They are shared across incoming WebSocket connections without re-initialization.
2. **Bounded Queues**:
   Each WebSocket session has bounded audio input (`asyncio.Queue(maxsize=100)`) and output queues to prevent memory accumulation if a client stalls.
3. **Cancellation & Cleanup**:
   Every async pipeline task is wrapped in an `asyncio.Task`. When a client disconnects or triggers barge-in:
   - All active generation tasks are cancelled immediately (`task.cancel()`).
   - Session buffers and VAD iterator states are reset.
   - Sockets are closed gracefully.

---

## 4. Memory Budget for 8 GB Unified Memory Mac

| Component | FP16 Memory | Int4/Quantized | Status on Apple Silicon |
| :--- | :--- | :--- | :--- |
| **macOS & System** | ~2.5 GB | ~2.5 GB | OS baseline |
| **Silero VAD v6** | ~5 MB | ~5 MB | Verified local CPU/MPS |
| **Whisper large-v3-turbo** | ~1.6 GB | ~800 MB | Verified architecture (809M) |
| **Qwen3 1.7B** | ~3.4 GB | ~1.1 GB | Verified local weights in HF cache |
| **Svara vLLM (Upstream)** | ~16 GB (GPU) | N/A | **Hard blocker**: vLLM requires CUDA |
| **Svara SNAC Decoder** | ~80 MB | ~80 MB | Verified local MPS/CPU |

**Operating Strategy**:
1. Run Silero VAD locally via TorchScript JIT (uses <10 MB).
2. Lazy-load STT / LLM models on demand.
3. For local Mac development, provide mock / test adapters that require 0 GB extra RAM.
4. For full pipeline with Svara, connect via remote HTTP transport to a GPU host running vLLM, or run quantized MLX/Transformers locally.

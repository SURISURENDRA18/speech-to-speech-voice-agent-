# Full-Duplex Speech-to-Speech WebSocket Protocol Specification

## 1. Protocol Overview

The speech-to-speech agent provides a bidirectional, full-duplex WebSocket connection at `/ws/voice`.
The connection handles streaming raw audio from the client, realtime Voice Activity Detection (VAD), Speech-to-Text (STT), Language Model (LLM) text generation, Indic text normalization, Text-to-Speech (TTS) audio synthesis, and immediate user interruption (barge-in).

### Transport & Encoding
- **WebSocket URL**: `ws://<host>:<port>/ws/voice`
- **Control & Event Messages**: JSON strings (UTF-8)
- **Audio Stream (Client to Server)**: 
  - Preferred: Binary WebSocket messages containing linear PCM 16-bit signed, little-endian, mono audio.
  - Fallback / Alternative: JSON messages with base64-encoded PCM audio: `{"type": "audio", "data": "<base64>"}`.
- **Audio Stream (Server to Client)**:
  - JSON chunks with base64-encoded audio payload: `{"type": "tts_chunk", ...}` or binary audio frames when configured.
  - Standard output rate: 24,000 Hz, 16-bit signed PCM mono (native Svara SNAC output).
  - Client input rate: Configurable (default 16,000 Hz or 24,000 Hz or 48,000 Hz). The server automatically resamples any input rate to 16,000 Hz for VAD and Whisper.

---

## 2. Session Lifecycle

```
Client                                                  Server
  │                                                       │
  │─── WebSocket Connect ────────────────────────────────>│
  │<── {"type": "connected", "session_id": "..."} ────────│
  │                                                       │
  │─── {"type": "start", "config": {...}} ───────────────>│
  │<── {"type": "session_started", ...} ──────────────────│
  │                                                       │
  │─── [Binary PCM Audio Chunk] ─────────────────────────>│
  │    (VAD detects voice start)                          │
  │<── {"type": "vad_speech_start", "timestamp": ...} ────│
  │                                                       │
  │─── [Binary PCM Audio Chunk] ... ─────────────────────>│
  │    (VAD detects voice end / turn complete)            │
  │<── {"type": "vad_speech_end", "duration_ms": ...} ───│
  │                                                       │
  │<── {"type": "transcript", "text": "...", ...} ────────│
  │<── {"type": "llm_chunk", "delta": "...", ...} ────────│
  │<── {"type": "tts_chunk", "audio": "<base64>", ...} ──│
  │<── {"type": "turn_complete", ...} ───────────────────│
  │                                                       │
  │─── [New Speech during TTS playback (Barge-in)] ──────>│
  │<── {"type": "interrupted", "reason": "user_speech"} ──│
  │                                                       │
  │─── {"type": "stop"} ─────────────────────────────────>│
  │<── {"type": "session_stopped"} ───────────────────────│
  │─── WebSocket Close ──────────────────────────────────>│
```

---

## 3. Client-to-Server Messages

### 3.1 `start` (Session Configuration)
Sent by the client immediately after connection to initialize session settings.
```json
{
  "type": "start",
  "config": {
    "sample_rate": 16000,
    "channels": 1,
    "language": "hi",
    "voice": "hi_female",
    "vad_threshold": 0.5,
    "min_silence_duration_ms": 300,
    "system_prompt": "You are a helpful, respectful multilingual Indian assistant."
  }
}
```

### 3.2 Audio Input
- **Binary Frame**: Bytes of 16-bit mono PCM audio (e.g. 512 samples @ 16kHz = 1,024 bytes per frame).
- **JSON Frame**:
```json
{
  "type": "audio",
  "data": "<base64-encoded PCM bytes>"
}
```

### 3.3 `interrupt` (Manual Cancellation)
Explicit barge-in triggered by the client UI.
```json
{
  "type": "interrupt"
}
```

### 3.4 `stop` (End Session)
Ends the conversation turn and resets session context.
```json
{
  "type": "stop"
}
```

---

## 4. Server-to-Client Messages

### 4.1 `connected`
Acknowledges initial WebSocket handshake.
```json
{
  "type": "connected",
  "session_id": "f5e9d9b4-b81b-4171-8b2b-0cb93d7cb43b",
  "server_time": 1774354200.123
}
```

### 4.2 `session_started`
Confirms active pipeline initialization.
```json
{
  "type": "session_started",
  "session_id": "f5e9d9b4-b81b-4171-8b2b-0cb93d7cb43b",
  "config": {
    "sample_rate": 16000,
    "language": "hi",
    "voice": "hi_female",
    "stt_model": "whisper-large-v3-turbo",
    "llm_model": "qwen3-1.7b",
    "tts_model": "svara-tts-v1"
  }
}
```

### 4.3 `vad_speech_start`
Triggered immediately when Silero VAD detects user speech.
```json
{
  "type": "vad_speech_start",
  "timestamp": 1774354201.450
}
```

### 4.4 `vad_speech_end`
Triggered when user silence exceeds `min_silence_duration_ms`.
```json
{
  "type": "vad_speech_end",
  "timestamp": 1774354203.250,
  "duration_ms": 1800
}
```

### 4.5 `transcript`
STT result from Whisper.
```json
{
  "type": "transcript",
  "text": "नमस्ते, आप कैसे हैं?",
  "language": "hi",
  "is_final": true
}
```

### 4.6 `llm_chunk`
Token or sentence delta streamed from the LLM.
```json
{
  "type": "llm_chunk",
  "delta": "नमस्ते! ",
  "accumulated": "नमस्ते! "
}
```

### 4.7 `tts_chunk`
Synthesized speech chunk from Svara TTS (SNAC 24kHz).
```json
{
  "type": "tts_chunk",
  "audio": "<base64-encoded PCM bytes>",
  "sample_rate": 24000,
  "channels": 1,
  "format": "pcm16",
  "is_last": false
}
```

### 4.8 `interrupted`
Emitted immediately when barge-in occurs. The client must immediately discard queued playback audio.
```json
{
  "type": "interrupted",
  "reason": "user_speech"
}
```

### 4.9 `turn_complete`
Emitted after all TTS chunks for a turn have been dispatched.
```json
{
  "type": "turn_complete",
  "user_text": "नमस्ते, आप कैसे हैं?",
  "assistant_text": "नमस्ते! मैं बिल्कुल ठीक हूँ। आपकी क्या सहायता कर सकता हूँ?"
}
```

### 4.10 `error`
Reports pipeline or protocol errors without dropping the WebSocket.
```json
{
  "type": "error",
  "code": "AUDIO_DECODE_ERROR",
  "message": "Malformed PCM audio packet received."
}
```

---

## 5. Audio Specs and Resampling
- **VAD Input**: Strict 16,000 Hz, 1-channel, 16-bit signed integer or float32. Input chunks of 512 samples (32ms).
- **Whisper Input**: 16,000 Hz, 1-channel float32 normalized [-1.0, 1.0].
- **Svara TTS Output**: 24,000 Hz, 1-channel, 16-bit signed PCM (or float32).
- **Internal Conversion**: Linear interpolation / Polyphase resampling with clamping to avoid clipping.

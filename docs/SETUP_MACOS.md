# macOS Apple Silicon Setup Guide (Mac M2 Pro, 8GB Unified Memory)

This guide documents the setup, execution, and memory optimization of the local Speech-to-Speech Voice Agent on Apple Silicon Macs running macOS.

---

## 1. System Requirements & Hardware Context

- **Hardware**: Apple Silicon Mac (M2 Pro)
- **Memory**: 8 GB Unified Memory (shared dynamically between CPU and GPU)
- **Accelerator**: Apple Metal Performance Shaders (MPS) via `torch.backends.mps`
- **Recommended Python**: Python 3.11 or 3.12 (for maximum package wheel compatibility with Numba and Pynini)

---

## 2. Environment Setup

### Option A: Using `uv` (Fastest, Recommended)

```bash
# Create and activate a Python 3.11 virtual environment
uv venv .venv --python 3.11
source .venv/bin/activate

# Install required core packages with Apple Silicon wheels
uv pip install \
  torch torchvision torchaudio \
  transformers accelerate \
  fastapi "uvicorn[standard]" pydantic pydantic-settings websockets \
  numpy soundfile pyyaml aiohttp pytest pytest-asyncio ruff

# Install local Apple Silicon Svara TTS (recommended for natural Indic speech)
uv pip install "mlx-audio[tts]"
```

### Option B: Using Homebrew Python 3.11

```bash
# Using existing Homebrew Python 3.11 on this Mac
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Component Compatibility & Runtime Strategy

### 3.1 Silero VAD v6 (Verified Local)
- **Engine**: TorchScript JIT (`silero_vad.jit`).
- **Memory**: < 10 MB.
- **Latency**: < 1 ms per 32 ms chunk on Apple Silicon CPU/MPS.
- **Usage**: Already pre-packaged in `silero-vad-master/src/silero_vad/data/silero_vad.jit`. Runs immediately with zero downloads.

### 3.2 Whisper large-v3-turbo (809M Parameters)
- **Architecture**: 809M parameters (32 encoder layers, 4 decoder layers).
- **Distinction**: Note that standard `large-v3` has ~1.55B parameters. `large-v3-turbo` uses approximately half the weights and memory.
- **Memory**: ~1.6 GB in FP16 on MPS.
- **Execution**:
  - Automatically loads via Hugging Face pipeline `openai/whisper-large-v3-turbo` or local upstream `whisper-main`.
  - For rapid testing or memory-constrained environments, use `--mock-stt`.

### 3.3 Qwen3 1.7B LLM (Verified Local Weights)
- **Weights Status**: Verified present in local Hugging Face cache at:
  `~/.cache/huggingface/hub/models--unsloth--Qwen3-1.7B/snapshots/6262b50d6c1f8ee5e4ac750d710c33603bfc2a0c/`
- **Memory**: ~3.4 GB in BFloat16/Float16.
- **Execution**:
  - Loads via `transformers.Qwen3ForCausalLM` or `AutoModelForCausalLM` with `torch_dtype=torch.bfloat16` and device `mps`.
  - Uses `TextIteratorStreamer` for low-latency asynchronous token streaming.
  - To save RAM during testing or multi-service development, use `--mock-llm`.

### 3.4 Svara TTS (Apple Silicon)
- **Upstream limitation**: The bundled `svara-tts-inference` server requires vLLM/CUDA and cannot run natively on macOS.
- **Local Apple Silicon route**: The adapter uses `mlx-community/svara-tts-v1-4bit` through `mlx-audio[tts]`. The model is about 1.86 GB, generates 24 kHz mono audio, and downloads from Hugging Face on first use. It is quantized for Apple Silicon and may still compete for unified memory with Whisper and Qwen.
- **Fallback**: If MLX or the model is unavailable, the adapter speaks with macOS `say`; this is audible speech but not Svara. `/health` identifies which path is selected. To use a remote GPU Svara server instead, set `SVARA_API_URL` in `.env`.

### 3.5 Indic Text Normalization (Pynini / OpenFST)
- **Hard Blocker for WFST**: `pynini` requires OpenFST C++ library and headers, which on macOS requires OpenFST 1.8.2/1.8.3. Homebrew currently supplies OpenFST 1.8.4, which breaks Pynini compilation.
- **Mac Solution**: The agent incorporates a dual-mode normalizer:
  - If Pynini is available, it uses NVIDIA NeMo / Kenpath WFST.
  - If Pynini is absent, it seamlessly falls back to the built-in rule-based Indic normalizer covering cardinals, ordinals, decimals, percentages, and currencies (₹/Rs) across Hindi, English, and Indic languages.

---

## 4. Running the Agent

### Test audible speech without loading Whisper or Qwen
```bash
python3 main.py --mock-stt --mock-llm
```
This uses canned STT/LLM responses. TTS uses local MLX Svara when installed, otherwise macOS `say`. `--mock-all` uses a synthetic test tone for TTS, not spoken words.

### Start with real local Silero VAD, Whisper, Qwen3, Indic normalization, and Svara
```bash
python3 main.py
```
The first Svara response downloads the MLX model. Do not pass mock flags for an actual conversation: mock STT returns canned text and mock LLM returns a canned answer.

### Inspect Endpoints
- **Health Check**: `curl http://localhost:8000/health`
- **Voice Profiles**: `curl http://localhost:8000/v1/voices`
- **Hardware & Model Info**: `curl http://localhost:8000/info`
- **WebSocket Protocol**: `ws://localhost:8000/ws/voice`

---

## 5. Running the Test Suite

```bash
# Run all unit, integration, and real smoke tests
python3 -m pytest tests/ -v

# Run code style & lint checks
python3 -m ruff check voice_agent/ tests/
```

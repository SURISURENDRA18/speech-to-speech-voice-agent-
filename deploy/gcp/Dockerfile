# Voice agent server: Silero VAD (CPU) + Whisper large-v3-turbo (CUDA) + WebSocket
# session. Talks to the Qwen3 and Svara vLLM containers over the compose network.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY deploy/gcp/requirements-gpu.txt /tmp/requirements-gpu.txt
RUN pip install -r /tmp/requirements-gpu.txt

# Silero VAD TorchScript model at the path voice_agent/vad/detector.py looks for.
RUN mkdir -p silero-vad-master/src/silero_vad/data \
    && curl -fsSL -o silero-vad-master/src/silero_vad/data/silero_vad.jit \
       https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.jit

# CUDA 12 libraries for CTranslate2 (faster-whisper).
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib

COPY main.py ./
COPY voice_agent ./voice_agent

EXPOSE 8000
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8000"]

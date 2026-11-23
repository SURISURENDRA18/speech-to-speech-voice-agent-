# syntax=docker/dockerfile:1
# Svara TTS server (Kenpath/svara-tts-inference) built on the official vLLM image.
# Upstream's Dockerfile installs Python from the deadsnakes PPA, which no longer
# publishes for Ubuntu 22.04, so it fails to build. The vLLM image already ships
# Python, CUDA PyTorch, vLLM and flashinfer, so only the server's small extras
# are added here. Build context: the cloned svara-tts-inference repo.
ARG VLLM_IMAGE=vllm/vllm-openai:v0.34.0
FROM ${VLLM_IMAGE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir \
    snac langcodes language_data python-dotenv pyyaml hf-transfer soundfile

# Svara imports torchaudio only for voice cloning (loading/resampling a reference
# clip). torchaudio is in maintenance mode and has no build for newer torch, so
# fall back to this minimal stand-in when no matching wheel exists.
COPY <<'EOF' /opt/torchaudio_shim/torchaudio/__init__.py
"""Minimal torchaudio stand-in (no torchaudio build exists for this torch)."""
import soundfile as _sf
import torch as _torch

from . import functional, transforms  # noqa: F401


def load(src, *args, **kwargs):
    data, sample_rate = _sf.read(src, dtype="float32", always_2d=True)
    return _torch.from_numpy(data.T.copy()), sample_rate
EOF

COPY <<'EOF' /opt/torchaudio_shim/torchaudio/functional.py
import torch
import torch.nn.functional as F


def resample(waveform, orig_freq, new_freq, *args, **kwargs):
    if int(orig_freq) == int(new_freq):
        return waveform
    shape = waveform.shape
    x = waveform.reshape(-1, 1, shape[-1]).float()
    n = int(round(shape[-1] * float(new_freq) / float(orig_freq)))
    y = F.interpolate(x, size=n, mode="linear", align_corners=False)
    return y.reshape(*shape[:-1], n).to(waveform.dtype)
EOF

COPY <<'EOF' /opt/torchaudio_shim/torchaudio/transforms.py
import torch

from .functional import resample


class Resample(torch.nn.Module):
    def __init__(self, orig_freq=16000, new_freq=16000, *args, **kwargs):
        super().__init__()
        self.orig_freq, self.new_freq = orig_freq, new_freq

    def forward(self, waveform):
        return resample(waveform, self.orig_freq, self.new_freq)
EOF

RUN TV="$(python3 -c 'import torch; print(torch.__version__.split("+")[0])')" \
    && CU="cu$(python3 -c 'import torch; print(torch.version.cuda.replace(".", ""))')" \
    && (python3 -m pip install --no-cache-dir "torchaudio==${TV}" --index-url "https://download.pytorch.org/whl/${CU}" \
        || python3 -m pip install --no-cache-dir --no-deps "torchaudio==${TV}" \
        || cp -r /opt/torchaudio_shim/torchaudio "$(python3 -c 'import site; print(site.getsitepackages()[0])')/") \
    && python3 -c "import torchaudio, torch; print('torch', torch.__version__, '| torchaudio', getattr(torchaudio, '__version__', 'stand-in'))"

WORKDIR /app
COPY tts_engine/ ./tts_engine/
COPY api/ ./api/
COPY assets/ ./assets/

# Fail the build early if the server's imports are broken.
RUN cd /app/api && python3 -c "import sys; sys.path.insert(0, '/app'); import tts_engine.utils, tts_engine.voice_config, api.models; print('svara imports OK')"

WORKDIR /app/api
ENV API_HOST=0.0.0.0 API_PORT=8080
EXPOSE 8080
ENTRYPOINT []
CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]

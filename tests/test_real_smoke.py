"""Real smoke tests for locally available models and dependencies on Apple Silicon."""

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from voice_agent.llm.qwen_llm import find_local_qwen3_path
from voice_agent.normalization.indic_norm import IndicTextNormalizer
from voice_agent.tts.svara_tts import load_svara_voices
from voice_agent.vad.detector import SileroVADDetector, find_silero_model_path


def test_real_mps_accelerator():
    """Verify Apple Silicon Metal Performance Shaders (MPS) availability."""
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        pytest.skip("This Python/PyTorch runtime does not have working MPS support")
    # Test a small tensor operation on MPS
    device = torch.device("mps")
    x = torch.ones((2, 2), device=device)
    y = x * 2.0
    assert torch.all(y == 2.0).item() is True


def test_real_silero_vad_smoke():
    """Verify real Silero VAD v6 TorchScript model inference."""
    model_path = find_silero_model_path()
    assert model_path.exists()

    detector = SileroVADDetector(model_path=model_path, threshold=0.4)
    # Process 512 samples of silence
    silence = np.zeros(512, dtype=np.float32)
    events = detector.process_chunk(silence)
    assert len(events) == 0

    # Process 512 samples of tone
    t = np.linspace(0, 512 / 16000, 512, endpoint=False, dtype=np.float32)
    tone = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    _ = detector.process_chunk(tone)


def test_real_qwen3_local_assets():
    """Verify local cached Qwen3 1.7B assets and tokenizer."""
    local_path = find_local_qwen3_path()
    assert local_path is not None, "Local Qwen3 1.7B snapshot should be detected in HF cache"
    assert (local_path / "model.safetensors").exists()
    assert (local_path / "config.json").exists()

    # Load real tokenizer from local snapshot
    tokenizer = AutoTokenizer.from_pretrained(str(local_path))
    tokens = tokenizer("नमस्ते दुनिया! How are you?")
    assert len(tokens["input_ids"]) > 0


def test_real_indic_normalization_smoke():
    """Verify Indic text normalization on actual text."""
    normalizer = IndicTextNormalizer(default_lang="hi")
    out = normalizer.normalize("कुल ₹500 का बिल है, 50% छूट मिली।", lang="hi")
    assert "रुपये" in out
    assert "प्रतिशत" in out


def test_real_svara_voices_metadata():
    """Verify Svara voice configurations extracted from assets."""
    voices = load_svara_voices()
    assert len(voices) >= 38  # 19 languages * 2 genders
    langs = {v["languages"][0] for v in voices if v.get("languages")}
    assert "hi" in langs
    assert "bn" in langs
    assert "ta" in langs
    assert "te" in langs
    assert "en" in langs

"""Unit tests for voice_agent/audio.py."""

import numpy as np
import torch

from voice_agent.audio import (
    BoundedAudioAccumulator,
    PCM16Coalescer,
    base64_to_pcm_bytes,
    compute_rms_db,
    float32_to_pcm16,
    pcm16_to_float32,
    pcm_bytes_to_base64,
    resample_audio,
    slice_into_frames,
)


def test_pcm16_and_float32_roundtrip():
    # Generate 16000 samples of 440Hz sine wave
    sample_rate = 16000
    t = np.linspace(0, 1, sample_rate, endpoint=False, dtype=np.float32)
    original_float = 0.5 * np.sin(2 * np.pi * 440 * t)

    pcm_bytes = float32_to_pcm16(original_float)
    assert len(pcm_bytes) == sample_rate * 2  # 2 bytes per sample (int16)

    restored_float = pcm16_to_float32(pcm_bytes)
    assert len(restored_float) == sample_rate
    # Quantization error for 16-bit PCM is <= 1 / 32768 ~ 3e-5
    np.testing.assert_allclose(original_float, restored_float, atol=1e-4)


def test_float32_from_torch_tensor():
    tensor = torch.tensor([0.0, 0.5, -0.5, 1.0, -1.0], dtype=torch.float32)
    pcm_bytes = float32_to_pcm16(tensor)
    restored = pcm16_to_float32(pcm_bytes)
    assert len(restored) == 5
    assert np.isclose(restored[1], 0.5, atol=1e-3)
    assert np.isclose(restored[2], -0.5, atol=1e-3)


def test_base64_conversions():
    raw = b"\x00\x00\xff\x7f\x00\x80"
    b64 = pcm_bytes_to_base64(raw)
    assert isinstance(b64, str)
    decoded = base64_to_pcm_bytes(b64)
    assert decoded == raw


def test_resample_audio():
    # 24kHz to 16kHz
    orig_sr = 24000
    target_sr = 16000
    audio_24k = np.ones(orig_sr, dtype=np.float32)
    resampled_16k = resample_audio(audio_24k, orig_sr, target_sr)
    assert len(resampled_16k) == target_sr
    np.testing.assert_allclose(resampled_16k, 1.0, atol=1e-5)

    # Empty array
    empty = np.empty(0, dtype=np.float32)
    assert len(resample_audio(empty, orig_sr, target_sr)) == 0

    # Same sample rate
    same = resample_audio(audio_24k, orig_sr, orig_sr)
    assert len(same) == len(audio_24k)


def test_slice_into_frames():
    audio = np.arange(1024, dtype=np.float32)
    frames = list(slice_into_frames(audio, frame_size=512))
    assert len(frames) == 2
    assert len(frames[0]) == 512
    assert len(frames[1]) == 512
    assert frames[0][0] == 0
    assert frames[1][0] == 512

    # Audio with remainder
    audio_odd = np.arange(1100, dtype=np.float32)
    frames_odd = list(slice_into_frames(audio_odd, frame_size=512))
    assert len(frames_odd) == 2


def test_compute_rms_db():
    silence = np.zeros(16000, dtype=np.float32)
    assert compute_rms_db(silence) == -100.0

    loud = np.ones(16000, dtype=np.float32)
    # RMS of 1 is 0 dBFS
    assert np.isclose(compute_rms_db(loud), 0.0, atol=1e-3)


def test_bounded_audio_accumulator():
    acc = BoundedAudioAccumulator(sample_rate=16000, max_seconds=1.0)
    assert acc.duration_seconds() == 0.0

    chunk = np.ones(8000, dtype=np.float32)
    success1 = acc.append(chunk)
    assert success1 is True
    assert acc.duration_seconds() == 0.5

    # Append another chunk that reaches the 1.0s limit (16000 samples)
    success2 = acc.append(chunk)
    assert success2 is True
    assert acc.duration_seconds() == 1.0

    # Append third chunk beyond limit
    success3 = acc.append(chunk)
    assert success3 is False
    assert acc.duration_seconds() == 1.0  # capped at 1.0s
    assert len(acc.get_audio()) == 16000

    acc.clear()
    assert len(acc) == 0
    assert len(acc.get_audio()) == 0


def test_pcm16_coalescer_merges_small_pieces_into_frames():
    # 24 kHz, 100 ms minimum -> 2400 samples = 4800 bytes per frame
    coalescer = PCM16Coalescer(sample_rate=24000, min_ms=100)
    piece = b"\x01\x00" * 600  # 25 ms
    assert coalescer.push(piece) is None
    assert coalescer.push(piece) is None
    assert coalescer.push(piece) is None
    frame = coalescer.push(piece)
    assert frame == piece * 4
    assert coalescer.flush() == b""


def test_pcm16_coalescer_passes_large_chunks_and_keeps_sample_alignment():
    coalescer = PCM16Coalescer(sample_rate=24000, min_ms=100)
    big = b"\x02\x00" * 3000 + b"\x07"  # 125 ms plus a dangling odd byte
    frame = coalescer.push(big)
    assert frame == b"\x02\x00" * 3000
    # The odd byte waits for its partner instead of splitting a sample.
    assert coalescer.push(b"\x00") is None
    assert coalescer.flush() == b"\x07\x00"

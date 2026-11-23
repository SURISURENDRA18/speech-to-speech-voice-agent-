"""Unit and real smoke tests for Silero VAD detector."""

import numpy as np

from voice_agent.audio import slice_into_frames
from voice_agent.vad.detector import SileroVADDetector, find_silero_model_path


def test_find_silero_model_path():
    path = find_silero_model_path()
    assert path.exists()
    assert path.name == "silero_vad.jit"


def test_silero_vad_initialization():
    detector = SileroVADDetector()
    assert detector.sample_rate == 16000
    assert detector.threshold == 0.5
    assert detector.is_speech_active is False


def test_silero_vad_silence_no_trigger():
    detector = SileroVADDetector()
    silence_chunk = np.zeros(512, dtype=np.float32)

    # Feed 10 chunks of pure silence
    for _ in range(10):
        events = detector.process_chunk(silence_chunk)
        assert len(events) == 0
        assert detector.is_speech_active is False


def test_silero_vad_speech_start_and_end_cycle():
    # Set threshold low so synthetic tone triggers speech reliably
    detector = SileroVADDetector(
        threshold=0.3,
        min_silence_duration_ms=100,  # 100ms silence = 1600 samples ~ 3-4 chunks
        speech_pad_ms=30,
    )

    t = np.linspace(0, 1, 16000, endpoint=False, dtype=np.float32)
    # Synthetic speech frequency sine wave
    speech_wave = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    # 1. Feed 5 silence chunks
    silence = np.zeros(512, dtype=np.float32)
    for _ in range(5):
        events = detector.process_chunk(silence)
        assert len(events) == 0

    # 2. Feed speech chunks
    all_events = []
    for frame in slice_into_frames(speech_wave[:4096], frame_size=512):
        events = detector.process_chunk(frame)
        all_events.extend(events)

    # Should have triggered speech_start
    start_events = [e for e in all_events if e.event_type == "speech_start"]
    assert len(start_events) >= 1
    assert detector.is_speech_active is True

    # 3. Feed silence chunks until speech_end triggers
    end_events = []
    for _ in range(15):
        events = detector.process_chunk(silence)
        end_events.extend([e for e in events if e.event_type == "speech_end"])

    assert len(end_events) >= 1
    end_event = end_events[0]
    assert end_event.speech_audio is not None
    assert len(end_event.speech_audio) > 0
    assert detector.is_speech_active is False


def test_silero_vad_reset():
    detector = SileroVADDetector(threshold=0.3)
    t = np.linspace(0, 0.5, 8000, endpoint=False, dtype=np.float32)
    speech_wave = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    for frame in slice_into_frames(speech_wave[:2048], frame_size=512):
        detector.process_chunk(frame)

    assert detector.is_speech_active is True
    detector.reset()
    assert detector.is_speech_active is False

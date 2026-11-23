"""Unit tests for the TTS underrun guard and synthesis rate tracking."""

import pytest

from voice_agent.pipeline.synthesis import (
    CALIBRATION_TEXT,
    PlayoutClock,
    SynthesisRateTracker,
    calibrate,
    reply_can_play_through,
    safe_to_start,
)
from voice_agent.tts.mock_tts import MockTTSAdapter


def test_fast_backend_never_waits():
    assert safe_to_start(0.0, 10.0, 0.0, realtime_factor=1.5, margin=1.2)


def test_slow_backend_waits_until_enough_audio_is_buffered():
    # 4 s sentence at 0.5x real time: playing 4 s takes 4 s, generating the
    # rest takes twice as long, so about half must be buffered first.
    kwargs = {"expected_seconds": 4.0, "queued_seconds": 0.0, "realtime_factor": 0.5, "margin": 1.0}
    assert not safe_to_start(1.0, **kwargs)
    assert safe_to_start(2.0, **kwargs)


def test_audio_already_queued_counts_toward_the_buffer():
    kwargs = {"expected_seconds": 4.0, "realtime_factor": 0.5, "margin": 1.0}
    assert not safe_to_start(0.5, queued_seconds=0.0, **kwargs)
    assert safe_to_start(0.5, queued_seconds=3.0, **kwargs)


def test_holdback_and_margin_make_the_guard_stricter():
    base = {"expected_seconds": 4.0, "queued_seconds": 0.0, "realtime_factor": 0.5}
    assert safe_to_start(2.0, margin=1.0, **base)
    assert not safe_to_start(2.0, margin=1.2, **base)
    assert not safe_to_start(2.0, margin=1.0, holdback_seconds=0.2, **base)


def test_tracker_replaces_prior_with_first_measurement_then_smooths():
    tracker = SynthesisRateTracker(realtime_factor=0.6, seconds_per_char=0.09)
    tracker.observe(
        "x" * 40,
        "hi",
        audio_seconds=4.0,
        streaming_seconds=3.0,
        first_chunk_seconds=1.0,
        request_seconds=4.0,
    )
    assert tracker.realtime_factor == pytest.approx(1.0)
    assert tracker.seconds_per_char("hi") == pytest.approx(0.1)
    assert tracker.seconds_per_char("en") == pytest.approx(0.09)  # per language

    tracker.observe(
        "x" * 40,
        "hi-IN",
        audio_seconds=2.0,
        streaming_seconds=3.0,
        first_chunk_seconds=0.5,
        request_seconds=4.0,
    )
    assert 0.5 < tracker.realtime_factor < 1.0
    assert 0.05 < tracker.seconds_per_char("hi") < 0.1


def test_slow_request_delivered_in_one_burst_is_not_measured_as_fast():
    # Regression: Svara sends a short sentence's audio in one burst at the end.
    # A burst after 0.8 s of generating 0.4 s of audio is 0.5x, not instant.
    tracker = SynthesisRateTracker(realtime_factor=0.6)
    tracker.observe(
        "x" * 10, "hi", audio_seconds=0.6, streaming_seconds=3.0,
        first_chunk_seconds=0.1, request_seconds=4.0,
    )  # a normal measurement: ~0.17x
    for streaming in (0.0, 0.06):  # one chunk, or a flush just past 50 ms
        tracker.observe(
            "x" * 5, "hi", audio_seconds=0.4, streaming_seconds=streaming,
            first_chunk_seconds=0.1, request_seconds=0.8,
        )
        assert tracker.realtime_factor < 1.0


def test_backend_that_answers_instantly_is_measured_as_fast():
    tracker = SynthesisRateTracker(realtime_factor=0.6)
    tracker.observe(
        "x" * 10, "hi", audio_seconds=1.0, streaming_seconds=0.0,
        first_chunk_seconds=1.0, request_seconds=0.05,
    )
    assert tracker.realtime_factor >= 1.0


def test_playout_clock_tracks_unplayed_audio():
    now = [100.0]
    clock = PlayoutClock(clock=lambda: now[0])
    clock.sent(1.0)
    clock.sent(0.5)
    assert clock.queued() == pytest.approx(1.5)
    now[0] += 1.0
    assert clock.queued() == pytest.approx(0.5)
    now[0] += 5.0
    assert clock.queued() == 0.0
    clock.sent(0.25)  # after running dry, playback restarts from "now"
    assert clock.queued() == pytest.approx(0.25)


class FakeJob:
    """Stand-in exposing what the look-ahead reads from a SynthesisJob."""

    def __init__(self, chars, generated=0.0, finished=False, started=True):
        self.tts_text = "x" * chars
        self.language = "hi"
        self.generated_seconds = generated
        self.finished = finished
        self.started = started


def _tracker():
    return SynthesisRateTracker(realtime_factor=0.5, seconds_per_char=0.1)


def test_look_ahead_holds_a_ready_sentence_for_a_slow_later_one():
    short_done = FakeJob(chars=10, generated=1.0, finished=True)
    long_running = FakeJob(chars=40, generated=0.5)  # 4 s expected, 0.5 s buffered
    jobs = [short_done, long_running]
    # Alone the finished sentence could play, but the next one would then stall.
    assert reply_can_play_through(jobs[:1], 0.0, _tracker(), margin=1.0)
    assert not reply_can_play_through(jobs, 0.0, _tracker(), margin=1.0)
    # Needs 1 s queued ahead + buffered >= remaining * (1/0.5 - 1).
    long_running.generated_seconds = 1.0  # 1 + 1.0 < 3.0
    assert not reply_can_play_through(jobs, 0.0, _tracker(), margin=1.0)
    long_running.generated_seconds = 1.5  # 1 + 1.5 >= 2.5
    assert reply_can_play_through(jobs, 0.0, _tracker(), margin=1.0)


def test_look_ahead_waits_for_the_current_job_to_start_but_not_later_ones():
    assert not reply_can_play_through([FakeJob(chars=10, started=False)], 0.0, _tracker(), 1.0)
    jobs = [FakeJob(chars=10, generated=1.0, finished=True), FakeJob(chars=40, started=False)]
    assert reply_can_play_through(jobs, 0.0, _tracker(), margin=1.0)


@pytest.mark.asyncio
async def test_calibrate_measures_the_backend():
    tracker = SynthesisRateTracker(realtime_factor=0.6, seconds_per_char=0.5)
    await calibrate(MockTTSAdapter(sample_rate=24000, chunk_duration_ms=50), tracker, "hi")
    # Mock TTS: 0.1 s of audio for the whole sentence, delivered almost at once.
    assert tracker.seconds_per_char("hi") == pytest.approx(0.1 / len(CALIBRATION_TEXT))
    assert tracker.realtime_factor > 1.0

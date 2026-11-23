"""Tests for streaming sentence segmentation and speech text cleanup."""

from voice_agent.pipeline.segmenter import (
    SentenceSegmenter,
    clean_for_speech,
    ends_with_terminator,
    group_for_synthesis,
)


def feed(segmenter: SentenceSegmenter, deltas: list[str]) -> list[str]:
    out: list[str] = []
    for delta in deltas:
        out.extend(segmenter.push(delta))
    return out


def test_splits_hindi_danda_and_latin_terminators():
    seg = SentenceSegmenter()
    sentences = feed(seg, ["नमस्ते", "!", " आप ", "कैसे हैं", "?", " मैं ठीक हूँ", "।"])
    assert sentences == ["नमस्ते!", "आप कैसे हैं?"]
    assert seg.flush() == "मैं ठीक हूँ।"


def test_does_not_split_decimal_numbers_across_tokens():
    seg = SentenceSegmenter()
    sentences = feed(seg, ["इसकी कीमत ₹2", ".", "5 लाख है", "।", " ठीक", "?"])
    assert sentences == ["इसकी कीमत ₹2.5 लाख है।"]
    assert seg.flush() == "ठीक?"


def test_does_not_split_on_abbreviations_or_initials():
    seg = SentenceSegmenter()
    sentences = feed(seg, ["Dr. Sharma met A. P. J. Kalam, e.g. ", "today. Next"])
    assert sentences == ["Dr. Sharma met A. P. J. Kalam, e.g. today."]
    assert seg.flush() == "Next"


def test_long_unpunctuated_run_is_split_for_latency():
    seg = SentenceSegmenter(max_chars=40)
    sentences = seg.push("one two three four five, six seven eight nine ten eleven")
    assert sentences == ["one two three four five,"]
    assert seg.pending.startswith(" six")


def test_skips_fragments_with_nothing_to_say():
    seg = SentenceSegmenter()
    assert feed(seg, ["😊", ".", " ", "हाँ।", " "]) == ["हाँ।"]


def test_clean_for_speech_strips_markdown_emojis_and_think_tags():
    raw = "<think>\n</think>**नमस्ते** 😊\n- [यहाँ](http://x) देखें `code` # शीर्षक"
    assert clean_for_speech(raw) == "नमस्ते यहाँ देखें code शीर्षक"
    # Hyphens inside words are kept.
    assert clean_for_speech("COVID-19 के बारे में") == "COVID-19 के बारे में"


def test_clean_for_speech_keeps_zero_width_joiner_inside_indic_words():
    malayalam = "അവന്‍ വന്നു"  # ZWJ forms the chillu letter
    bengali = "র‍্যাব"  # ZWJ forms the ya-phala
    assert clean_for_speech(malayalam) == malayalam
    assert clean_for_speech(bengali) == bengali
    # A ZWJ emoji sequence is still removed entirely.
    assert clean_for_speech("ठीक 👨‍👩‍👧 है") == "ठीक है"


def test_ends_with_terminator():
    assert ends_with_terminator("ठीक है।")
    assert ends_with_terminator('He said "yes."')
    assert not ends_with_terminator("और फिर")


def test_group_for_synthesis_merges_up_to_limit_without_cutting():
    sentences = ["a" * 30, "b" * 30, "c" * 30, "d" * 100]
    assert group_for_synthesis(sentences, max_chars=70) == [
        f"{'a' * 30} {'b' * 30}",
        "c" * 30,
        "d" * 100,
    ]
    assert group_for_synthesis([], max_chars=70) == []

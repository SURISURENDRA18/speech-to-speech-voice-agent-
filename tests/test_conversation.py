"""Tests for per-session conversation memory and prompt construction."""

import pytest

from voice_agent.languages import base_language, language_name
from voice_agent.llm.base import build_chat_messages
from voice_agent.pipeline.conversation import ConversationMemory, build_system_prompt


def test_memory_keeps_last_n_turns_in_order():
    memory = ConversationMemory(max_turns=2)
    for i in range(3):
        memory.record_turn(f"u{i}", f"a{i}")
    assert memory.history() == [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]


def test_unanswered_utterance_is_carried_into_next_turn():
    memory = ConversationMemory()
    first = memory.compose_user_message("मेरा नाम रवि है")
    memory.record_turn(first, "")  # interrupted before any audio was heard
    assert len(memory) == 0

    second = memory.compose_user_message("मुझे दिल्ली जाना है")
    assert second == "मेरा नाम रवि है मुझे दिल्ली जाना है"
    memory.record_turn(second, "ज़रूर रवि जी।")
    assert memory.compose_user_message("कब?") == "कब?"


def test_rejects_non_positive_max_turns():
    with pytest.raises(ValueError):
        ConversationMemory(max_turns=0)


def test_system_prompt_names_reply_language():
    assert "Reply only in Tamil" in build_system_prompt("Base.", "ta-IN")
    assert "Indian English" in build_system_prompt("Base.", "en")
    assert build_system_prompt("Base.", None).startswith("Base.")


def test_language_helpers():
    assert base_language("HI_in") == "hi"
    assert base_language(None) == "hi"
    assert language_name("bn") == "Bengali"
    assert language_name("xx") == "Hindi"


def test_build_chat_messages_does_not_truncate_history():
    history = [{"role": "user", "content": str(i)} for i in range(10)]
    messages = build_chat_messages("now", history, "sys")
    assert messages[0] == {"role": "system", "content": "sys"}
    assert messages[1:-1] == history
    assert messages[-1] == {"role": "user", "content": "now"}

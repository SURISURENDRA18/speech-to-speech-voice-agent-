"""Tests for the Qwen conversational prompt adapter."""

from voice_agent.llm.qwen_llm import Qwen3LLMAdapter


def test_qwen_voice_prompt_disables_thinking_tokens():
    class RecordingTokenizer:
        def __init__(self):
            self.options = None

        def apply_chat_template(self, messages, **options):
            self.options = options
            return "formatted voice prompt"

    adapter = Qwen3LLMAdapter()
    tokenizer = RecordingTokenizer()
    adapter._tokenizer = tokenizer

    prompt = adapter._format_chat_prompt([{"role": "user", "content": "नमस्ते"}])

    assert prompt == "formatted voice prompt"
    assert tokenizer.options["enable_thinking"] is False
    assert tokenizer.options["add_generation_prompt"] is True

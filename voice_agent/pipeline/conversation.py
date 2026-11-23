"""Per-session conversation memory and prompt construction."""

from dataclasses import dataclass

from voice_agent.languages import base_language, language_name

ChatMessage = dict[str, str]


@dataclass(frozen=True)
class Turn:
    """One completed exchange, holding only what the user actually heard."""

    user: str
    assistant: str


class ConversationMemory:
    """Bounded, per-session dialogue history.

    Responsibilities:
    * keep the last ``max_turns`` exchanges so the LLM can follow the thread;
    * record what the assistant actually *spoke*, not what it generated, so the
      model never refers to words the user did not hear;
    * carry an interrupted user utterance into the next prompt instead of
      dropping it.
    """

    def __init__(self, max_turns: int = 6):
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self._max_turns = max_turns
        self._turns: list[Turn] = []
        self._carry_over = ""

    def __len__(self) -> int:
        return len(self._turns)

    @property
    def turns(self) -> tuple[Turn, ...]:
        return tuple(self._turns)

    def compose_user_message(self, user_text: str) -> str:
        """User text for this turn, prefixed with an unanswered earlier utterance."""
        return f"{self._carry_over} {user_text}".strip()

    def record_turn(self, user_text: str, spoken_reply: str) -> None:
        """Store a finished or interrupted exchange.

        ``user_text`` must be the value returned by :meth:`compose_user_message`.
        If nothing was spoken (interrupted before audio), the user's words are
        carried into the next turn so the context is not lost.
        """
        user_text, spoken_reply = user_text.strip(), spoken_reply.strip()
        if not user_text:
            return
        if not spoken_reply:
            self._carry_over = user_text
            return
        self._carry_over = ""
        self._turns.append(Turn(user=user_text, assistant=spoken_reply))
        del self._turns[: -self._max_turns]

    def history(self) -> list[ChatMessage]:
        """Alternating user/assistant messages, oldest first."""
        messages: list[ChatMessage] = []
        for turn in self._turns:
            messages.append({"role": "user", "content": turn.user})
            messages.append({"role": "assistant", "content": turn.assistant})
        return messages

    def clear(self) -> None:
        self._turns.clear()
        self._carry_over = ""


def build_system_prompt(base_prompt: str, language: str | None) -> str:
    """Append an explicit reply-language rule to the base persona prompt.

    Small models follow "reply in the user's language" unreliably; naming the
    language (and script) the STT heard makes the reply match the speaker.
    """
    code = base_language(language)
    name = language_name(code)
    if code == "en":
        rule = "The user is speaking English. Reply in simple Indian English."
    else:
        rule = (
            f"The user is speaking {name}. Reply only in {name}, written in its "
            "native script, unless the user asks for another language."
        )
    return f"{base_prompt.strip()}\n\n{rule}"

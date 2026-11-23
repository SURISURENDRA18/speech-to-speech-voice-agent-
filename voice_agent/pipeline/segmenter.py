"""Turn a streaming LLM token feed into clean, speakable sentences.

TTS quality depends heavily on what text it receives. This module owns the
three rules that keep the voice from breaking:

1. Split only at real sentence ends (not ``2.5``, ``Dr.`` or ``e.g.``).
2. Strip text that is meant to be read, not spoken (markdown, emojis, think tags).
3. Group sentences so slow local TTS is not restarted for every sentence.
"""

import re

# Characters that end a sentence in the scripts we support (Latin + Indic danda).
_TERMINATORS = frozenset(".!?।॥")
# Closing marks that may follow a terminator and still belong to the sentence.
_CLOSERS = frozenset("\"'”’)]")
# Short Latin abbreviations that end with a period but do not end a sentence.
_ABBREVIATIONS = frozenset(
    {"dr", "mr", "mrs", "ms", "prof", "sr", "jr", "vs", "etc", "eg", "ie", "rs"}
)

_THINK_BLOCK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)
_MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MARKDOWN_SYMBOLS = re.compile(r"[*_`#>~|]+")
_LIST_BULLET = re.compile(r"^\s*(?:[-•]|\d+[.)])\s+", re.MULTILINE)
_EMOJI = re.compile(
    "["
    "\U0001f000-\U0001faff"  # pictographs, emoticons, symbols & pictographs
    "\U00002600-\U000027bf"  # misc symbols, dingbats
    "\U0000fe0f\U0000200d"  # variation selector, zero-width joiner
    "]+"
)
_WHITESPACE = re.compile(r"\s+")


def clean_for_speech(text: str) -> str:
    """Remove formatting a TTS engine would read aloud or choke on."""
    text = _THINK_BLOCK.sub(" ", text)
    text = _MARKDOWN_LINK.sub(r"\1", text)
    text = _LIST_BULLET.sub("", text)
    text = _MARKDOWN_SYMBOLS.sub(" ", text)
    text = _EMOJI.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def ends_with_terminator(text: str) -> bool:
    """Whether ``text`` ends like a finished sentence (ignoring closing quotes)."""
    stripped = text.rstrip().rstrip("".join(_CLOSERS))
    return bool(stripped) and stripped[-1] in _TERMINATORS


def _is_sentence_end(text: str, index: int) -> bool:
    """Whether the terminator at ``text[index]`` really ends a sentence.

    Requires the terminator to be followed by whitespace (so a token boundary
    in the middle of ``3.14`` is never treated as an end) and rejects common
    abbreviations and single-letter initials.
    """
    end = index + 1
    while end < len(text) and text[end] in _CLOSERS:
        end += 1
    if end >= len(text) or not text[end].isspace():
        return False
    if text[index] != ".":
        return True
    word = text[:index].rsplit(None, 1)[-1] if text[:index].strip() else ""
    word = word.lstrip("(\"'").lower().replace(".", "")
    return not (word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha()))


class SentenceSegmenter:
    """Incrementally split streamed text into complete sentences.

    Usage::

        segmenter = SentenceSegmenter()
        for delta in llm_stream:
            for sentence in segmenter.push(delta):
                speak(sentence)
        tail = segmenter.flush()  # unterminated remainder, may be ""
    """

    def __init__(self, max_chars: int = 240):
        # Very long unpunctuated runs are split at the last comma/space so the
        # first audio is not held back indefinitely.
        self._max_chars = max_chars
        self._buffer = ""

    @property
    def pending(self) -> str:
        """Text received but not yet emitted as a sentence."""
        return self._buffer

    def push(self, delta: str) -> list[str]:
        """Add a text delta and return any sentences it completed."""
        self._buffer += delta
        sentences: list[str] = []
        start = 0
        for i, ch in enumerate(self._buffer):
            if ch == "\n" or (ch in _TERMINATORS and _is_sentence_end(self._buffer, i)):
                end = i + 1
                while end < len(self._buffer) and self._buffer[end] in _CLOSERS:
                    end += 1
                self._emit(self._buffer[start:end], sentences)
                start = end
        self._buffer = self._buffer[start:]

        if len(self._buffer) > self._max_chars:
            comma = self._buffer.rfind(",", 0, self._max_chars)
            cut = comma if comma > self._max_chars // 2 else self._buffer.rfind(" ", 0, self._max_chars)
            if cut > 0:
                self._emit(self._buffer[: cut + 1], sentences)
                self._buffer = self._buffer[cut + 1 :]
        return sentences

    def flush(self) -> str:
        """Return and clear the unterminated remainder (cleaned for speech)."""
        tail, self._buffer = clean_for_speech(self._buffer), ""
        return tail

    @staticmethod
    def _emit(raw: str, out: list[str]) -> None:
        sentence = clean_for_speech(raw)
        # Skip fragments with nothing pronounceable (e.g. a lone "." or emoji).
        if any(ch.isalnum() for ch in sentence):
            out.append(sentence)


def group_for_synthesis(sentences: list[str], max_chars: int) -> list[str]:
    """Merge consecutive sentences into TTS requests of at most ``max_chars``.

    Each TTS request restarts the model's prosody and, on local Svara, costs a
    noticeable warm-up gap. Fewer, larger requests sound continuous. A single
    sentence longer than ``max_chars`` is kept whole rather than cut mid-word.
    """
    groups: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            groups.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        groups.append(current)
    return groups

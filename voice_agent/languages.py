"""Single source of truth for the language codes the agent understands."""

DEFAULT_LANGUAGE = "hi"

# ISO-639 style codes used by Whisper / Svara -> human readable names.
LANGUAGE_NAMES: dict[str, str] = {
    "as": "Assamese",
    "bn": "Bengali",
    "bh": "Bhojpuri",
    "brx": "Bodo",
    "doi": "Dogri",
    "en": "English (Indian)",
    "gu": "Gujarati",
    "hi": "Hindi",
    "hne": "Chhattisgarhi",
    "kn": "Kannada",
    "mag": "Magahi",
    "mai": "Maithili",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "ta": "Tamil",
    "te": "Telugu",
}


def base_language(code: str | None, default: str = DEFAULT_LANGUAGE) -> str:
    """Normalise ``"hi-IN"`` / ``"HI"`` / ``None`` to a bare lowercase code like ``"hi"``."""
    if not code:
        return default
    return code.strip().lower().replace("_", "-").split("-", 1)[0] or default


def language_name(code: str | None, default: str = "Hindi") -> str:
    """Human readable name for a language code, ``default`` when unknown."""
    return LANGUAGE_NAMES.get(base_language(code), default)

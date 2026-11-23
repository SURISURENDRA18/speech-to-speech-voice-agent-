"""Indic Text Normalization adapter with WFST and resilient rule-based fallback."""

import logging
import re
import sys

from voice_agent.config import settings

logger = logging.getLogger(__name__)

# Add upstream indic-text-normalization-main directory to sys.path if available
UPSTREAM_INDIC_DIR = settings.workspace_dir / "indic-text-normalization-main"
if UPSTREAM_INDIC_DIR.exists() and str(UPSTREAM_INDIC_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_INDIC_DIR))

# Hindi number maps
HINDI_DIGITS = {
    "0": "शून्य",
    "1": "एक",
    "2": "दो",
    "3": "तीन",
    "4": "चार",
    "5": "पाँच",
    "6": "छह",
    "7": "सात",
    "8": "आठ",
    "9": "नौ",
}

# Devanagari numerals
DEVANAGARI_DIGITS = {
    "०": "0",
    "१": "1",
    "२": "2",
    "३": "3",
    "४": "4",
    "५": "5",
    "६": "6",
    "७": "7",
    "८": "8",
    "९": "9",
}

HINDI_CARDINALS = {
    0: "शून्य",
    1: "एक",
    2: "दो",
    3: "तीन",
    4: "चार",
    5: "पाँच",
    6: "छह",
    7: "सात",
    8: "आठ",
    9: "नौ",
    10: "दस",
    11: "ग्यारह",
    12: "बारह",
    13: "तेरह",
    14: "चौदह",
    15: "पंद्रह",
    16: "सोलह",
    17: "सत्रह",
    18: "अठारह",
    19: "उन्नीस",
    20: "बीस",
    25: "पच्चीस",
    30: "तीस",
    40: "चालीस",
    50: "पचास",
    60: "साठ",
    70: "सत्तर",
    80: "अस्सी",
    90: "नब्बे",
    100: "एक सौ",
    1000: "एक हज़ार",
}

ENGLISH_CARDINALS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}


def _int_to_english(n: int) -> str:
    """Convert integer to English words (supports up to millions)."""
    if n in ENGLISH_CARDINALS:
        return ENGLISH_CARDINALS[n]
    if n < 100:
        tens = (n // 10) * 10
        rem = n % 10
        return f"{ENGLISH_CARDINALS[tens]} {ENGLISH_CARDINALS[rem]}"
    if n < 1000:
        hundreds = n // 100
        rem = n % 100
        rest = f" and {_int_to_english(rem)}" if rem else ""
        return f"{ENGLISH_CARDINALS[hundreds]} hundred{rest}"
    if n < 1000000:
        thousands = n // 1000
        rem = n % 1000
        rest = f" {_int_to_english(rem)}" if rem else ""
        return f"{_int_to_english(thousands)} thousand{rest}"
    return str(n)


def _int_to_hindi(n: int) -> str:
    """Convert integer to Hindi words."""
    if n in HINDI_CARDINALS:
        return HINDI_CARDINALS[n]
    if n < 100:
        # Fall back to digit reading if not in small lookup
        return " ".join(HINDI_DIGITS[d] for d in str(n))
    if n < 1000:
        hundreds = n // 100
        rem = n % 100
        rest = f" {_int_to_hindi(rem)}" if rem else ""
        return f"{HINDI_DIGITS[str(hundreds)]} सौ{rest}"
    if n < 100000:
        thousands = n // 1000
        rem = n % 1000
        rest = f" {_int_to_hindi(rem)}" if rem else ""
        return f"{_int_to_hindi(thousands)} हज़ार{rest}"
    if n < 10000000:
        lakhs = n // 100000
        rem = n % 100000
        rest = f" {_int_to_hindi(rem)}" if rem else ""
        return f"{_int_to_hindi(lakhs)} लाख{rest}"
    return " ".join(HINDI_DIGITS.get(d, d) for d in str(n))


class RuleBasedIndicNormalizer:
    """Fallback rule-based text normalizer for Indic languages and Indian English."""

    def __init__(self, lang: str = "hi"):
        self.lang = lang

    def normalize(self, text: str, lang: str | None = None) -> str:
        """Apply rule-based normalization on text."""
        target_lang = lang or self.lang
        result = text

        # 1. Normalize Devanagari digits to ASCII for uniform parsing
        for dev_digit, ascii_digit in DEVANAGARI_DIGITS.items():
            result = result.replace(dev_digit, ascii_digit)

        # 2. Currency normalization (e.g. ₹500, Rs. 500, Rs 500, $50)
        def replace_rupees(m: re.Match) -> str:
            val = int(m.group(1))
            if target_lang == "hi":
                return f"{_int_to_hindi(val)} रुपये"
            return f"{_int_to_english(val)} rupees"

        result = re.sub(r"(?:₹|Rs\.?|INR)\s*(\d+)", replace_rupees, result, flags=re.IGNORECASE)

        def replace_dollars(m: re.Match) -> str:
            val = int(m.group(1))
            if target_lang == "hi":
                return f"{_int_to_hindi(val)} डॉलर"
            return f"{_int_to_english(val)} dollars"

        result = re.sub(r"\$\s*(\d+)", replace_dollars, result)

        # 3. Percentages (e.g. 50%)
        def replace_percent(m: re.Match) -> str:
            val = int(m.group(1))
            if target_lang == "hi":
                return f"{_int_to_hindi(val)} प्रतिशत"
            return f"{_int_to_english(val)} percent"

        result = re.sub(r"(\d+)%", replace_percent, result)

        # 4. Decimals (e.g. 3.14)
        def replace_decimal(m: re.Match) -> str:
            whole = int(m.group(1))
            fraction = m.group(2)
            if target_lang == "hi":
                frac_words = " ".join(HINDI_DIGITS.get(d, d) for d in fraction)
                return f"{_int_to_hindi(whole)} दशमलव {frac_words}"
            frac_words = " ".join(ENGLISH_CARDINALS.get(int(d), d) for d in fraction)
            return f"{_int_to_english(whole)} point {frac_words}"

        result = re.sub(r"(\d+)\.(\d+)", replace_decimal, result)

        # 5. Ordinals (1st, 2nd, 3rd, 4th, etc.)
        def replace_ordinal_en(m: re.Match) -> str:
            val = int(m.group(1))
            special = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}
            return special.get(val, f"{_int_to_english(val)}th")

        def replace_ordinal_hi(m: re.Match) -> str:
            val = int(m.group(1))
            special = {1: "पहला", 2: "दूसरा", 3: "तीसरा", 4: "चौथा", 5: "पाँचवाँ"}
            return special.get(val, f"{_int_to_hindi(val)}वाँ")

        if target_lang == "hi":
            result = re.sub(r"(\d+)(?:st|nd|rd|th)", replace_ordinal_hi, result, flags=re.IGNORECASE)
        else:
            result = re.sub(r"(\d+)(?:st|nd|rd|th)", replace_ordinal_en, result, flags=re.IGNORECASE)

        # 6. Cardinal numbers
        def replace_cardinal(m: re.Match) -> str:
            val = int(m.group(0))
            if target_lang == "hi":
                return _int_to_hindi(val)
            return _int_to_english(val)

        result = re.sub(r"\b\d+\b", replace_cardinal, result)

        # 7. Clean excessive whitespace
        result = re.sub(r"\s+", " ", result).strip()
        return result


class IndicTextNormalizer:
    """Unified Indic text normalizer.

    Uses upstream WFST-based `indic_text_normalization.Normalizer` when Pynini is present,
    or falls back automatically to RuleBasedIndicNormalizer on platforms where Pynini
    cannot be compiled.
    """

    def __init__(self, default_lang: str = "hi", use_wfst: bool = True):
        self.default_lang = default_lang
        self.use_wfst = use_wfst
        self._wfst_normalizers: dict[str, object] = {}
        self._fallback_normalizer = RuleBasedIndicNormalizer(lang=default_lang)
        self.is_wfst_available = False

        if use_wfst:
            try:
                import pynini  # noqa: F401
                from indic_text_normalization.normalize import Normalizer

                self._normalizer_cls = Normalizer
                self.is_wfst_available = True
                logger.info("WFST Indic Text Normalizer (Pynini) initialized successfully.")
            except (ImportError, ModuleNotFoundError, RuntimeError, OSError) as exc:
                self.is_wfst_available = False
                logger.info(
                    "Pynini / WFST Indic normalizer not available (%s). "
                    "Using built-in rule-based Indic normalizer fallback.",
                    exc,
                )

    def normalize(self, text: str, lang: str | None = None) -> str:
        """Normalize semiotic classes in text for the given language."""
        if not text:
            return ""

        target_lang = lang or self.default_lang

        if self.is_wfst_available:
            try:
                if target_lang not in self._wfst_normalizers:
                    self._wfst_normalizers[target_lang] = self._normalizer_cls(
                        input_case="cased", lang=target_lang
                    )
                normalizer = self._wfst_normalizers[target_lang]
                return normalizer.normalize(text)
            except (RuntimeError, ValueError, TypeError, KeyError) as e:
                logger.warning(
                    "WFST normalization failed for lang '%s' (%s). Using fallback.",
                    target_lang,
                    e,
                )

        return self._fallback_normalizer.normalize(text, lang=target_lang)

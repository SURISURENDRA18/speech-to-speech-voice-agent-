"""Unit tests for Indic text normalization."""

from voice_agent.normalization.indic_norm import (
    IndicTextNormalizer,
    RuleBasedIndicNormalizer,
)


def test_rule_based_hindi_cardinals():
    norm = RuleBasedIndicNormalizer(lang="hi")
    assert norm.normalize("1") == "एक"
    assert norm.normalize("10") == "दस"
    assert norm.normalize("100") == "एक सौ"
    assert norm.normalize("1000") == "एक हज़ार"


def test_rule_based_english_cardinals():
    norm = RuleBasedIndicNormalizer(lang="en")
    assert norm.normalize("1") == "one"
    assert norm.normalize("25") == "twenty five"
    assert norm.normalize("100") == "one hundred"
    assert norm.normalize("1500") == "one thousand five hundred"


def test_currency_normalization():
    norm_hi = RuleBasedIndicNormalizer(lang="hi")
    res_hi = norm_hi.normalize("कुल कीमत ₹500 है")
    assert "पाँच सौ रुपये" in res_hi

    norm_en = RuleBasedIndicNormalizer(lang="en")
    res_en = norm_en.normalize("Price is Rs. 500")
    assert "five hundred rupees" in res_en


def test_percentage_normalization():
    norm_hi = RuleBasedIndicNormalizer(lang="hi")
    assert "पचास प्रतिशत" in norm_hi.normalize("सफलता दर 50% है")

    norm_en = RuleBasedIndicNormalizer(lang="en")
    assert "fifty percent" in norm_en.normalize("Success rate is 50%")


def test_decimal_normalization():
    norm_hi = RuleBasedIndicNormalizer(lang="hi")
    res_hi = norm_hi.normalize("पाई का मान 3.14 है")
    assert "तीन दशमलव एक चार" in res_hi

    norm_en = RuleBasedIndicNormalizer(lang="en")
    res_en = norm_en.normalize("Value is 3.14")
    assert "three point one four" in res_en


def test_ordinal_normalization():
    norm_en = RuleBasedIndicNormalizer(lang="en")
    assert norm_en.normalize("1st prize") == "first prize"
    assert norm_en.normalize("2nd place") == "second place"

    norm_hi = RuleBasedIndicNormalizer(lang="hi")
    assert norm_hi.normalize("1st पुरस्कार") == "पहला पुरस्कार"


def test_devanagari_digits():
    norm_hi = RuleBasedIndicNormalizer(lang="hi")
    res = norm_hi.normalize("संख्या १००")
    assert "एक सौ" in res


def test_indic_text_normalizer_adapter():
    normalizer = IndicTextNormalizer(default_lang="hi", use_wfst=False)
    assert normalizer.is_wfst_available is False
    res = normalizer.normalize("₹100 का भुगतान करें", lang="hi")
    assert "एक सौ रुपये" in res

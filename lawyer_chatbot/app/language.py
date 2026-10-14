"""Deterministic language detection shared by text and voice features."""

from __future__ import annotations

import re

from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0

LANGUAGE_NAMES = {"en": "English", "ur": "Urdu", "hi": "Hindi", "ps": "Pashto"}
TTS_LOCALES = {"en": "en-US", "ur": "ur-PK", "hi": "hi-IN", "ps": "ps-AF"}
_DEVANAGARI = re.compile(r"[\u0900-\u097f]")
_ARABIC = re.compile(r"[\u0600-\u06ff]")
_PASHTO_UNIQUE_CHARS = set("ځڅښګڼۍ")
_URDU_UNIQUE_CHARS = set("ٹڈڑںےھ")
_URDU_COMMON_WORDS = {"ہے", "ہیں", "تھا", "تھی", "تھے", "کے", "کی", "کا", "کو", "میں", "سے", "نے", "پر", "ہو", "ہوا", "ہوئی", "ہوئے", "اور", "کر", "کیا", "یہ", "وہ", "اگر", "تو", "پاس"}
_PASHTO_WORDS = {"زه", "موږ", "کوي", "کولای", "شوي", "شوې", "څنګه", "ولې", "ستاسو", "پوښتنه", "وشي"}


def detect_language(text: str, fallback: str = "en") -> str:
    """Return one of the supported ISO language codes without a network call."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return fallback
    if _DEVANAGARI.search(cleaned):
        return "hi"
    if _ARABIC.search(cleaned):
        words = set(re.findall(r"[\u0600-\u06ff]+", cleaned))
        # Prioritize Urdu if common Urdu markers/characters exist without unique Pashto letters
        has_urdu_chars = any(char in _URDU_UNIQUE_CHARS for char in cleaned)
        has_urdu_words = bool(words.intersection(_URDU_COMMON_WORDS))
        has_pashto_chars = any(char in _PASHTO_UNIQUE_CHARS for char in cleaned)
        has_pashto_words = bool(words.intersection(_PASHTO_WORDS))

        if (has_urdu_chars or has_urdu_words) and not has_pashto_chars:
            return "ur"
        if has_pashto_chars or has_pashto_words:
            return "ps"
        if has_urdu_chars or has_urdu_words:
            return "ur"
        try:
            return "ps" if detect(cleaned) == "ps" else "ur"
        except LangDetectException:
            return "ur"
    try:
        guessed = detect(cleaned)
    except LangDetectException:
        return fallback
    return guessed if guessed in LANGUAGE_NAMES else fallback


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, LANGUAGE_NAMES["en"])

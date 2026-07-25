"""Deterministic language detection shared by text and voice features."""

from __future__ import annotations

import re

from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0

LANGUAGE_NAMES = {"en": "English", "ur": "Urdu", "hi": "Hindi", "ps": "Pashto"}
TTS_LOCALES = {"en": "en-US", "ur": "ur-PK", "hi": "hi-IN", "ps": "ps-AF"}
_DEVANAGARI = re.compile(r"[\u0900-\u097f]")
_ARABIC = re.compile(r"[\u0600-\u06ff]")
_PASHTO_CHARS = set("ځڅډړږښګڼېۍ")
_URDU_CHARS = set("ٹڈڑںھہےکگچپژ")
_PASHTO_WORDS = {"زه", "ته", "دی", "ده", "دي", "وشي", "شوی", "شوې", "څه", "څنګه", "کوم"}


def detect_language(text: str, fallback: str = "en") -> str:
    """Return one of the supported ISO language codes without a network call."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return fallback
    if _DEVANAGARI.search(cleaned):
        return "hi"
    if _ARABIC.search(cleaned):
        words = set(re.findall(r"[\u0600-\u06ff]+", cleaned))
        if any(char in _PASHTO_CHARS for char in cleaned) or words.intersection(_PASHTO_WORDS):
            return "ps"
        if any(char in _URDU_CHARS for char in cleaned):
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

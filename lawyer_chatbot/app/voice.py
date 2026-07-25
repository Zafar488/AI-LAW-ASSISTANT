"""Speech-to-text service with a provider-neutral interface."""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass

from groq import Groq

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Transcript:
    text: str
    provider: str
    model: str


class SpeechToTextService:
    """Transcribe user recordings in memory; recordings are never persisted."""

    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or os.getenv("STT_PROVIDER", "groq")).lower()
        self.model = model or os.getenv("STT_MODEL", "whisper-large-v3-turbo")
        if self.provider != "groq":
            raise ValueError(f"Unsupported STT provider: {self.provider}")
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not configured")
        self.client = Groq(api_key=key)

    def transcribe(self, content: bytes, filename: str = "recording.wav") -> Transcript:
        if not content:
            raise ValueError("The recording is empty")
        if len(content) > 25 * 1024 * 1024:
            raise ValueError("The recording exceeds the 25 MB limit")
        audio = io.BytesIO(content)
        audio.name = filename
        try:
            result = self.client.audio.transcriptions.create(
                file=(filename, audio.read()),
                model=self.model,
                response_format="json",
                temperature=0.0,
            )
        except Exception:
            LOGGER.exception("Speech transcription failed")
            raise RuntimeError("Speech recognition failed. Please try again.") from None
        text = (getattr(result, "text", "") or "").strip()
        if not text:
            raise RuntimeError("I could not hear that clearly. Please repeat.")
        return Transcript(text=text, provider=self.provider, model=self.model)


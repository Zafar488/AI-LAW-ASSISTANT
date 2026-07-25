import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.voice import SpeechToTextService


class VoiceServiceTests(unittest.TestCase):
    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("app.voice.Groq")
    def test_transcription_returns_provider_metadata(self, groq_class):
        client = Mock()
        client.audio.transcriptions.create.return_value = SimpleNamespace(text="Hello")
        groq_class.return_value = client

        result = SpeechToTextService().transcribe(b"fake audio", "voice.wav")

        self.assertEqual(result.text, "Hello")
        self.assertEqual(result.provider, "groq")
        self.assertEqual(result.model, "whisper-large-v3-turbo")

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("app.voice.Groq")
    def test_empty_audio_is_rejected(self, groq_class):
        groq_class.return_value = Mock()
        with self.assertRaisesRegex(ValueError, "empty"):
            SpeechToTextService().transcribe(b"")

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
    @patch("app.voice.Groq")
    def test_provider_failure_has_safe_message(self, groq_class):
        client = Mock()
        client.audio.transcriptions.create.side_effect = RuntimeError("provider secret")
        groq_class.return_value = client
        with self.assertRaisesRegex(RuntimeError, "Please try again"):
            SpeechToTextService().transcribe(b"fake audio")


if __name__ == "__main__":
    unittest.main()

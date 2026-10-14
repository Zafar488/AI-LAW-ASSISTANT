import io
import unittest
from unittest.mock import patch
from starlette.testclient import TestClient

from api.server import app, API_KEY


class TestApiSecurityAndResourceLimits(unittest.TestCase):
    """Verifies API authentication, payload size enforcement, and audio format validation
    without contacting Groq or any external third-party service.
    """

    def setUp(self):
        self.client = TestClient(app)
        self.valid_key = API_KEY or "test_key"

    def test_health_endpoint_public_access(self):
        """Health endpoint should be accessible without authentication."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("knowledge_base_ready", data)
        self.assertIn("chunk_count", data)

    def test_ask_endpoint_missing_api_key(self):
        """Missing API key header should return 401 Unauthorized."""
        response = self.client.post("/ask", json={"message": "What is Section 302?"})
        self.assertEqual(response.status_code, 401)
        self.assertIn("Invalid or missing API Key", response.json()["detail"])

    def test_ask_endpoint_invalid_api_key(self):
        """Invalid API key header should return 401 Unauthorized."""
        response = self.client.post(
            "/ask",
            headers={"key": "wrong_invalid_key_123"},
            json={"message": "What is Section 302?"}
        )
        self.assertEqual(response.status_code, 401)
        self.assertIn("Invalid or missing API Key", response.json()["detail"])

    def test_transcribe_missing_api_key(self):
        """Transcribe endpoint must reject requests with missing API key."""
        fake_wav = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 20
        response = self.client.post(
            "/transcribe",
            files={"audio": ("test.wav", io.BytesIO(fake_wav), "audio/wav")}
        )
        self.assertEqual(response.status_code, 401)

    def test_transcribe_invalid_api_key(self):
        """Transcribe endpoint must reject requests with invalid API key."""
        fake_wav = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 20
        response = self.client.post(
            "/transcribe",
            headers={"key": "invalid_api_token"},
            files={"audio": ("test.wav", io.BytesIO(fake_wav), "audio/wav")}
        )
        self.assertEqual(response.status_code, 401)

    def test_transcribe_oversized_audio_returns_413(self):
        """Audio payload exceeding MAX_AUDIO_SIZE_BYTES must be rejected with 413 Payload Too Large."""
        # Use a small limit threshold for the test to avoid allocating 10MB memory
        with patch("api.server.MAX_AUDIO_SIZE_BYTES", 512):
            oversized_data = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 1024  # 1032 bytes > 512
            response = self.client.post(
                "/transcribe",
                headers={"key": self.valid_key},
                files={"audio": ("test.wav", io.BytesIO(oversized_data), "audio/wav")}
            )
            self.assertEqual(response.status_code, 413)
            self.assertIn("exceeds maximum permitted size", response.json()["detail"])

    def test_transcribe_invalid_audio_signature_returns_400(self):
        """Upload with invalid binary signature (e.g. text/exe pretending to be wav) must be rejected with 400."""
        fake_text_file = b"This is not a real audio file at all. Just plain text."
        response = self.client.post(
            "/transcribe",
            headers={"key": self.valid_key},
            files={"audio": ("fake.wav", io.BytesIO(fake_text_file), "audio/wav")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("validation failed", response.json()["detail"])

    def test_transcribe_unsupported_extension_returns_400(self):
        """Upload with unsupported file extension must return 400."""
        some_bytes = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 20
        response = self.client.post(
            "/transcribe",
            headers={"key": self.valid_key},
            files={"audio": ("exploit.exe", io.BytesIO(some_bytes), "application/octet-stream")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported audio file extension", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()

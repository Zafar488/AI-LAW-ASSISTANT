import json
import unittest
from unittest.mock import MagicMock, patch

from starlette.testclient import TestClient

from api.server import app
from app.chatbot import LegalChatbot


class TestChatbotStreaming(unittest.TestCase):
    def setUp(self):
        self.mock_retriever = MagicMock()
        self.mock_retriever.retrieve.return_value = [
            {
                "id": "chunk_01",
                "document_title": "Pakistan Penal Code",
                "section": "Section 489-F",
                "page": 112,
                "source": "Pakistan Penal Code.pdf",
                "text": "Whoever dishonestly issues a cheque...",
                "hybrid_score": 0.85,
            }
        ]
        self.mock_client = MagicMock()

    def test_ask_stream_yields_tokens_and_sources(self):
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock(delta=MagicMock(content="Section 489-F ", reasoning=None))]
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock(delta=MagicMock(content="deals with cheque dishonour.", reasoning=None))]

        self.mock_client.chat.completions.create.return_value = iter([chunk1, chunk2])

        bot = LegalChatbot(retriever=self.mock_retriever, client=self.mock_client)
        token_gen, sources = bot.ask_stream("What is 489-F PPC?")

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["section"], "Section 489-F")

        tokens = list(token_gen)
        self.assertEqual(tokens, ["Section 489-F ", "deals with cheque dishonour."])
        self.assertEqual("".join(tokens), "Section 489-F deals with cheque dishonour.")

    def test_ask_stream_empty_query(self):
        bot = LegalChatbot(retriever=self.mock_retriever, client=self.mock_client)
        token_gen, sources = bot.ask_stream("   ")
        self.assertEqual(sources, [])
        tokens = list(token_gen)
        self.assertIn("valid legal question", "".join(tokens))

    @patch("api.server.get_bot")
    def test_api_ask_stream_endpoint(self, mock_get_bot):
        mock_bot = MagicMock()
        mock_bot.ask_stream.return_value = (
            iter(["First token ", "Second token."]),
            [{"id": "c1", "document": "PPC", "section": "302"}],
        )
        mock_get_bot.return_value = mock_bot

        client = TestClient(app)
        response = client.post(
            "/ask-stream",
            headers={"key": "zafar123-secret-key"},
            json={"message": "Section 302 PPC penalty", "top_k": 3},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        body = response.text
        self.assertIn('"type": "sources"', body)
        self.assertIn('"type": "token"', body)
        self.assertIn('"type": "done"', body)
        self.assertIn("[DONE]", body)


if __name__ == "__main__":
    unittest.main()

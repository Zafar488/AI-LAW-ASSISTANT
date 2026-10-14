import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.database import (
    get_conversation_count,
    get_recent_conversations,
    init_db,
    save_conversation,
    search_conversations,
)


class TestConversationDatabase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_db_path = Path(self.temp_dir.name) / "test_conversations.db"
        self.patcher = patch("app.database.DB_PATH", self.test_db_path)
        self.patcher.start()
        init_db()

    def tearDown(self):
        self.patcher.stop()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_init_db_creates_tables(self):
        self.assertTrue(self.test_db_path.exists())
        conn = sqlite3.connect(str(self.test_db_path))
        try:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            self.assertIn("conversations", tables)
        finally:
            conn.close()

    def test_save_and_retrieve_conversation(self):
        sources = [
            {"id": "chunk_01", "document": "Pakistan Penal Code", "section": "Section 489-F", "page": 112}
        ]
        conv_id = save_conversation(
            user_query="اگر چیک باؤنس ہو جائے تو کیا سزا ہے؟",
            bot_answer="سیکشن 489-F لاگو ہوگی اور 3 سال تک قید ہو سکتی ہے۔",
            sources=sources,
            session_id="sess_123",
            detected_language="ur",
            top_k=5,
            latency_sec=2.45,
            client_ip="127.0.0.1",
        )
        self.assertGreater(conv_id, 0)

        # Check total count
        self.assertEqual(get_conversation_count(), 1)

        # Retrieve and verify
        recent = get_recent_conversations(limit=10)
        self.assertEqual(len(recent), 1)
        item = recent[0]
        self.assertEqual(item["id"], conv_id)
        self.assertEqual(item["user_query"], "اگر چیک باؤنس ہو جائے تو کیا سزا ہے؟")
        self.assertEqual(item["session_id"], "sess_123")
        self.assertEqual(item["detected_language"], "ur")
        self.assertEqual(item["top_k"], 5)
        self.assertEqual(len(item["sources"]), 1)
        self.assertEqual(item["sources"][0]["section"], "Section 489-F")

    def test_search_conversations(self):
        save_conversation(
            user_query="سٹے آرڈر کی خلاف ورزی",
            bot_answer="آرڈر 39 رول 2(3) کے تحت کارروائی ہوگی۔",
            session_id="sess_a",
        )
        save_conversation(
            user_query="خاندان کا نان نفقہ",
            bot_answer="فیملی کورٹس ایکٹ 1964 لاگو ہوگا۔",
            session_id="sess_b",
        )

        hits = search_conversations("خلاف ورزی")
        self.assertEqual(len(hits), 1)
        self.assertIn("سٹے آرڈر", hits[0]["user_query"])

        hits2 = search_conversations("فیملی کورٹس")
        self.assertEqual(len(hits2), 1)
        self.assertIn("نان نفقہ", hits2[0]["user_query"])


if __name__ == "__main__":
    unittest.main()

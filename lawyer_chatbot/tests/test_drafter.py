import io
import unittest
from unittest.mock import MagicMock, patch

from docx import Document
from starlette.testclient import TestClient

from api.server import app
from app.drafter import DOCUMENT_TYPES, LegalDrafter


class TestLegalDrafter(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.drafter = LegalDrafter(client=self.mock_client)

    def test_build_prompt_english(self):
        prompt = self.drafter.build_drafting_prompt(
            doc_type="bail_application",
            case_facts="Accused was falsely implicated in FIR 123.",
            parties={"petitioner": "Muhammad Ali", "respondent": "The State"},
            court_info={"court_name": "In the Court of Sessions Judge", "city": "Lahore", "fir_no": "123/2026"},
            language="en",
        )
        self.assertIn("IN THE COURT OF", prompt)
        self.assertIn("Muhammad Ali", prompt)
        self.assertIn("123/2026", prompt)
        self.assertIn("GROUNDS", prompt)

    def test_build_prompt_urdu(self):
        prompt = self.drafter.build_drafting_prompt(
            doc_type="legal_notice",
            case_facts="پانچ لاکھ کا چیک باؤنس ہو گیا۔",
            parties={"petitioner": "عثمان", "respondent": "کامران"},
            court_info={"court_name": "قانونی نوٹس", "city": "اسلام آباد", "amount": "500,000"},
            language="ur",
        )
        self.assertIn("قانونی درخواست / نوٹس", prompt)
        self.assertIn("عثمان", prompt)
        self.assertIn("کامران", prompt)
        self.assertIn("500,000", prompt)

    def test_export_to_docx(self):
        sample_pleading = """
IN THE COURT OF SESSIONS JUDGE, LAHORE
Muhammad Ali VERSUS The State

APPLICATION UNDER SECTION 497 Cr.P.C.

1. That the petitioner is innocent.
2. That there is no independent witness.

PRAYER:
It is prayed that bail may be granted.
"""
        docx_bytes = self.drafter.export_to_docx(
            content=sample_pleading,
            title="Bail Application",
            court_name="In the Court of Sessions Judge",
            city="Lahore",
        )
        self.assertIsInstance(docx_bytes, bytes)
        self.assertGreater(len(docx_bytes), 1000)

        # Verify it can be read back by python-docx
        doc = Document(io.BytesIO(docx_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text]
        self.assertTrue(any("IN THE COURT" in p for p in paragraphs))
        self.assertTrue(any("Muhammad Ali" in p for p in paragraphs))

    @patch("api.server.get_drafter")
    def test_api_draft_endpoints(self, mock_get_drafter):
        mock_engine = MagicMock()
        mock_engine.generate_draft.return_value = "COURT OF SESSIONS JUDGE: BAIL APPLICATION DRAFT"
        mock_engine.export_to_docx.return_value = b"PK\x03\x04mock_docx_bytes"
        mock_get_drafter.return_value = mock_engine

        client = TestClient(app)

        # 1. Test /draft/types
        resp_types = client.get("/draft/types")
        self.assertEqual(resp_types.status_code, 200)
        self.assertIn("bail_application", resp_types.json()["types"])

        # 2. Test /draft
        resp_draft = client.post(
            "/draft",
            headers={"key": "zafar123-secret-key"},
            json={
                "doc_type": "bail_application",
                "case_facts": "The accused was falsely implicated in dispute.",
                "parties": {"petitioner": "Ali", "respondent": "State"},
                "court_info": {"court_name": "Sessions Court", "city": "Rawalpindi"},
                "language": "en",
            },
        )
        self.assertEqual(resp_draft.status_code, 200)
        data = resp_draft.json()
        self.assertTrue(data["success"])
        self.assertIn("BAIL APPLICATION DRAFT", data["draft"])

        # 3. Test /draft/docx
        resp_docx = client.post(
            "/draft/docx",
            headers={"key": "zafar123-secret-key"},
            json={
                "content": "Sample Court Pleading Text",
                "title": "Bail Application",
                "court_name": "Sessions Court",
                "city": "Rawalpindi",
            },
        )
        self.assertEqual(resp_docx.status_code, 200)
        self.assertIn("wordprocessingml.document", resp_docx.headers["content-type"])
        self.assertEqual(resp_docx.content, b"PK\x03\x04mock_docx_bytes")


if __name__ == "__main__":
    unittest.main()

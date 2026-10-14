"""Unit tests for Legal Document & FIR OCR Scanner and API endpoints."""

import io
import unittest
from unittest.mock import MagicMock, patch

import fitz
from PIL import Image, ImageDraw
from starlette.testclient import TestClient

from api.server import app
from app.scanner import LegalDocumentScanner


class TestLegalScanner(unittest.TestCase):
    """Test suite for LegalDocumentScanner."""

    def setUp(self):
        self.scanner = LegalDocumentScanner(groq_api_key="mock_key")
        self.client = TestClient(app)
        self.api_key = "zafar123-secret-key"

    def test_extract_text_from_digital_pdf(self):
        """Test digital text extraction from an in-memory PDF document."""
        doc = fitz.open()
        page = doc.new_page()
        test_content = (
            "FIRST INFORMATION REPORT (FIR No. 99/2026)\n"
            "Police Station Civil Lines, Lahore\n"
            "Under Section 379 / 411 PPC\n"
            "Complainant: Aslam Khan\n"
            "Accused: Unknown persons"
        )
        page.insert_text((50, 72), test_content, fontsize=12)
        pdf_bytes = doc.write()
        doc.close()

        result = self.scanner.extract_text_from_bytes(pdf_bytes, "fir_99.pdf", max_pages=1)
        self.assertIn("digital", result["method"])
        self.assertEqual(result["page_count"], 1)
        self.assertIn("FIRST INFORMATION REPORT", result["extracted_text"])
        self.assertIn("Civil Lines", result["extracted_text"])

    def test_extract_text_from_image(self):
        """Test OCR extraction on an in-memory PIL image."""
        img = Image.new("RGB", (300, 100), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((20, 30), "POLICE STATION", fill=(0, 0, 0))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        result = self.scanner.extract_text_from_bytes(png_bytes, "document.png")
        self.assertEqual(result["method"], "ocr")
        self.assertEqual(result["page_count"], 1)

    def test_build_analysis_prompt_languages(self):
        """Test prompt construction with different language instructions."""
        prompt_both = self.scanner.build_analysis_prompt("FIR under 302 PPC", language="both")
        self.assertIn("bilingual format", prompt_both)
        self.assertIn("FIR under 302 PPC", prompt_both)

        prompt_ur = self.scanner.build_analysis_prompt("FIR under 302 PPC", language="ur")
        self.assertIn("عدالتی اردو", prompt_ur)

        prompt_en = self.scanner.build_analysis_prompt("FIR under 302 PPC", language="en")
        self.assertIn("formal legal English", prompt_en)

    def test_export_analysis_docx(self):
        """Test Microsoft Word export byte generation and signature."""
        markdown_sample = (
            "### 1. DOCUMENT IDENTIFICATION\n"
            "- Document Type: FIR\n"
            "- Police Station: Civil Lines\n\n"
            "### 2. STATUTORY SECTIONS\n"
            "- 379 PPC: Theft\n"
        )
        raw_text = "Raw FIR OCR text line 1\nRaw FIR OCR text line 2"
        docx_bytes = self.scanner.export_analysis_docx(markdown_sample, raw_text, "fir_sample.pdf")

        # Standard ZIP magic number for docx (.openxml)
        self.assertTrue(docx_bytes.startswith(b"PK\x03\x04"))
        self.assertGreater(len(docx_bytes), 5000)

    def test_api_scan_upload_unauthorized(self):
        """Endpoint /scan/upload must reject requests without valid API key."""
        resp = self.client.post("/scan/upload", files={"file": ("test.pdf", b"%PDF-1.4 mock content", "application/pdf")})
        self.assertEqual(resp.status_code, 401)

    def test_api_scan_upload_unsupported_ext(self):
        """Endpoint /scan/upload must reject unsupported file extensions."""
        resp = self.client.post(
            "/scan/upload",
            files={"file": ("malicious.exe", b"binarycontent", "application/octet-stream")},
            headers={"key": self.api_key},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unsupported file extension", resp.json()["detail"])

    @patch.object(LegalDocumentScanner, "analyze_document", return_value="### Mock Legal Audit Report")
    def test_api_scan_upload_success(self, mock_analyze):
        """Endpoint /scan/upload successfully returns extracted text and analysis."""
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 72), "FIR No 12/2026 PS Model Town Lahore under section 379 PPC", fontsize=12)
        pdf_bytes = doc.write()
        doc.close()

        resp = self.client.post(
            "/scan/upload",
            files={"file": ("fir_test.pdf", pdf_bytes, "application/pdf")},
            data={"language": "both", "max_pages": 1},
            headers={"key": self.api_key},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["filename"], "fir_test.pdf")
        self.assertIn("extraction", data)
        self.assertIn("Mock Legal Audit Report", data["analysis"])

    def test_is_sentence_complete(self):
        """Test sentence completion and terminal punctuation detector."""
        self.assertTrue(LegalDocumentScanner._is_sentence_complete("This is a complete sentence."))
        self.assertTrue(LegalDocumentScanner._is_sentence_complete("یہ ایک مکمل جملہ ہے۔"))
        self.assertTrue(LegalDocumentScanner._is_sentence_complete("Completed point!**"))
        self.assertFalse(LegalDocumentScanner._is_sentence_complete("Critical Note: This excerpt does not contain the substantive definitions"))
        self.assertFalse(LegalDocumentScanner._is_sentence_complete("2. **مضمون نمبروں کی تصدیق کریں:**"))

    def test_ensure_report_completion_repairs_missing_sections(self):
        """Test that _ensure_report_completion guarantees sections 5 & 6 and complete sentences."""
        incomplete_report = (
            "### 1. DOCUMENT IDENTIFICATION & SCOPE AUDIT\nDetails here.\n\n"
            "### 2. CHAPTER-WISE STATUTORY BREAKDOWN\nTable here.\n\n"
            "### 3. CRUCIAL TIMELINE & LIMITATION DATES\nTimeline here.\n\n"
            "### 4. EXECUTIVE FACTUAL SUMMARY\nSummary here.\n"
            "Critical Note: This excerpt does not contain the substantive definitions"
        )
        completed = self.scanner._ensure_report_completion(
            report=incomplete_report,
            client=None,
            model="mock_model",
            language="both",
        )
        self.assertTrue(LegalDocumentScanner._is_sentence_complete(completed))
        self.assertIn("5. LEGAL AUDIT", completed)
        self.assertIn("6. RECOMMENDED ACTIONABLE LEGAL STRATEGY", completed)
        self.assertIn("Section 22 ('Movable property')", completed)


if __name__ == "__main__":
    unittest.main()

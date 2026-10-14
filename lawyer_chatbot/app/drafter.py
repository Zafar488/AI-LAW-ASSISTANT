"""Pakistani Court Pleading & Legal Notice Drafter.

Generates court-standard pleadings, bail applications, 22-A/22-B CrPC petitions,
and statutory legal notices with direct Microsoft Word (.docx) export.
"""

from __future__ import annotations

import io
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from dotenv import load_dotenv
from groq import Groq

from app.config import BASE_DIR

load_dotenv(BASE_DIR / ".env", override=True)

LOGGER = logging.getLogger("legalbot.drafter")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
FALLBACK_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
]

DOCUMENT_TYPES = {
    "bail_application": {
        "title": "Bail Application (Under Section 497 / 498 CrPC)",
        "law": "Code of Criminal Procedure, 1898",
        "description": "Formal bail petition for Sessions Court or High Court with statutory grounds and prayer.",
    },
    "legal_notice": {
        "title": "14-Day Statutory Legal Notice (Cheque Dishonour / Contract Breach)",
        "law": "Section 489-F PPC & Contract Act, 1872",
        "description": "Pre-litigation legal notice served through counsel with 14-day compliance period.",
    },
    "petition_22a": {
        "title": "Petition Under Section 22-A / 22-B CrPC (Registration of FIR)",
        "law": "Code of Criminal Procedure, 1898",
        "description": "Petition to Ex-Officio Justice of the Peace (Sessions Judge) directing police to lodge FIR.",
    },
    "stay_application": {
        "title": "Application for Temporary Injunction (Order XXXIX Rules 1 & 2 CPC)",
        "law": "Code of Civil Procedure, 1908",
        "description": "Application for interim stay order to restrain dispossession, alienation, or construction.",
    },
    "custom_pleading": {
        "title": "Custom Court Pleading / Written Statement",
        "law": "Pakistan Civil / Criminal Codes",
        "description": "General legal pleading tailored to custom dispute facts.",
    },
}


class LegalDrafter:
    """Enterprise legal pleading and court notice generation engine."""

    def __init__(self, client: Any | None = None):
        if client is not None:
            self.client = client
        else:
            try:
                self.client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
            except Exception as e:
                LOGGER.warning("Groq client initialization failed in drafter: %s", e)
                self.client = None

    def build_drafting_prompt(
        self,
        doc_type: str,
        case_facts: str,
        parties: dict[str, str],
        court_info: dict[str, str],
        language: str = "en",
    ) -> str:
        """Construct court-standard drafting prompt with strict Pakistani pleading etiquette."""
        doc_meta = DOCUMENT_TYPES.get(doc_type, DOCUMENT_TYPES["custom_pleading"])
        petitioner = parties.get("petitioner", "Complainant / Petitioner")
        respondent = parties.get("respondent", "Respondent(s) / Accused")
        court_name = court_info.get("court_name", "In the Court of Sessions Judge")
        city = court_info.get("city", "Lahore / Islamabad")
        fir_no = court_info.get("fir_no", "")
        police_station = court_info.get("police_station", "")
        sections = court_info.get("sections", "")
        amount = court_info.get("amount", "")
        cheque_no = court_info.get("cheque_no", "")

        is_urdu = language.lower() in ("ur", "urdu")

        if is_urdu:
            prompt = f"""
آپ ایک سینئر وکیل (Advocate High Court / Supreme Court of Pakistan) ہیں۔
آپ کا کام پاکستانی عدالتی طریقہ کار (High Court Rules & Orders) کے عین مطابق مکمل قانونی درخواست / نوٹس تیار کرنا ہے۔

دستاویز کی قسم: {doc_meta['title']}
متعلقہ قانون: {doc_meta['law']}

فریقین کی تفصیل:
- سائل / مدعی / نوٹس کنندہ: {petitioner}
- مخالف فریق / ملزم: {respondent}
- عدالت کا نام: {court_name}, {city}
{"- مقدمہ نمبر (FIR No): " + fir_no if fir_no else ""}
{"- متعلقہ تھانہ: " + police_station if police_station else ""}
{"- لاگو دفعات: " + sections if sections else ""}
{"- رقم / چیک نمبر: " + amount + " " + cheque_no if (amount or cheque_no) else ""}

کیس کے بنیادی حقائق:
{case_facts}

ہدایات برائے ڈرافٹنگ (MANDATORY COURT RULES):
1. دستاویز باقاعدہ عدالتی ہیڈنگ سے شروع کریں (مثال: 'عدالتِ عالیہ / عدالت جناب سیشن جج صاحب، {city}')۔
2. فریقین کے نام اور ولدیت/پتہ مناسب عدالتی فارمیٹ (عنوانِ مقدمہ / Memo of Parties) میں لکھیں۔
3. اگر ایف آئی آر ہے تو 'مقدمہ نمبر، تاریخ، بجرم دفعات، تھانہ' واضح درج کریں۔
4. واقعات کو ترتیب وار نمبروں (1, 2, 3...) میں واضح، قانونی اور باوقار انداز میں تحریر کریں۔
5. ٹھوس قانونی بنیادیں (Grounds) درج کریں (مثلاً بے گناہی، تاخیر، ثبوت کی عدم دستیابی، بدنیتی، یا آئینی حق)۔
6. آخر میں باقاعدہ استدعا (Prayer Clause) تحریر کریں۔
7. تصدیق (Verification) اور وکیل کے دستخط کا خانہ (Through: Advocate High Court) شامل کریں۔
8. کوئی بھی فرضی یا نامکمل نوٹ نہ چھوڑیں۔ مکمل تیار شدہ ڈرافٹ پیش کریں۔
"""
        else:
            prompt = f"""
You are a Senior Advocate of the High Courts and Supreme Court of Pakistan.
Your objective is to draft a court-standard, formal legal pleading / notice strictly conforming to the High Court Rules and Orders and statutory practice in Pakistan.

DOCUMENT TYPE: {doc_meta['title']}
GOVERNING STATUTE: {doc_meta['law']}

PARTIES & COURT DETAILS:
- Court Title: {court_name}, {city}
- Petitioner / Complainant / Client: {petitioner}
- Respondent / Opponent / Accused: {respondent}
{"- FIR Details: FIR No. " + fir_no + " Police Station: " + police_station if fir_no else ""}
{"- Offence / Sections: " + sections if sections else ""}
{"- Amount / Cheque Ref: " + amount + " " + cheque_no if (amount or cheque_no) else ""}

CASE FACTS & INSTRUCTIONS:
{case_facts}

MANDATORY PLEADING RULES:
1. Start with the formal Court Heading: 'IN THE COURT OF {court_name.upper()}, {city.upper()}'.
2. Provide formal Memo of Parties ('Petitioner(s) ... VERSUS ... Respondent(s)').
3. Include formal cause title (e.g. 'APPLICATION UNDER SECTION 497 Cr.P.C. FOR GRANT OF POST-ARREST BAIL').
4. Present facts in numbered, dignified legal paragraphs ('1. That the petitioner is a law-abiding citizen...').
5. Set forth comprehensive, authoritative GROUNDS (e.g. delay, lack of independent witnesses u/s 103 CrPC, further inquiry u/s 497(2) CrPC, mala fides, prima facie absence of criminal intent).
6. Conclude with an unambiguous PRAYER clause.
7. Include standard Verification clause, Affidavit declaration, and Counsel Signature Block ('PETITIONER THROUGH COUNSEL: ADVOCATE HIGH COURT').
8. Produce a complete, polished draft without placeholders.
"""
        return prompt.strip()

    def generate_draft(
        self,
        doc_type: str,
        case_facts: str,
        parties: dict[str, str],
        court_info: dict[str, str],
        language: str = "en",
    ) -> str:
        """Generate legal pleading draft via Groq LLM with fallback models."""
        prompt = self.build_drafting_prompt(doc_type, case_facts, parties, court_info, language)
        if not self.client:
            raise RuntimeError("Groq API client is not configured.")

        models_to_try = [GROQ_MODEL] + [m for m in FALLBACK_MODELS if m != GROQ_MODEL]
        last_error = None

        system_instruction = (
            "You are a Senior High Court Advocate and Master Pleading Drafter in Pakistan. "
            "Write formal, court-ready legal pleadings. Use authoritative legal terminology, proper numbering, "
            "prayer clauses, and verification certificates."
        )

        for model in models_to_try:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=2500,
                )
                msg = response.choices[0].message
                content = msg.content
                if not content or not content.strip():
                    reasoning = getattr(msg, "reasoning", None)
                    if reasoning and reasoning.strip():
                        content = reasoning.strip()

                if content and content.strip():
                    return content.strip()
            except Exception as exc:
                LOGGER.warning("Drafting with model %s failed: %s. Trying fallback...", model, exc)
                last_error = exc
                continue

        raise RuntimeError(f"All drafting models failed. Last error: {last_error}")

    def export_to_docx(
        self,
        content: str,
        title: str = "Court Pleading",
        court_name: str = "",
        city: str = "",
    ) -> bytes:
        """Export legal pleading into court-formatted Microsoft Word (.docx) document."""
        doc = Document()

        # Set standard legal margins: 1 inch top/bottom/right, 1.25 inch left (for court binding)
        sections = doc.sections
        for sec in sections:
            sec.top_margin = Inches(1.0)
            sec.bottom_margin = Inches(1.0)
            sec.left_margin = Inches(1.25)
            sec.right_margin = Inches(1.0)

        # Style configurations
        style = doc.styles["Normal"]
        font = style.font
        font.name = "Times New Roman"
        font.size = Pt(12)
        font.color.rgb = RGBColor(0x11, 0x18, 0x27)

        # Header Title
        title_para = doc.add_paragraph()
        title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title_run = title_para.add_run(court_name.upper() if court_name else title.upper())
        title_run.bold = True
        title_run.font.size = Pt(14)

        if city:
            city_para = doc.add_paragraph()
            city_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            city_run = city_para.add_run(f"AT {city.upper()}\n")
            city_run.bold = True
            city_run.font.size = Pt(13)

        # Divider line
        p_div = doc.add_paragraph()
        p_div.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_div.paragraph_format.space_after = Pt(12)
        p_div.add_run("—" * 38)

        # Content formatting line by line
        lines = content.split("\n")
        for line in lines:
            line_str = line.strip()
            if not line_str:
                doc.add_paragraph()
                continue

            # Heading detection
            if line_str.startswith("# ") or line_str.startswith("IN THE COURT") or line_str.startswith("عدالت"):
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                clean_heading = line_str.lstrip("# ").strip()
                r = p.add_run(clean_heading)
                r.bold = True
                r.font.size = Pt(13)
                p.paragraph_format.space_before = Pt(8)
                p.paragraph_format.space_after = Pt(4)
            elif line_str.startswith("## ") or line_str.startswith("PRAYER") or line_str.startswith("استدعا") or line_str.startswith("GROUNDS:"):
                p = doc.add_paragraph()
                clean_heading = line_str.lstrip("# ").strip()
                r = p.add_run(clean_heading)
                r.bold = True
                r.font.size = Pt(12)
                p.paragraph_format.space_before = Pt(8)
                p.paragraph_format.space_after = Pt(2)
            elif "VERSUS" in line_str.upper() or "بمقابلہ" in line_str:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r = p.add_run(line_str)
                r.bold = True
                p.paragraph_format.space_before = Pt(4)
                p.paragraph_format.space_after = Pt(4)
            else:
                p = doc.add_paragraph()
                p.paragraph_format.line_spacing = 1.15
                p.paragraph_format.space_after = Pt(3)
                p.add_run(line_str)

        # Bottom Advocate Block
        doc.add_paragraph()
        p_sign = doc.add_paragraph()
        p_sign.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r_sign = p_sign.add_run(
            "Petitioner / Complainant\nThrough Counsel:\nADVOCATE HIGH COURT / SUPREME COURT\n"
        )
        r_sign.bold = True

        buf = io.BytesIO()
        doc.save(buf)
        buf.seek(0)
        return buf.getvalue()

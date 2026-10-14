import os
import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            # First page is the cover / executive header
            return

        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#1E3A8A"))
        self.drawString(54, 755, "PAKISTAN LEGALBOT")
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(155, 755, "|  Mobile & Client Application Software Requirements (SRS)")

        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(54, 748, 558, 748)

        # Footer
        self.line(54, 45, 558, 45)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(54, 32, "Confidential - Prepared for Mobile Application Engineering Team")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 32, page_str)
        self.restoreState()

def build_pdf(output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    NAVY = colors.HexColor("#0F172A")
    SLATE_BLUE = colors.HexColor("#1E3A8A")
    ACCENT_GOLD = colors.HexColor("#B45309")
    BODY_COLOR = colors.HexColor("#1E293B")
    MUTED_GRAY = colors.HexColor("#64748B")
    BG_LIGHT = colors.HexColor("#F8FAFC")
    BORDER_LIGHT = colors.HexColor("#E2E8F0")

    # Custom typography styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=NAVY,
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=SLATE_BLUE,
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        'DocH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=SLATE_BLUE,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=NAVY,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=BODY_COLOR,
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=BODY_COLOR,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3
    )

    code_style = ParagraphStyle(
        'DocCode',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0F172A")
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=BODY_COLOR
    )

    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#92400E")
    )

    story = []

    # ---------------------------------------------------------
    # COVER / HEADER BLOCK
    # ---------------------------------------------------------
    story.append(Paragraph("PAKISTAN LEGALBOT", title_style))
    story.append(Paragraph("Mobile &amp; Client Application - Software Requirements Specification (SRS) &amp; Architecture Blueprint", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=SLATE_BLUE, spaceBefore=2, spaceAfter=10))

    meta_table_data = [
        [
            Paragraph("<b>Target Platforms:</b> iOS, Android (Flutter / React Native), Responsive Web", table_cell_style),
            Paragraph("<b>Document Version:</b> 2.0 (Production Blueprint)", table_cell_style),
        ],
        [
            Paragraph("<b>Backend Stack:</b> FastAPI, Meta FAISS (46,442 Chunks), Groq LLMs (GPT-OSS 120B / Qwen 27B)", table_cell_style),
            Paragraph("<b>Date:</b> October 2026", table_cell_style)
        ],
        [
            Paragraph("<b>Base Backend URL:</b> http://127.0.0.1:8000 (Production: https://api.yourdomain.com)", table_cell_style),
            Paragraph("<b>Security:</b> API Key Header Authentication + HTTPS", table_cell_style)
        ]
    ]
    meta_table = Table(meta_table_data, colWidths=[270, 234])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
        ('BOX', (0,0), (-1,-1), 1, BORDER_LIGHT),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_LIGHT),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # SECTION 1: EXECUTIVE SUMMARY & SYSTEM OVERVIEW
    # ---------------------------------------------------------
    story.append(Paragraph("1. Executive Summary & Product Overview", h1_style))
    story.append(Paragraph(
        "<b>Pakistan LegalBot</b> is an enterprise-grade AI legal research and court document automation system built specifically for the Pakistani legal landscape. The platform bridges the gap between complex statutory enactments and actionable legal representation by utilizing <b>Retrieval-Augmented Generation (RAG)</b> over 46,442 indexed statutory passages from 77+ Pakistani legal codes, parliamentary acts, and court precedents.",
        body_style
    ))
    story.append(Paragraph(
        "The purpose of this specification is to provide the Mobile Application Developer with comprehensive functional requirements, system architecture, API schemas, and screen-by-screen UX designs required to build a native or cross-platform (Flutter / React Native) client application that interfaces with the live backend engine.",
        body_style
    ))

    # Core Value Pillars Table
    pillar_data = [
        [Paragraph("Pillar", table_header_style), Paragraph("Technical Capability", table_header_style), Paragraph("User Impact", table_header_style)],
        [
            Paragraph("<b>Multilingual RAG</b>", table_cell_style),
            Paragraph("Natural language understanding in Urdu, Pashto, Roman Urdu, and English mapped dynamically to English statutory keywords.", table_cell_style),
            Paragraph("Advocates and citizens can query legal disputes in their native language without translation errors.", table_cell_style)
        ],
        [
            Paragraph("<b>Senior Advocate Opinions</b>", table_cell_style),
            Paragraph("Strict 8-part opinion format covering Direct Answer, Law, Sections, Ingredients, Precedents, Remedies, Procedures, and Advice.", table_cell_style),
            Paragraph("Outputs court-ready research ready for client consultation and court submissions.", table_cell_style)
        ],
        [
            Paragraph("<b>Real-Time Streaming</b>", table_cell_style),
            Paragraph("Server-Sent Events (SSE) token-by-token streaming over HTTP with sub-second first-token response times.", table_cell_style),
            Paragraph("Engaging, conversational ChatGPT-style mobile experience without timeout issues.", table_cell_style)
        ],
        [
            Paragraph("<b>Automated Drafting</b>", table_cell_style),
            Paragraph("Generates court-standard pleadings (Bail 497/498 CrPC, 22-A FIR Petitions, Stay Orders) with native .docx export.", table_cell_style),
            Paragraph("Reduces lawyer drafting time from 2 hours to 60 seconds directly on mobile devices.", table_cell_style)
        ],
        [
            Paragraph("<b>Document / FIR OCR</b>", table_cell_style),
            Paragraph("Extracts text from scanned mobile camera photos and PDFs, delivering statutory audits and defense points.", table_cell_style),
            Paragraph("Enables instant on-site case evaluation at police stations and courtrooms.", table_cell_style)
        ]
    ]
    pillar_table = Table(pillar_data, colWidths=[100, 240, 164])
    pillar_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SLATE_BLUE),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOX', (0,0), (-1,-1), 1, BORDER_LIGHT),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_LIGHT),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(pillar_table)
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # SECTION 2: END-TO-END SYSTEM ARCHITECTURE
    # ---------------------------------------------------------
    story.append(Paragraph("2. System Architecture & Component Interaction", h1_style))
    story.append(Paragraph(
        "The application follows a decoupled client-server architecture. The mobile application functions as a high-performance presentation layer communicating via asynchronous REST and SSE streams with the FastAPI backend engine.",
        body_style
    ))

    arch_data = [
        [Paragraph("Layer", table_header_style), Paragraph("Technology / Component", table_header_style), Paragraph("Responsibility & Data Flow", table_header_style)],
        [
            Paragraph("<b>Mobile Client</b>", table_cell_style),
            Paragraph("Flutter / React Native<br/>(iOS & Android)", table_cell_style),
            Paragraph("Handles UI/UX, RTL Nastaliq text rendering, in-memory voice recording, SSE stream parsing, local SQLite caching, and file sharing (.docx).", table_cell_style)
        ],
        [
            Paragraph("<b>API Gateway & Server</b>", table_cell_style),
            Paragraph("FastAPI (Uvicorn)<br/>Port: 8000", table_cell_style),
            Paragraph("Authenticates requests via header `key`, manages threadpools, coordinates RAG pipelines, and handles binary document generation.", table_cell_style)
        ],
        [
            Paragraph("<b>Vector Search (RAG)</b>", table_cell_style),
            Paragraph("Meta FAISS (IndexFlatIP)<br/>all-MiniLM-L6-v2 (384-dim)", table_cell_style),
            Paragraph("Performs sub-50ms cosine similarity search across 46,442 Pakistani statutory chunks with catalog boosting and exact section injection.", table_cell_style)
        ],
        [
            Paragraph("<b>LLM Inference</b>", table_cell_style),
            Paragraph("Groq Cloud Hardware<br/>GPT-OSS 120B / Qwen 27B", table_cell_style),
            Paragraph("Executes multi-lingual translation, legal query expansion, statutory drafting, and 8-part legal research opinion generation.", table_cell_style)
        ],
        [
            Paragraph("<b>Voice Intelligence</b>", table_cell_style),
            Paragraph("Groq Whisper Large v3", table_cell_style),
            Paragraph("Pure in-memory speech-to-text conversion for Urdu, Pashto, and English voice queries with automatic punctuation.", table_cell_style)
        ],
        [
            Paragraph("<b>Document Engine</b>", table_cell_style),
            Paragraph("PyMuPDF, RapidOCR & python-docx", table_cell_style),
            Paragraph("Performs multi-page document parsing, OCR on photographed court summons/FIRs, and exports formatted Microsoft Word (.docx) files.", table_cell_style)
        ],
        [
            Paragraph("<b>Persistence Layer</b>", table_cell_style),
            Paragraph("SQLite (WAL Mode)<br/>`conversations.db`", table_cell_style),
            Paragraph("Stores queries, opinions, citations, session IDs, execution latency, and detected languages.", table_cell_style)
        ]
    ]
    arch_table = Table(arch_data, colWidths=[90, 140, 274])
    arch_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOX', (0,0), (-1,-1), 1, BORDER_LIGHT),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_LIGHT),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(arch_table)
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # SECTION 3: DETAILED MOBILE SCREEN-BY-SCREEN REQUIREMENTS
    # ---------------------------------------------------------
    story.append(Paragraph("3. Mobile Application Screen-by-Screen Specification", h1_style))
    story.append(Paragraph(
        "The mobile app should feature a bottom navigation bar with 4 primary tabs: <b>1. Legal Chat</b>, <b>2. Legal Drafter</b>, <b>3. Scanner/OCR</b>, and <b>4. Case History</b>, plus a <b>Settings/Config modal</b>.",
        body_style
    ))

    # Screen 1 & 2
    story.append(Paragraph("Screen 1: Splash & Server Authentication", h2_style))
    story.append(Paragraph("&bull; <b>Configuration Storage:</b> The app must store <code>api_base_url</code> (default: <code>http://10.0.2.2:8000</code> for Android emulator or production domain) and <code>api_key</code> in secure storage (iOS Keychain / Android Keystore).", bullet_style))
    story.append(Paragraph("&bull; <b>Health Handshake:</b> On app launch, perform <code>GET /health</code>. If reachable and <code>status == 'ready'</code>, transition to the main interface. If unreachable, display a non-blocking configuration banner allowing the user to update the server URL.", bullet_style))

    story.append(Paragraph("Screen 2: Core Legal Research & Consultation Chat (Primary Tab)", h2_style))
    story.append(Paragraph("&bull; <b>ChatGPT-Style Conversational Interface:</b> Message bubbles distinguishing User and LegalBot. Bot responses render GitHub-flavored Markdown with bold headings, numbered lists, and quote alerts.", bullet_style))
    story.append(Paragraph("&bull; <b>Token Streaming (SSE):</b> When sending a query, connect to <code>POST /ask-stream</code>. Stream tokens in real time to the active message bubble. Auto-scroll as tokens arrive.", bullet_style))
    story.append(Paragraph("&bull; <b>Voice-to-Text Button:</b> A microphone icon next to the text input. Press or hold to record audio (16kHz mono WAV/M4A). On release, send audio bytes to <code>POST /transcribe</code>. Automatically insert the returned transcribed text into the query box and submit or let the user review.", bullet_style))
    story.append(Paragraph("&bull; <b>Collapsible Statutory Citations:</b> Render a collapsible 'Verified Statutory References' drawer beneath the answer. Each item displays statute title, section/article, page number, and source passage text.", bullet_style))
    story.append(Paragraph("&bull; <b>Multilingual & RTL Support:</b> Detect Urdu or Pashto input automatically. Render RTL text using Urdu typography (e.g. <code>Noto Nastaliq Urdu</code>). Left-to-right for English.", bullet_style))
    story.append(Paragraph("&bull; <b>Action Toolbar:</b> Quick action buttons on each bot message: 'Copy Text', 'Share to WhatsApp/Email', and 'Re-query with follow-up'.", bullet_style))

    # Screen 3 & 4
    story.append(Paragraph("Screen 3: Court Pleading & Notice Drafter (Tab 2)", h2_style))
    story.append(Paragraph("&bull; <b>Template Selector:</b> Dropdown / segmented picker with 5 pre-configured document templates:", bullet_style))
    story.append(Paragraph("   - <i>Bail Application Under Section 497 / 498 CrPC</i> (Sessions or High Court)", bullet_style))
    story.append(Paragraph("   - <i>14-Day Statutory Legal Notice</i> (Cheque Dishonour Sec 489-F PPC / Contract Breach)", bullet_style))
    story.append(Paragraph("   - <i>Petition Under Section 22-A / 22-B CrPC</i> (Direction to Police to Lodge FIR)", bullet_style))
    story.append(Paragraph("   - <i>Application for Temporary Injunction</i> (Order XXXIX Rules 1 & 2 CPC Stay Order)", bullet_style))
    story.append(Paragraph("   - <i>Custom Court Pleading / Written Statement</i>", bullet_style))
    story.append(Paragraph("&bull; <b>Input Form:</b> Form fields for Case Facts (multi-line textarea), Petitioner/Complainant Details, Respondent/Accused Details, Court Name, City/District, and Language (English or Urdu).", bullet_style))
    story.append(Paragraph("&bull; <b>Live Draft Preview:</b> On submitting to <code>POST /draft</code>, display the generated legal pleading with professional court layout.", bullet_style))
    story.append(Paragraph("&bull; <b>One-Tap Word Export:</b> 'Download Word Document' button calling <code>POST /draft/docx</code>. Saves the <code>.docx</code> file into the device's Downloads directory and opens the native OS Share Sheet.", bullet_style))

    story.append(Paragraph("Screen 4: FIR & Document Scanner / OCR (Tab 3)", h2_style))
    story.append(Paragraph("&bull; <b>Media Capture:</b> Take photo using device camera or select multi-page PDF / image from gallery.", bullet_style))
    story.append(Paragraph("&bull; <b>Instant OCR Analysis:</b> Submit file to <code>POST /scan/upload</code>. Display an animated scanning radar or progress bar.", bullet_style))
    story.append(Paragraph("&bull; <b>Intelligence Audit Report:</b> Tabbed view displaying: 1. Statutory Breakdown (charged sections, statutory penalties), 2. Legal Liabilities, 3. Defense Grounds & Quashment Opportunities, 4. Extracted Clean Text.", bullet_style))
    story.append(Paragraph("&bull; <b>Audit Docx Export:</b> 'Export Legal Audit (.docx)' button calling <code>POST /scan/docx</code> to share a formatted legal brief.", bullet_style))

    story.append(Paragraph("Screen 5: Research Archive & History (Tab 4)", h2_style))
    story.append(Paragraph("&bull; <b>Chronological Consultation Feed:</b> Fetch stored sessions via <code>GET /conversations</code>. Display query snippet, date/time, latency, and language pill.", bullet_style))
    story.append(Paragraph("&bull; <b>Search & Filter:</b> Filter past cases by keyword or law branch (Criminal, Civil, Corporate, Family).", bullet_style))
    story.append(Paragraph("&bull; <b>Offline Cache:</b> Store fetched history in local SQLite or Hive database so lawyers can review citations inside courtrooms without internet connectivity.", bullet_style))
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # SECTION 4: FULL BACKEND API REFERENCE FOR DEVELOPERS
    # ---------------------------------------------------------
    story.append(Paragraph("4. Complete Backend API Specifications & Payload Schemas", h1_style))
    story.append(Paragraph(
        "All endpoints require the HTTP Header <code>key: &lt;API_KEY&gt;</code> (except <code>GET /health</code> and <code>GET /</code>). The default API Key configured on the server is <code>zafar123-secret-key</code>.",
        body_style
    ))

    api_table_data = [
        [Paragraph("Method & Endpoint", table_header_style), Paragraph("Headers & Params", table_header_style), Paragraph("Request Body", table_header_style), Paragraph("Response Format", table_header_style)],
        [
            Paragraph("<b>GET</b><br/><code>/health</code>", code_style),
            Paragraph("None (Public)", table_cell_style),
            Paragraph("None", table_cell_style),
            Paragraph("<code>{'status': 'ready', 'chunk_count': 46442, 'model_loaded': true, ...}</code>", code_style)
        ],
        [
            Paragraph("<b>POST</b><br/><code>/ask</code>", code_style),
            Paragraph("<code>key: &lt;API_KEY&gt;</code><br/><code>Content-Type: application/json</code>", code_style),
            Paragraph("<code>{<br/>&nbsp;&nbsp;\"message\": \"string\",<br/>&nbsp;&nbsp;\"top_k\": 8,<br/>&nbsp;&nbsp;\"session_id\": \"opt_str\"<br/>}</code>", code_style),
            Paragraph("<code>{<br/>&nbsp;&nbsp;\"success\": true,<br/>&nbsp;&nbsp;\"answer\": \"string (8-part)\",<br/>&nbsp;&nbsp;\"sources\": [{...}],<br/>&nbsp;&nbsp;\"latency_sec\": 4.12<br/>}</code>", code_style)
        ],
        [
            Paragraph("<b>POST</b><br/><code>/ask-stream</code>", code_style),
            Paragraph("<code>key: &lt;API_KEY&gt;</code><br/><code>Accept: text/event-stream</code>", code_style),
            Paragraph("<code>{\"message\": \"...\", \"top_k\": 8}</code>", code_style),
            Paragraph("<b>SSE Stream Events:</b><br/>• <code>data: {\"type\": \"sources\", ...}</code><br/>• <code>data: {\"type\": \"token\", \"token\": \"...\"}</code><br/>• <code>data: {\"type\": \"done\", ...}</code><br/>• <code>data: [DONE]</code>", code_style)
        ],
        [
            Paragraph("<b>POST</b><br/><code>/transcribe</code>", code_style),
            Paragraph("<code>key: &lt;API_KEY&gt;</code><br/><code>Content-Type: multipart/form-data</code>", code_style),
            Paragraph("Form file: <code>audio</code> (binary WAV/M4A/MP3, max 10MB)", table_cell_style),
            Paragraph("<code>{<br/>&nbsp;&nbsp;\"text\": \"Transcribed text\",<br/>&nbsp;&nbsp;\"language\": \"ur\",<br/>&nbsp;&nbsp;\"provider\": \"groq_whisper\"<br/>}</code>", code_style)
        ],
        [
            Paragraph("<b>GET</b><br/><code>/draft/types</code>", code_style),
            Paragraph("<code>key: &lt;API_KEY&gt;</code>", code_style),
            Paragraph("None", table_cell_style),
            Paragraph("<code>{\"types\": {\"bail_application\": {...}, \"legal_notice\": {...}}}</code>", code_style)
        ],
        [
            Paragraph("<b>POST</b><br/><code>/draft</code>", code_style),
            Paragraph("<code>key: &lt;API_KEY&gt;</code><br/><code>Content-Type: application/json</code>", code_style),
            Paragraph("<code>{<br/>&nbsp;&nbsp;\"doc_type\": \"bail_application\",<br/>&nbsp;&nbsp;\"case_facts\": \"...\",<br/>&nbsp;&nbsp;\"parties\": {\"petitioner\": \"...\"},<br/>&nbsp;&nbsp;\"court_info\": {\"court\": \"...\"},<br/>&nbsp;&nbsp;\"language\": \"en\"<br/>}</code>", code_style),
            Paragraph("<code>{<br/>&nbsp;&nbsp;\"success\": true,<br/>&nbsp;&nbsp;\"doc_type\": \"...\",<br/>&nbsp;&nbsp;\"draft\": \"Formatted pleading text\",<br/>&nbsp;&nbsp;\"length\": 3400<br/>}</code>", code_style)
        ],
        [
            Paragraph("<b>POST</b><br/><code>/draft/docx</code>", code_style),
            Paragraph("<code>key: &lt;API_KEY&gt;</code><br/><code>Content-Type: application/json</code>", code_style),
            Paragraph("<code>{<br/>&nbsp;&nbsp;\"content\": \"...\",<br/>&nbsp;&nbsp;\"title\": \"Bail Application\",<br/>&nbsp;&nbsp;\"court_name\": \"Sessions Court\",<br/>&nbsp;&nbsp;\"city\": \"Lahore\"<br/>}</code>", code_style),
            Paragraph("<b>Binary Stream:</b><br/><code>application/vnd.openxmlformats-...</code><br/>Headers: <code>Content-Disposition: attachment; filename=...docx</code>", table_cell_style)
        ],
        [
            Paragraph("<b>POST</b><br/><code>/scan/upload</code>", code_style),
            Paragraph("<code>key: &lt;API_KEY&gt;</code><br/><code>Content-Type: multipart/form-data</code>", code_style),
            Paragraph("Form fields:<br/>• <code>file</code>: Binary PDF/image<br/>• <code>language</code>: 'both' / 'en' / 'ur'<br/>• <code>max_pages</code>: 10", table_cell_style),
            Paragraph("<code>{<br/>&nbsp;&nbsp;\"analysis_markdown\": \"...\",<br/>&nbsp;&nbsp;\"extracted_text\": \"...\",<br/>&nbsp;&nbsp;\"filename\": \"fir_302.jpg\"<br/>}</code>", code_style)
        ],
        [
            Paragraph("<b>GET</b><br/><code>/conversations</code>", code_style),
            Paragraph("<code>key: &lt;API_KEY&gt;</code><br/>Query: <code>limit=50&offset=0</code>", code_style),
            Paragraph("None", table_cell_style),
            Paragraph("<code>{\"total\": 14, \"conversations\": [{...}]}</code>", code_style)
        ]
    ]

    api_table = Table(api_table_data, colWidths=[90, 115, 155, 144])
    api_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOX', (0,0), (-1,-1), 1, BORDER_LIGHT),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_LIGHT),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(api_table)
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # SECTION 5: RECOMMENDED MOBILE CLIENT TECH STACK
    # ---------------------------------------------------------
    story.append(Paragraph("5. Recommended Mobile Tech Stack & Libraries", h1_style))
    story.append(Paragraph(
        "For optimal velocity and cross-platform fidelity, <b>Flutter (Dart)</b> or <b>React Native (TypeScript)</b> is strongly recommended:",
        body_style
    ))

    stack_data = [
        [Paragraph("Category", table_header_style), Paragraph("Flutter Recommendation", table_header_style), Paragraph("React Native Recommendation", table_header_style)],
        [
            Paragraph("<b>Framework</b>", table_cell_style),
            Paragraph("Flutter 3.x (Dart 3)", table_cell_style),
            Paragraph("React Native 0.74+ (Expo SDK 51+)", table_cell_style)
        ],
        [
            Paragraph("<b>SSE Streaming</b>", table_cell_style),
            Paragraph("<code>flutter_client_sse</code> or <code>http.Client</code> with chunked stream", code_style),
            Paragraph("<code>react-native-sse</code> or <code>@microsoft/fetch-event-source</code>", code_style)
        ],
        [
            Paragraph("<b>Audio Recording</b>", table_cell_style),
            Paragraph("<code>record</code> or <code>flutter_sound</code> (16kHz WAV/M4A)", code_style),
            Paragraph("<code>expo-av</code> or <code>react-native-audio-recorder-player</code>", code_style)
        ],
        [
            Paragraph("<b>Markdown Rendering</b>", table_cell_style),
            Paragraph("<code>flutter_markdown</code> with custom table & code block builder", code_style),
            Paragraph("<code>react-native-markdown-display</code>", code_style)
        ],
        [
            Paragraph("<b>Secure Storage</b>", table_cell_style),
            Paragraph("<code>flutter_secure_storage</code> (Keychain / Keystore)", code_style),
            Paragraph("<code>expo-secure-store</code> or <code>react-native-keychain</code>", code_style)
        ],
        [
            Paragraph("<b>Local Caching</b>", table_cell_style),
            Paragraph("<code>sqflite</code> or <code>hive_flutter</code>", code_style),
            Paragraph("<code>expo-sqlite</code> or <code>watermelondb</code>", code_style)
        ],
        [
            Paragraph("<b>File Handling</b>", table_cell_style),
            Paragraph("<code>path_provider</code>, <code>open_filex</code>, <code>share_plus</code>", code_style),
            Paragraph("<code>expo-file-system</code>, <code>expo-sharing</code>", code_style)
        ],
        [
            Paragraph("<b>Typography & RTL</b>", table_cell_style),
            Paragraph("<code>google_fonts</code> (Noto Nastaliq Urdu, Inter)", code_style),
            Paragraph("Custom bundled TTF/OTF fonts with `I18nManager.isRTL`", code_style)
        ]
    ]
    stack_table = Table(stack_data, colWidths=[110, 195, 199])
    stack_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SLATE_BLUE),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOX', (0,0), (-1,-1), 1, BORDER_LIGHT),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_LIGHT),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(stack_table)
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # SECTION 6: NON-FUNCTIONAL REQUIREMENTS & DEVELOPER GUIDELINES
    # ---------------------------------------------------------
    story.append(Paragraph("6. Non-Functional Requirements &amp; Engineering Guidelines", h1_style))
    story.append(Paragraph("&bull; <b>Sub-Second First-Token Latency:</b> The mobile client must prioritize SSE streaming (<code>/ask-stream</code>) over the blocking <code>/ask</code> endpoint to ensure the user perceives immediate responsiveness within 800ms - 1.2s.", bullet_style))
    story.append(Paragraph("&bull; <b>In-Memory Audio Privacy:</b> Audio recordings sent to <code>/transcribe</code> are never persisted on the server disk. The mobile app must also purge temporary local audio cache files immediately after successful transcription.", bullet_style))
    story.append(Paragraph("&bull; <b>Strict Pakistani Legal Vocabulary:</b> The app must preserve original statutory terms (e.g. <i>Hukm-e-Imtinaee</i> for Injunction, <i>Qabil-e-Dastandazi-e-Police</i> for Cognizable Offence) and never substitute generic translations.", bullet_style))
    story.append(Paragraph("&bull; <b>Network Resilience &amp; Offline Handling:</b> When network connectivity is lost, allow users to browse and search previously cached legal opinions and citations locally. Display an offline badge on the UI.", bullet_style))
    story.append(Paragraph("&bull; <b>File Size Constraints:</b> Audio uploads are limited to 10 MB. Scanned document uploads are limited to 25 MB and 10 pages per scan session.", bullet_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated SRS PDF at: {output_path}")

if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "Pakistan_LegalBot_App_Specification.pdf"
    build_pdf(out_file)

# ⚖️ Pakistan LegalBot: Enterprise AI Legal Research Assistant

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.139.2-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.59.2-FF4B4B.svg)](https://streamlit.io/)
[![Groq LLM](https://img.shields.io/badge/Groq-High--Speed%20Inference-f55036.svg)](https://groq.com/)
[![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS%20Cosine-green.svg)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-Proprietary%20%2F%20Research-lightgrey.svg)]()

An enterprise-grade, retrieval-augmented legal research platform powered by **FAISS dense vector search**, **Pakistani Statutory Catalog Boosting**, and **Groq High-Performance LLMs (GPT-OSS 120B / Qwen 27B / LLaMA 3.3)**.

Designed specifically for the Pakistani legal system, **LegalBot** analyzes complex disputes and generates senior advocate-grade legal research opinions complete with exact statutory citations, applicable sections, authoritative judicial precedents, and practical legal remedies in **Urdu (اردو)**, **Pashto (پښتو)**, **Roman Urdu**, and **English**.

---

## 🏛️ System Architecture

```
                          ┌───────────────────────────┐
                          │     User Legal Query      │
                          │ (Urdu / Pashto / English) │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │   Universal Legal Query Formulator (LLM)   │
                  │   - Identifies branch of Pakistani law    │
                  │   - Extracts sections, articles & orders  │
                  │   - Maps to English statutory keywords    │
                  └─────────────────────┬─────────────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
              ┌─────────────────────┐       ┌──────────────────────┐
              │  Dual-Layer Hybrid  │       │  Statutory Catalog   │
              │  Embedding Search   │       │  & Section Injection │
              │  (all-MiniLM-L6-v2) │       │  (80+ Statutory Maps)│
              └──────────┬──────────┘       └──────────┬───────────┘
                         │                             │
                         └──────────────┬──────────────┘
                                        ▼
                      ┌─────────────────────────────────┐
                      │    Hybrid Re-Ranker & Scorer    │
                      │  - Dense semantic cosine score  │
                      │  - Exact Section / Article hit  │
                      │  - Order & Rule exact matching  │
                      │  - Statute domain title match   │
                      └─────────────────┬───────────────┘
                                        │ (Top-K Passages)
                                        ▼
                      ┌─────────────────────────────────┐
                      │  Structured Legal Research      │
                      │  Prompt with Strict Dictionary  │
                      └─────────────────┬───────────────┘
                                        │
                                        ▼
                      ┌─────────────────────────────────┐
                      │      Groq LLM Generation        │
                      │   (Primary: GPT-OSS 120B)       │
                      │   (Fallback: Qwen 27B / 20B)    │
                      └─────────────────┬───────────────┘
                                        │
                                        ▼
                      ┌─────────────────────────────────┐
                      │    8-Part Senior Advocate       │
                      │    Legal Research Opinion       │
                      └─────────────────────────────────┘
```

---

## 🌟 Core Modules & Capabilities

### 1. 🤖 Multilingual Legal Assistant (RAG)
- **46,442+ Statutory Vectors**: Indexed from 80+ official Pakistani acts, ordinances, presidential orders, and gazettes.
- **Native Multilingual**: Query in English, Urdu (*اردو*), Pashto (*پښتو*), or Roman Urdu.
- **Strict Anti-Hallucination**: Directs the LLM using verified statutory grounding and an authoritative Pakistani legal dictionary.
- **Live SSE Token Streaming**: ChatGPT-style response streaming via `/ask-stream`.

### 2. 🎙️ In-Memory Speech-to-Text & Audio Playback
- **Groq Whisper Transcription**: Voice questions transcribed in memory without writing temporary audio files to disk.
- **Audio Verification**: Validates file magic bytes (WAV, MP3, OGG, FLAC, WEBM, M4A) and enforces maximum payload limits (10 MB).
- **Voice Response Player**: Integrated HTML5 player for responsive speech audio output.

### 3. 📜 Court Pleading & Statutory Notice Drafter (`app/drafter.py`)
- **Ready-to-File Court Petitions**:
  - Bail Application u/s 497/498 CrPC
  - Ex-Officio Justice of the Peace Petition u/s 22-A & 22-B CrPC
  - Temporary Injunction & Stay Application under Order XXXIX Rules 1 & 2 CPC
  - 14-Day Statutory Legal Notice for Dishonoured Cheque u/s 489-F PPC
- **High Court Standard Formatting**: 1.25" left margin for ribbon/tape binding, formal cause titles, memo of parties, statutory grounds, prayer, and verified affidavits.
- **One-Click Word (.docx) Export**: Download editable Microsoft Word documents formatted for Pakistani courts.

### 4. 🔍 Legal Document & FIR OCR Scanner (`app/scanner.py`)
- **Deep Document Inspection**: Process photos and scans of FIRs, court orders, bail orders, and agreements.
- **RapidOCR & PyMuPDF Engine**: Fast in-memory text extraction for scanned documents.
- **Statutory Intelligence Report**: Automatically parses parties, relevant penal sections, court jurisdiction, procedural timeline, legal red flags/loopholes, and recommended next steps.
- **Inter-Module Integration**: Transfer extracted facts directly into the Court Drafter with one click, or export reports to Word (`.docx`).

### 5. 💾 SQLite Chat History & Archive (`app/database.py`)
- Persistent thread storage with message metadata, source references, timestamps, and session tracking.

---

## 📚 Knowledge Base Coverage (80+ Statutes & 46,442 Passages)

The vector database (`embeddings/faiss.index`) indexes **46,442 verified legal passages** across all primary branches of Pakistani statutory law:

| Legal Domain | Key Enactments & Statutory Codes | Primary Provisions Covered |
|---|---|---|
| **Constitutional Law & Amendments** | Constitution of Pakistan, 1973 & **26th Amendment Act, 2024** | Fundamental Rights (Arts 9–25), Right to Clean Environment (Art 9A), Constitutional Benches (Arts 175A, 191A, 202A), Writ Jurisdiction (Art 199), Supreme Court (184(3)) |
| **Supreme Court Practice** | Supreme Court (Practice and Procedure) Act, 2023 | Bench Constitution (CJP + 2 Senior Judges), Appeals against 184(3) Orders within 30 days, Urgent Applications Hearing |
| **Overseas Pakistanis Property** | Establishment of Special Court (Overseas Pakistanis Property) Act, 2024 (Act XXVIII of 2024) | Special Courts for Overseas Pakistanis, Summary Disposal, Urgent Eviction & Property Protection, E-filing of Petitions |
| **Criminal Law** | Pakistan Penal Code, 1860 (PPC) | Murder (302), Hurt (337), Rape (376), Cheating (420), Dishonoured Cheque (489-F), Harassment of Women (509), Trespass (441/447), False Evidence (193) |
| **Criminal Procedure** | Code of Criminal Procedure, 1898 (CrPC) | FIR (154), Investigation (156), Remand (167), Challan (173), Ex-Officio Justice of Peace (22-A/22-B), Bail (497/498), Early Acquittal (249-A/265-K), Inherent Quashment (561-A) |
| **Civil Procedure** | Code of Civil Procedure, 1908 (CPC) | Temporary Injunctions (Order XXXIX Rules 1 & 2), Breach of Stay / Contempt (Order XXXIX Rule 2(3)), Inherent Powers & Status Quo Ante (Section 151), Rejection of Plaint (Order VII Rule 11), Summary Suits (Order XXXVII) |
| **Elections & Democracy** | Elections Act, 2017 & Election Rules (ECP) | Conduct of Elections, Delimitation, Nomination Scrutiny, Election Tribunals, Reserved Seats, Qualifications & Disqualifications (Sec 239) |
| **Banking & Finance** | State Bank of Pakistan Act, 1956 & Banking Companies Ordinance, 1962 | Monetary Policy, Foreign Exchange Regulation, Prudential Regulations for Commercial and Microfinance Banks, Inspection Powers (Sec 48) |
| **Competition & Monopolies** | Competition Act, 2010 (CCP) | Abuse of Dominant Position (Sec 3), Prohibited Agreements & Cartels (Sec 4), Exemptions (Sec 5), Deceptive Marketing (Sec 10), Merger Clearance (Sec 11) |
| **Evidence Law** | Qanun-e-Shahadat Order, 1984 (QSO) | Competency of Witnesses (Art 3), Admissions & Confessions (Arts 37–40), Dying Declaration (Art 46), Expert Opinion (Art 59), Burden of Proof (Arts 117–119), Modern Electronic Evidence (Art 164) |
| **Family & Personal Law** | Family Courts Act 1964, MFLO 1961, Guardians & Wards Act 1890, Child Marriage Restraint Acts, Dowry Act 1976 | Khula & Dissolution of Marriage, Past & Future Maintenance (*Nafaqah*), Dower (*Mahr*), Custody & Guardianship of Minors, Notice of Talaq, Underage Marriage Liabilities |
| **Property & Tenancy** | Illegal Dispossession Act 2005, Specific Relief Act 1877, Transfer of Property Act 1882, Easements Act 1882 | Land Grabbing & Eviction (Secs 3, 4, 7), Specific Performance, Declaratory Decrees (Sec 42), Perpetual Injunctions (Sec 54), Lis Pendens (Sec 52 TPA), Part Performance (Sec 53A) |
| **Corporate & Commercial** | Companies Act 2017, Contract Act 1872, Arbitration Act 1940, Sale of Goods Act 1930, Asaan Karobar Act 2025 | Breach of Contract & Damages (Secs 73/74), SECP Regulations, Director Liabilities, Arbitration Awards, Deregulation & Business Registration |
| **Cyber & Special Crimes** | PECA 2016, Harassment of Women at Workplace Act 2010/2022, NAO 1999 (NAB), ATA 1997, CNSA 1997, Acid Crime Act 2011, JJSA 2018 | Cyber Stalking (Sec 24), Online Defamation (Sec 20), Workplace Inquiry Committees & Ombudsman, NAB Corruption (Sec 9), Anti-Terrorism Courts & 4th Schedule, Narcotics Penalties |
| **Police & Administration** | Police Order 2002, Police Rules (Punjab, Sindh, KPK), Rules of Business (Federal & Provincial) | Police Conduct, Cognizable Complaints, Roznamcha/Diary Entries, Public Safety Commissions, Provincial Cabinet & Secretariat Procedures |

---

## 🎯 Strict Pakistani Legal Terminology Dictionary

To ensure court-standard precision, the model strictly enforces standard Pakistani legal terminology and eliminates translation hallucinations:

| English Legal Term | Approved Pakistani Urdu Standard | Prohibited Incorrect Translations |
|---|---|---|
| **Jurisdiction** | دائرہ اختیار / اختیارِ سماعت | ~~جھڑپ~~ |
| **Cognizable Offence** | قابلِ دست اندازیِ پولیس جرم | ~~شناختی جرم~~ |
| **Non-Cognizable Offence** | نا قابلِ دست اندازیِ پولیس جرم | ~~غیر شناختی جرم~~ |
| **Mala fides / Malice** | بدنیتی | ~~بے رحمی~~ |
| **Quashment of FIR** | درخواستِ اخراجِ مقدمہ / کوئیشمنٹ پٹیشن | ~~منسوخی کیس~~ |
| **Inherent Powers** | اندرونی / موروثی اختیارات | ~~قدرتی طاقت~~ |
| **Abuse of Process of Court** | عدالتی طریقہ کار کا ناجائز استعمال | ~~عدالتی عمل کا غلط استعمال~~ |
| **Dishonoured Cheque** | چیک کا باؤنس / مسترد ہونا | ~~چیک کی توہین~~ |
| **Temporary Injunction** | حکمِ امتناعی / سٹے آرڈر | ~~عارضی روک تھام~~ |
| **Status Quo Ante** | سابقہ پوزیشن کی بحالی | ~~پرانی حالت~~ |
| **Civil Prison** | سول جیل | ~~شہری قید خانہ~~ |
| **Attachment of Property** | جائیداد کی قرقی | ~~جائیداد جوڑنا~~ |

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
- **Python**: 3.11, 3.12, or 3.13
- **Git & Git LFS**: Installed ([git-lfs.com](https://git-lfs.com/))
- **Groq API Key**: Free API key from [console.groq.com](https://console.groq.com/)

### 2. Clone Repository & Setup Virtual Environment

```powershell
# Clone the repository
git clone https://github.com/Zafar488/AI-LAW-ASSISTANT.git
cd AI-LAW-ASSISTANT

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install production dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables (`.env`)

Copy `.env.example` to `lawyer_chatbot/.env`:

```powershell
copy lawyer_chatbot\.env.example lawyer_chatbot\.env
```

Update `lawyer_chatbot/.env` with your credentials:

```dotenv
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
API_KEY=zafar123-secret-key
GROQ_MODEL=openai/gpt-oss-120b
API_URL=http://127.0.0.1:8000
ALLOWED_ORIGINS=http://localhost:8501,http://127.0.0.1:8501
```

### 4. Start Backend Server (FastAPI)

```powershell
cd lawyer_chatbot
python run_server.py
```
*The FastAPI backend starts on `http://127.0.0.1:8000` with interactive Swagger docs at `http://127.0.0.1:8000/docs`.*

### 5. Launch User Interface (Streamlit)

In a new terminal window:
```powershell
cd lawyer_chatbot
..\.venv\Scripts\Activate.ps1
streamlit run ui/streamlit_app.py --server.port 8501
```
*Access the workspace in your browser at `http://localhost:8501`.*

---

## 📡 API Reference

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `GET` | `/health` | Check service readiness, FAISS status, and vector counts | Public |
| `POST` | `/ask` | Full legal research RAG pipeline (JSON response) | Header: `key: <API_KEY>` |
| `POST` | `/ask-stream` | Server-Sent Events (SSE) token streaming for real-time answers | Header: `key: <API_KEY>` |
| `POST` | `/transcribe` | In-memory voice transcription via Groq Whisper | Header: `key: <API_KEY>` |
| `POST` | `/draft` | Generate court petitions and statutory notices (JSON & HTML) | Header: `key: <API_KEY>` |
| `POST` | `/draft/docx` | Download formatted court petition as Microsoft Word (`.docx`) | Header: `key: <API_KEY>` |
| `POST` | `/scan/upload` | OCR scan image/PDF and generate full legal intelligence report | Header: `key: <API_KEY>` |
| `POST` | `/scan/docx` | Download legal scan intelligence report as Word (`.docx`) | Header: `key: <API_KEY>` |

### Sample Research Request (`/ask`)

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "key: zafar123-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "اگر کوئی شخص جان بوجھ کر جعلی چیک دے جو بینک سے باؤنس ہو جائے تو پاکستان پینل کوڈ کی کون سی دفعہ لاگو ہوگی اور کیا سزا ہوگی؟",
    "top_k": 8
  }'
```

---

## 🧪 Automated Test Suite

The repository includes a comprehensive 37-test automated test suite covering security, database persistence, streaming, legal drafting, OCR scanning, and metric calculations:

```powershell
cd lawyer_chatbot
..\.venv\Scripts\python.exe -m unittest discover tests
```

### Verified Test Results:
```text
Ran 37 tests in 8.041s
OK
```

- `tests/test_api_security.py`: API key validation, 401 unauthorized checks, audio file validation, and 413 payload size limits.
- `tests/test_database.py`: SQLite conversation persistence and session queries.
- `tests/test_drafter.py`: Pleading structure, cause title, memo of parties, and Word docx export.
- `tests/test_scanner.py`: Document classification, party extraction, penal section mapping, and loophole detection.
- `tests/test_streaming.py`: Server-Sent Events (SSE) stream endpoints.
- `tests/test_legal_prompt.py`: Prompt builder verification and Pakistani legal dictionary enforcement.
- `tests/test_retriever_metric.py`: Cosine similarity ranking and section injection logic.

---

## 📦 Large Artifacts & Git LFS

Due to GitHub's 100 MB file size limit, large vector chunks (`official_web_chunks.json` ~366 MB) are tracked using **Git LFS (Large File Storage)**. 

To ensure complete vector data is downloaded when cloning:
```powershell
git lfs install
git lfs pull
```

---

## ⚖️ Legal Disclaimer

*Pakistan LegalBot is an AI-assisted legal research and educational reference platform. It does not create an advocate-client relationship or provide binding legal counsel. In all legal disputes, consult an advocate licensed with the Pakistan Bar Council or Provincial Bar Council, and verify statutory provisions against official Gazette notifications.*

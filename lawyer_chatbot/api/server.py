"""FastAPI Application Server for Pakistan Legal Research Assistant.

Production Security & Deployment Notes:
- Rate Limiting: For production exposure, configure a reverse proxy (e.g. Nginx, Cloudflare)
  or FastAPI limiter middleware to enforce per-IP and per-key rate limits.
- HTTPS Termination: TLS must be terminated at the reverse proxy or cloud gateway.
- CORS: Allowed origins should be restricted via the ALLOWED_ORIGINS environment variable.
- In-memory Voice: Audio uploads are processed purely in memory and never stored on disk.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

# Get project root directory
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Load .env from project root explicitly
load_dotenv(BASE_DIR / ".env", override=True)

from app.chatbot import LegalChatbot
from app.database import (
    get_conversation_count,
    get_recent_conversations,
    save_conversation,
)
from app.drafter import DOCUMENT_TYPES, LegalDrafter
from app.language import detect_language
from app.scanner import LegalDocumentScanner

LOGGER = logging.getLogger("legalbot.api")
API_KEY = os.getenv("API_KEY")

# Maximum permitted audio upload: 10 MB
MAX_AUDIO_SIZE_BYTES = int(os.getenv("MAX_AUDIO_SIZE_MB", "10")) * 1024 * 1024
CHUNK_READ_SIZE = 64 * 1024

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="⚖️ Lawyer LegalBot API",
    description="Pakistan Legal Research AI using FAISS + Groq LLM",
    version="2.0",
)

# -----------------------------
# CORS Middleware
# -----------------------------
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# -----------------------------
# Lazy Chatbot Singleton
# -----------------------------
bot: LegalChatbot | None = None
bot_lock = Lock()


def get_bot() -> LegalChatbot:
    """Initialize expensive ML dependencies only for the first query."""
    global bot
    if bot is None:
        with bot_lock:
            if bot is None:
                bot = LegalChatbot()
    return bot


# Lazy Drafter Singleton
drafter: LegalDrafter | None = None
drafter_lock = Lock()


def get_drafter() -> LegalDrafter:
    """Initialize drafter singleton on demand."""
    global drafter
    if drafter is None:
        with drafter_lock:
            if drafter is None:
                drafter = LegalDrafter()
    return drafter


# Lazy Scanner Singleton
scanner: LegalDocumentScanner | None = None
scanner_lock = Lock()


def get_scanner() -> LegalDocumentScanner:
    """Initialize scanner singleton on demand."""
    global scanner
    if scanner is None:
        with scanner_lock:
            if scanner is None:
                scanner = LegalDocumentScanner()
    return scanner


# -----------------------------
# Request Models
# -----------------------------
class Query(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    top_k: int = Field(default=8, ge=1, le=15)
    session_id: str | None = Field(default=None, max_length=128)


class DraftRequest(BaseModel):
    doc_type: str = Field(default="bail_application")
    case_facts: str = Field(min_length=5, max_length=20_000)
    parties: dict[str, str] = Field(default_factory=dict)
    court_info: dict[str, str] = Field(default_factory=dict)
    language: str = Field(default="en")


class DocxExportRequest(BaseModel):
    content: str = Field(min_length=1)
    title: str = Field(default="Court Pleading")
    court_name: str = Field(default="")
    city: str = Field(default="")


class DocxScanExportRequest(BaseModel):
    analysis_markdown: str = Field(min_length=1)
    extracted_text: str = Field(default="")
    filename: str = Field(default="legal_document")


# -----------------------------
# API Key Verification
# -----------------------------
def verify_api_key(key: str | None = Header(None)) -> None:
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API_KEY is not configured on the server",
        )
    if not key or key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )


def validate_audio_content(content: bytes, filename: str | None) -> None:
    """Validate audio file size and magic-byte signature."""
    if len(content) < 12:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file is empty or corrupted.",
        )

    ext = Path(filename or "").suffix.casefold() if filename else ""
    valid_extensions = {".wav", ".mp3", ".m4a", ".mp4", ".ogg", ".flac", ".webm"}
    if ext and ext not in valid_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio file extension '{ext}'. Allowed: {sorted(valid_extensions)}",
        )

    # Magic byte header verification
    is_wav = content.startswith(b"RIFF") and b"WAVE" in content[8:16]
    is_mp3 = content.startswith(b"ID3") or (len(content) > 2 and content[0] == 0xFF and (content[1] & 0xE0) == 0xE0)
    is_ogg = content.startswith(b"OggS")
    is_flac = content.startswith(b"fLaC")
    is_webm = content.startswith(b"\x1aE\xdf\xa3")
    is_m4a = len(content) >= 8 and (content[4:8] in (b"ftyp", b"M4A ", b"isom", b"mp42"))

    if not (is_wav or is_mp3 or is_ogg or is_flac or is_webm or is_m4a):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio signature validation failed: content does not match a supported audio format.",
        )


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
async def root() -> dict[str, str]:
    return {
        "message": "⚖️ Lawyer LegalBot API is running.",
        "use": "POST /ask with headers: { key: YOUR_API_KEY }",
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    """Liveness and readiness check with dynamic knowledge base metadata."""
    index_file = BASE_DIR / "embeddings" / "faiss.index"
    chunks_file = BASE_DIR / "embeddings" / "chunks.pkl"
    manifest_file = BASE_DIR / "embeddings" / "index_manifest.json"

    index_ready = index_file.is_file()
    chunks_ready = chunks_file.is_file()
    groq_ready = bool(os.getenv("GROQ_API_KEY"))
    ready = bool(API_KEY and groq_ready and index_ready and chunks_ready)

    manifest_data: dict[str, Any] = {}
    if manifest_file.is_file():
        try:
            with manifest_file.open("r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception as exc:
            LOGGER.warning("Could not read index_manifest.json: %s", exc)

    saved_conversations = 0
    try:
        saved_conversations = get_conversation_count()
    except Exception as exc:
        LOGGER.warning("Could not read conversation count: %s", exc)

    return {
        "status": "ready" if ready else "not_ready",
        "api_key_configured": bool(API_KEY),
        "groq_configured": groq_ready,
        "knowledge_base_ready": index_ready and chunks_ready,
        "database_ready": True,
        "saved_conversations": saved_conversations,
        "model_loaded": bot is not None,
        "chunk_count": manifest_data.get("chunk_count") or manifest_data.get("total_chunks"),
        "dimension": manifest_data.get("dimension", 384),
        "metric": manifest_data.get("metric", "cosine"),
        "updated_at": manifest_data.get("updated_at") or manifest_data.get("created_at"),
    }


@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    key: str | None = Header(None),
) -> dict[str, Any]:
    """Transcribe a user-initiated recording in memory with strict size limits."""
    verify_api_key(key)

    # 1. Enforce payload size limit while reading chunks to prevent memory exhaustion
    body_chunks = []
    total_bytes = 0
    while True:
        chunk = await audio.read(CHUNK_READ_SIZE)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > MAX_AUDIO_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Audio file exceeds maximum permitted size of {MAX_AUDIO_SIZE_BYTES // (1024 * 1024)}MB.",
            )
        body_chunks.append(chunk)

    content = b"".join(body_chunks)

    # 2. Validate audio magic bytes
    validate_audio_content(content, audio.filename)

    # 3. Transcribe in memory via Groq Whisper
    try:
        from app.language import detect_language
        from app.voice import SpeechToTextService

        transcript = await run_in_threadpool(
            SpeechToTextService().transcribe,
            content,
            audio.filename or "recording.wav",
        )
        return {
            "text": transcript.text,
            "language": detect_language(transcript.text),
            "provider": transcript.provider,
            "model": transcript.model,
        }
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        # Sanitize internal errors: never leak API keys or secrets
        LOGGER.error("Transcription service runtime error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Speech transcription service is temporarily unavailable.",
        ) from exc
    except Exception as exc:
        LOGGER.exception("Unexpected error during audio transcription: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to transcribe audio recording.",
        ) from exc


@app.post("/ask")
async def ask(
    q: Query,
    request: Request,
    key: str | None = Header(None),
) -> dict[str, Any]:
    """Execute legal research RAG pipeline and store interaction in SQLite database."""
    verify_api_key(key)
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GROQ_API_KEY is not configured",
        )

    if not q.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty",
        )

    try:
        t0 = time.perf_counter()
        chatbot = await run_in_threadpool(get_bot)
        answer, sources = await run_in_threadpool(
            chatbot.ask, q.message.strip(), q.top_k
        )
        latency = time.perf_counter() - t0
        client_ip = request.client.host if request.client else None

        # Automatically store conversation in persistent SQLite database
        conv_id = await run_in_threadpool(
            save_conversation,
            user_query=q.message.strip(),
            bot_answer=answer,
            sources=sources,
            session_id=q.session_id,
            detected_language=detect_language(q.message.strip()),
            top_k=q.top_k,
            latency_sec=latency,
            client_ip=client_ip,
        )

        return {
            "success": True,
            "conversation_id": conv_id,
            "answer": answer,
            "sources": sources,
            "message_length": len(answer),
            "latency_sec": round(latency, 2),
        }
    except Exception as exc:
        LOGGER.exception("Chatbot pipeline failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error during legal research generation.",
        ) from exc


@app.post("/ask-stream")
async def ask_stream_endpoint(
    q: Query,
    request: Request,
    key: str | None = Header(None),
) -> StreamingResponse:
    """Execute legal research pipeline and stream response tokens via Server-Sent Events (SSE)."""
    verify_api_key(key)
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GROQ_API_KEY is not configured",
        )

    if not q.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty",
        )

    try:
        chatbot = await run_in_threadpool(get_bot)
        token_gen, sources = await run_in_threadpool(
            chatbot.ask_stream, q.message.strip(), q.top_k
        )
    except Exception as exc:
        LOGGER.exception("Chatbot pipeline retrieval failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error during legal research generation.",
        ) from exc

    client_ip = request.client.host if request.client else None
    t0 = time.perf_counter()

    def event_stream():
        accumulated_text = []
        try:
            # 1. Send sources metadata first
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"

            # 2. Stream tokens as they arrive
            for token in token_gen:
                accumulated_text.append(token)
                yield f"data: {json.dumps({'type': 'token', 'token': token}, ensure_ascii=False)}\n\n"

            # 3. Finalize and persist to SQLite database
            full_answer = "".join(accumulated_text)
            latency = time.perf_counter() - t0
            conv_id = save_conversation(
                user_query=q.message.strip(),
                bot_answer=full_answer,
                sources=sources,
                session_id=q.session_id,
                detected_language=detect_language(q.message.strip()),
                top_k=q.top_k,
                latency_sec=latency,
                client_ip=client_ip,
            )

            # 4. Send completion event
            yield f"data: {json.dumps({'type': 'done', 'conversation_id': conv_id, 'latency_sec': round(latency, 2)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as err:
            LOGGER.exception("Error during event streaming: %s", err)
            yield f"data: {json.dumps({'type': 'error', 'error': str(err)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/conversations")
async def list_conversations(
    limit: int = 50,
    offset: int = 0,
    session_id: str | None = None,
    key: str | None = Header(None),
) -> dict[str, Any]:
    """Retrieve stored conversations from SQLite database."""
    verify_api_key(key)
    items = await run_in_threadpool(get_recent_conversations, limit, offset, session_id)
    total = await run_in_threadpool(get_conversation_count)
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "conversations": items,
    }


# -----------------------------
# Legal Drafting Endpoints
# -----------------------------
@app.get("/draft/types")
async def get_draft_types() -> dict[str, Any]:
    """List supported legal pleading and notice templates."""
    return {"types": DOCUMENT_TYPES}


@app.post("/draft")
async def generate_legal_draft(
    req: DraftRequest,
    key: str | None = Header(None),
) -> dict[str, Any]:
    """Generate court-standard legal pleading or statutory notice."""
    verify_api_key(key)
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GROQ_API_KEY is not configured",
        )

    try:
        engine = await run_in_threadpool(get_drafter)
        draft_text = await run_in_threadpool(
            engine.generate_draft,
            doc_type=req.doc_type,
            case_facts=req.case_facts,
            parties=req.parties,
            court_info=req.court_info,
            language=req.language,
        )
        return {
            "success": True,
            "doc_type": req.doc_type,
            "draft": draft_text,
            "length": len(draft_text),
        }
    except Exception as exc:
        LOGGER.exception("Legal drafting failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating legal draft: {exc}",
        ) from exc


@app.post("/draft/docx")
async def export_draft_docx(
    req: DocxExportRequest,
    key: str | None = Header(None),
) -> Response:
    """Export legal draft into a formatted Microsoft Word (.docx) document."""
    verify_api_key(key)
    try:
        engine = await run_in_threadpool(get_drafter)
        docx_bytes = await run_in_threadpool(
            engine.export_to_docx,
            content=req.content,
            title=req.title,
            court_name=req.court_name,
            city=req.city,
        )
        safe_filename = req.title.replace(" ", "_").lower() + ".docx"
        return Response(
            content=docx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{safe_filename}"'},
        )
    except Exception as exc:
        LOGGER.exception("Docx export failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error exporting Word document: {exc}",
        ) from exc


# -----------------------------
# Legal Document & FIR OCR Scanner Endpoints
# -----------------------------
MAX_DOC_SIZE_BYTES = int(os.getenv("MAX_DOC_SIZE_MB", "25")) * 1024 * 1024


@app.post("/scan/upload")
async def scan_and_analyze_document(
    file: UploadFile = File(...),
    language: str = Form("both"),
    max_pages: int = Form(10),
    key: str | None = Header(None),
) -> dict[str, Any]:
    """Extract text from scanned PDF/image via RapidOCR and produce Pakistani statutory analysis."""
    verify_api_key(key)

    filename = file.filename or "document.pdf"
    ext = Path(filename).suffix.lower()
    allowed_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed: {sorted(allowed_extensions)}",
        )

    # Read chunks enforcing size limit
    body_chunks = []
    total_bytes = 0
    while True:
        chunk = await file.read(CHUNK_READ_SIZE)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > MAX_DOC_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum permitted size of {MAX_DOC_SIZE_BYTES // (1024 * 1024)}MB.",
            )
        body_chunks.append(chunk)

    file_bytes = b"".join(body_chunks)
    if len(file_bytes) < 16:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty or corrupted.",
        )

    try:
        engine = await run_in_threadpool(get_scanner)
        result = await run_in_threadpool(
            engine.scan_and_analyze,
            file_bytes=file_bytes,
            filename=filename,
            language=language,
            max_pages=max_pages,
        )
        return result
    except Exception as exc:
        LOGGER.exception("OCR Scanning and analysis failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document OCR and legal analysis failed: {exc}",
        ) from exc


@app.post("/scan/docx")
async def export_scan_docx(
    req: DocxScanExportRequest,
    key: str | None = Header(None),
) -> Response:
    """Export legal intelligence analysis report into formatted Microsoft Word (.docx)."""
    verify_api_key(key)
    try:
        engine = await run_in_threadpool(get_scanner)
        docx_bytes = await run_in_threadpool(
            engine.export_analysis_docx,
            analysis_markdown=req.analysis_markdown,
            extracted_text=req.extracted_text,
            filename=req.filename,
        )
        safe_filename = "legal_audit_" + Path(req.filename).stem.replace(" ", "_").lower() + ".docx"
        return Response(
            content=docx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{safe_filename}"'},
        )
    except Exception as exc:
        LOGGER.exception("Scan Docx export failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error exporting Word document: {exc}",
        ) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

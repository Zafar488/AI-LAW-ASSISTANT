from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
from threading import Lock
import sys
import os
from starlette.concurrency import run_in_threadpool

# Get project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Add the parent directory to the path so we can import app modules
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
# Load .env from project root explicitly
load_dotenv(BASE_DIR / ".env", override=True)

from app.chatbot import LegalChatbot
API_KEY = os.getenv("API_KEY")

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="⚖️ Lawyer LegalBot API",
    description="Pakistan Legal Research AI using FAISS + Groq LLM",
    version="2.0"
)

# -----------------------------
# CORS Middleware
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Initialize Chatbot
# -----------------------------
bot = None
bot_lock = Lock()


def get_bot():
    """Initialize expensive ML dependencies only for the first question."""
    global bot
    if bot is None:
        with bot_lock:
            if bot is None:
                bot = LegalChatbot()
    return bot

# -----------------------------
# Request Model
# -----------------------------
class Query(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    top_k: int = Field(default=8, ge=1, le=15)


# -----------------------------
# API Key Verification
# -----------------------------
def verify_api_key(key: str = Header(None)):
    if not API_KEY:
        raise HTTPException(status_code=503, detail="API_KEY is not configured")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="❌ Invalid or Missing API Key")


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...), key: str = Header(None)):
    """Transcribe a user-initiated recording without persisting its audio."""
    verify_api_key(key)
    content = await audio.read()
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
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# -----------------------------
# Root Endpoint
# -----------------------------
@app.get("/")
async def root():
    return {
        "message": "⚖️ Lawyer LegalBot API is running.",
        "use": "POST /ask with headers: { key: YOUR_API_KEY }"
    }


@app.get("/health")
async def health():
    """Cheap liveness/readiness information without loading ML models."""
    index_ready = (BASE_DIR / "embeddings" / "faiss.index").is_file()
    chunks_ready = (BASE_DIR / "embeddings" / "chunks.pkl").is_file()
    ready = bool(API_KEY and os.getenv("GROQ_API_KEY") and index_ready and chunks_ready)
    return {
        "status": "ready" if ready else "not_ready",
        "api_key_configured": bool(API_KEY),
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
        "knowledge_base_ready": index_ready and chunks_ready,
        "model_loaded": bot is not None,
    }


# -----------------------------
# Ask Endpoint
# -----------------------------
@app.post("/ask")
async def ask(q: Query, key: str = Header(None)):
    # 1️⃣ Verify API key
    verify_api_key(key)
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not configured")

    # 2️⃣ Validate user input
    if not q.message.strip():
        raise HTTPException(status_code=400, detail="❌ Message cannot be empty")

    # 3️⃣ Process question through chatbot
    try:
        chatbot = await run_in_threadpool(get_bot)
        answer, sources = await run_in_threadpool(
            chatbot.ask, q.message.strip(), q.top_k
        )

        return {
            "success": True,
            "answer": answer,
            "sources": sources,  
            "message_length": len(answer)
        }

    except Exception as e:
        print("Chatbot Error:", str(e))
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

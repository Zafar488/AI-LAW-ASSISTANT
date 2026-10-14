"""Premium research-focused Streamlit interface for Pakistan LegalBot.

Features:
1. Live ChatGPT-style token streaming for legal research queries.
2. In-memory speech transcription (Groq Whisper) and voice playback.
3. Persistent SQLite conversation history and database archive.
4. Court Pleading & Statutory Notice Drafter with Microsoft Word (.docx) export.
"""

from __future__ import annotations

import html
import io
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

from app.drafter import DOCUMENT_TYPES, LegalDrafter
from app.language import LANGUAGE_NAMES, detect_language
from app.scanner import LegalDocumentScanner
from ui.voice_output import render_voice_player

API_BASE = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
if API_BASE.endswith("/ask"):
    API_BASE = API_BASE[:-4]
API_KEY = os.getenv("API_KEY", "")
ASK_URL = f"{API_BASE}/ask"
STREAM_URL = f"{API_BASE}/ask-stream"
DRAFT_URL = f"{API_BASE}/draft"
TRANSCRIBE_URL = f"{API_BASE}/transcribe"
SCAN_URL = f"{API_BASE}/scan/upload"
SCAN_DOCX_URL = f"{API_BASE}/scan/docx"

st.set_page_config(
    page_title="Pakistan LegalBot | Professional Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

for key, default in {
    "history": [],
    "last_audio_id": None,
    "detected_language": "en",
    "query": "",
    "clear_query": False,
    "session_id": str(uuid.uuid4())[:8],
    "current_draft_text": None,
    "current_draft_meta": {},
    "current_scan_result": None,
    "current_scan_filename": None,
    "pending_draft_facts": None,
}.items():
    st.session_state.setdefault(key, default)

if st.session_state.clear_query:
    st.session_state.query = ""
    st.session_state.clear_query = False


def error_detail(response: requests.Response) -> str:
    try:
        return str(response.json().get("detail", response.text))
    except ValueError:
        return response.text or "The service returned an unexpected response."


@st.cache_data(ttl=10)
def check_backend_health() -> dict[str, Any]:
    """Query FastAPI /health endpoint for dynamic readiness and passage count."""
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=1.5)
        if resp.ok:
            return resp.json()
    except Exception:
        pass
    return {"status": "unavailable", "chunk_count": None, "knowledge_base_ready": False}


def choose_prompt(prompt: str) -> None:
    st.session_state.query = prompt
    st.session_state.detected_language = detect_language(prompt)


def reset_workspace() -> None:
    st.session_state.history = []
    st.session_state.query = ""
    st.session_state.last_audio_id = None


health_info = check_backend_health()
is_ready = health_info.get("status") == "ready"
raw_chunk_count = health_info.get("chunk_count")
passage_display = f"{raw_chunk_count:,} Passages" if raw_chunk_count else "Statutory Corpus"
top_badge = f"● Statutory Canon Indexed ({passage_display})" if raw_chunk_count else "● Pakistani Statutory Canon"

st.markdown(
    """
    <style>
      :root {--ink:#0c1a32;--navy:#102a54;--blue:#246bdb;--gold:#d3a74d;--line:#dbe5f2;--muted:#687991;}
      .stApp {background:linear-gradient(135deg,#f4f8fc 0%,#edf3fa 52%,#f9f6ef 100%);color:var(--ink)}
      #MainMenu,footer{visibility:hidden}[data-testid="stHeader"]{background:rgba(244,248,252,.86);backdrop-filter:blur(14px)}
      .block-container{max-width:1440px;padding-top:2.15rem;padding-bottom:2rem}
      [data-testid="stSidebar"]{background:#0b1930;border-right:1px solid rgba(255,255,255,.08)}
      [data-testid="stSidebar"] *{color:#e8effa}[data-testid="stSidebar"] .stCaption{color:#9eb1cc!important}[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.12)}
      .brand{padding:5px 0 18px}.brand-kicker{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:#d9bb70;font-weight:800}.brand-name{font-size:24px;font-weight:760;letter-spacing:-.04em;color:#fff;margin:5px 0}.brand-copy{font-size:13px;line-height:1.55;color:#9eb1cc}
      .status{padding:10px 12px;border:1px solid rgba(126,213,166,.22);border-radius:10px;background:rgba(33,105,74,.25);font-size:12px;font-weight:700;margin:8px 0 18px}.status:before{content:"";display:inline-block;width:7px;height:7px;margin-right:8px;border-radius:50%;background:#65d795;box-shadow:0 0 0 4px rgba(101,215,149,.12)}
      .status-offline{padding:10px 12px;border:1px solid rgba(235,87,87,.28);border-radius:10px;background:rgba(180,40,40,.22);color:#ffc4c4!important;font-size:12px;font-weight:700;margin:8px 0 18px}.status-offline:before{content:"";display:inline-block;width:7px;height:7px;margin-right:8px;border-radius:50%;background:#ff5e5e;box-shadow:0 0 0 4px rgba(255,94,94,.15)}
      .topline{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px}.eyebrow{font-size:11px;letter-spacing:.15em;text-transform:uppercase;color:#627b9f;font-weight:800}.pill{padding:7px 11px;border-radius:999px;background:#eaf7ef;border:1px solid #c7ead3;color:#287446;font-size:12px;font-weight:750}
      .hero{position:relative;overflow:hidden;padding:34px 38px;border-radius:20px;background:linear-gradient(117deg,#0b1930 0%,#12386f 58%,#245eaa 100%);color:#fff;box-shadow:0 18px 45px rgba(19,46,88,.2);margin-bottom:20px}.hero:after{content:"§";position:absolute;right:35px;top:-56px;color:rgba(255,255,255,.08);font-size:270px;line-height:1;font-family:Georgia,serif}.hero-kicker{position:relative;color:#e4c476;font-size:11px;font-weight:800;letter-spacing:.17em;text-transform:uppercase}.hero h1{position:relative;margin:9px 0 8px;font-size:36px;line-height:1.1;letter-spacing:-.04em}.hero p{position:relative;max-width:750px;margin:0;color:#cedced;font-size:15px;line-height:1.6}.hero-tags{position:relative;display:flex;flex-wrap:wrap;gap:8px;margin-top:18px}.hero-tags span{padding:5px 10px;border:1px solid rgba(255,255,255,.21);border-radius:999px;background:rgba(255,255,255,.07);font-size:12px;color:#e8eff8}
      .metric{min-height:84px;padding:15px 16px;border:1px solid var(--line);border-radius:14px;background:rgba(255,255,255,.85);box-shadow:0 7px 18px rgba(24,52,87,.04)}.metric-label{font-size:11px;font-weight:800;letter-spacing:.09em;text-transform:uppercase;color:#7587a1}.metric-value{margin-top:6px;font-size:20px;font-weight:760;letter-spacing:-.025em;color:#102a54}.metric-sub{margin-top:3px;font-size:12px;color:#718096}
      .panel{padding:20px 22px 16px;border:1px solid var(--line);border-radius:16px;background:#fff;box-shadow:0 10px 30px rgba(23,50,84,.055)}.panel-title{font-size:17px;font-weight:760;color:#102a54}.panel-copy{margin:4px 0 11px;font-size:13px;color:#6b7b92}.section-title{font-size:16px;font-weight:760;color:#102a54;letter-spacing:-.02em;margin:8px 0 4px}.section-copy{font-size:13px;color:#6b7b92;margin-bottom:10px}.tip{padding:14px 15px;border:1px solid #d9e4f4;border-radius:12px;background:#f8fbff;color:#58708f;font-size:13px;line-height:1.5}
      .language{display:inline-flex;align-items:center;gap:7px;margin:6px 0 4px;padding:6px 10px;border:1px solid #d9e7fb;border-radius:999px;background:#f1f6ff;color:#28558e;font-size:12px;font-weight:720}.language:before{content:"";width:7px;height:7px;border-radius:50%;background:#3f85e6;box-shadow:0 0 0 4px #e0edff}
      [data-testid="stChatMessage"]{padding:16px 18px;margin:11px 0;border:1px solid var(--line);border-radius:16px;background:#fff;box-shadow:0 8px 23px rgba(25,52,86,.04)}[data-testid="stChatMessage"] p{line-height:1.65}.answer-meta{margin:7px 0 2px;font-size:12px;color:#73839a}.source{padding:10px 12px;margin:7px 0;border-left:3px solid var(--gold);border-radius:999px;background:#fbfaf6;color:#3e526f;font-size:13px}.notice{padding:12px 14px;border:1px solid #eadfbd;border-radius:11px;background:#fffaf0;color:#796544;font-size:12px;line-height:1.55}
      .stButton>button,[data-testid="stDownloadButton"]>button{border-radius:9px;font-weight:700}.stButton>button[kind="primary"]{background:var(--blue);border-color:var(--blue);box-shadow:0 7px 15px rgba(36,107,219,.2)}textarea{border-radius:10px!important}
      .stTabs [data-baseweb="tab-list"]{gap:8px;border-bottom:2px solid var(--line);margin-bottom:16px}
      .stTabs [data-baseweb="tab"]{height:46px;border-radius:10px 10px 0 0;padding:0 18px;font-weight:750;font-size:14px;color:#506687}
      .stTabs [aria-selected="true"]{color:#102a54!important;border-bottom:3px solid #246bdb!important}
      @media(max-width:760px){.block-container{padding:1.1rem .85rem 1.5rem}.hero{padding:24px 20px}.hero h1{font-size:27px}}
    </style>
    """,
    unsafe_allow_html=True,
)

status_html = (
    '<div class="status">Backend: Active & Ready</div>'
    if is_ready
    else '<div class="status-offline">Backend: Offline / Disconnected</div>'
)

with st.sidebar:
    st.markdown(
        f"""<div class="brand"><div class="brand-kicker">Pakistan legal research</div><div class="brand-name">LegalBot Workspace</div><div class="brand-copy">An evidence-focused AI research & court drafting assistant for Pakistani law.</div></div>{status_html}""",
        unsafe_allow_html=True,
    )
    if not is_ready:
        st.warning("FastAPI backend is unreachable. Start it with: `python run_server.py`")
    st.caption("RESEARCH CONTROLS")
    top_k = st.slider("Source depth", 3, 15, 8, help="The number of relevant legal passages searched for each answer.")
    stream_enabled = st.toggle("Live Streaming (ChatGPT-style)", value=True, help="Streams legal opinions token-by-token in real time.")
    voice_enabled = st.toggle("Read answers aloud", value=True)
    auto_play = st.toggle("Auto-play latest answer", value=False, disabled=not voice_enabled)
    st.caption("Voice uses the browser voices installed on this device.")
    st.divider()
    st.caption("SESSION & DATABASE STATS")
    st.metric("Session questions", len(st.session_state.history))
    db_count = health_info.get("saved_conversations")
    if db_count is not None:
        st.metric("Total questions in Database", db_count)
    if st.button("Clear workspace", use_container_width=True, on_click=reset_workspace):
        st.rerun()
    if st.session_state.history:
        st.download_button("Export research notes", json.dumps(st.session_state.history, ensure_ascii=False, indent=2), "legalbot_research.json", "application/json", use_container_width=True)
    
    with st.expander("🗄️ Database Archive (Past Questions)"):
        try:
            from app.database import get_recent_conversations
            past_chats = get_recent_conversations(limit=10)
            if past_chats:
                for c in past_chats:
                    q_text = c.get("user_query", "")
                    q_disp = q_text[:55] + "…" if len(q_text) > 55 else q_text
                    st.caption(f"🕒 {c.get('created_at', '')}")
                    st.markdown(f"**Q:** {html.escape(q_disp)}")
                    st.markdown("---")
            else:
                st.caption("Database is currently empty. Ask a question to store it!")
        except Exception:
            st.caption("Could not load archive.")
    st.divider()
    st.markdown("<div class='notice'><strong>Privacy guarantee:</strong> Audio is processed securely in-memory and is never stored on disk.</div>", unsafe_allow_html=True)

st.markdown(f"""<div class="topline"><div class="eyebrow">Legal intelligence · Pakistan</div><div class="pill">{html.escape(top_badge)}</div></div><section class="hero"><div class="hero-kicker">Evidence-First Legal Intelligence</div><h1>Comprehensive Pakistani Legal Research & Court Drafting</h1><p>Consult across 72+ Pakistani statutes in Urdu, Pashto, Roman Urdu, or English, and generate ready-to-file court pleadings and statutory notices with direct Microsoft Word export.</p><div class="hero-tags"><span>Civil & Criminal Pleadings</span><span>Bail & 22-A CrPC</span><span>Statutory Notices (489-F PPC)</span><span>Constitution & Writs</span></div></section>""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_research, tab_drafter, tab_scanner = st.tabs([
    "💬 AI Legal Research (Chat & Live Streaming)",
    "📜 Court Pleading & Statutory Notice Drafter",
    "📷 Legal Document & FIR OCR Scanner",
])

# ==============================================================================
# TAB 1: AI LEGAL RESEARCH ASSISTANT
# ==============================================================================
with tab_research:
    card_a, card_b, card_c = st.columns(3)
    with card_a: st.markdown(f'<div class="metric"><div class="metric-label">Knowledge Base</div><div class="metric-value">{html.escape(passage_display)}</div><div class="metric-sub">Enacted statutes, codes & gazettes</div></div>', unsafe_allow_html=True)
    with card_b: card_b_metric = '<div class="metric"><div class="metric-label">Primary AI Engine</div><div class="metric-value">GPT-OSS 120B</div><div class="metric-sub">Full statutory RAG pipeline</div></div>'; st.markdown(card_b_metric, unsafe_allow_html=True)
    with card_c: st.markdown('<div class="metric"><div class="metric-label">Detected Language</div><div class="metric-value">{}</div><div class="metric-sub">Automatic multi-script detection</div></div>'.format(html.escape(LANGUAGE_NAMES.get(st.session_state.detected_language, "English"))), unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    composer_col, prompt_col = st.columns([1.7, 1], gap="large")

    with prompt_col:
        st.markdown('<div class="section-title">Common Legal Inquiries</div><div class="section-copy">Click a benchmark query to test the system:</div>', unsafe_allow_html=True)
        prompts = [
            "اگر چیک باؤنس ہو جائے تو 489-F کے تحت کیا کارروائی ہوگی اور کیا یہ قابلِ ضمانت ہے؟",
            "اگر عدالت کے سٹے آرڈر کے باوجود زمین پر قبضہ کر لیا جائے تو کیا کارروائی ہوگی؟",
            "سیشن کورٹ کی طرف سے 22-A کے تحت درج کروائی گئی جھوٹی FIR کو ہائی کورٹ میں کیسے ختم کروائیں؟",
            "ورک پلیس پر خاتون کو ہراساں کرنے کی صورت میں کون سا قانونی فورم دستیاب ہے؟",
            "پیکا (PECA) کے تحت آن لائن بلیک میلنگ اور فیک اکاؤنٹ کے خلاف کیا کارروائی ہوگی؟"
        ]
        for index, prompt in enumerate(prompts):
            st.button(prompt, key=f"prompt_{index}", use_container_width=True, on_click=choose_prompt, args=(prompt,))
        st.markdown("<div style='height:8px'></div><div class='tip'><strong>Tip:</strong> You can type or speak in Urdu (اردو), Pashto (پښتو), Roman Urdu, or English. The assistant responds with exact statutory citations and binding case law principles.</div>", unsafe_allow_html=True)

    with composer_col:
        st.markdown('<div class="panel"><div class="panel-title">✦ Research Question</div><div class="panel-copy">Type your legal scenario, or record via microphone.</div>', unsafe_allow_html=True)
        audio = st.audio_input("Record a question", label_visibility="collapsed", help="Microphone permission is requested only when recording begins.")
        if audio is not None:
            audio_data = audio.getvalue()
            audio_id = hash(audio_data)
            if audio_id != st.session_state.last_audio_id:
                with st.status("Transcribing recording…", expanded=True) as status:
                    try:
                        response = requests.post(
                            TRANSCRIBE_URL,
                            headers={"key": API_KEY},
                            files={"audio": (audio.name or "recording.wav", audio_data, audio.type)},
                            timeout=90
                        )
                        if response.ok:
                            transcript = response.json()
                            st.session_state.query = transcript["text"]
                            st.session_state.detected_language = transcript["language"]
                            st.session_state.last_audio_id = audio_id
                            status.update(label="Transcript ready for research", state="complete")
                        else:
                            status.update(label=f"Could not transcribe: {error_detail(response)}", state="error")
                    except requests.RequestException:
                        status.update(label="Speech recognition service is unavailable. Please type your question.", state="error")
        
        question = st.text_area(
            "Legal research question",
            key="query",
            height=132,
            label_visibility="collapsed",
            placeholder="مثال: اگر کوئی شخص معاہدے کی خلاف ورزی کرے تو ہرجانے کا کیا قانون ہے؟ یا Section 302 PPC کے عناصر کیا ہیں؟"
        )
        language = detect_language(question) if question.strip() else st.session_state.detected_language
        st.session_state.detected_language = language
        st.markdown(f'<span class="language">Detected Language: {html.escape(LANGUAGE_NAMES.get(language, "English"))}</span>', unsafe_allow_html=True)
        
        send_col, clear_col, info_col = st.columns([1.2, 1, 2.2])
        with send_col:
            send = st.button("Generate Legal Opinion", type="primary", use_container_width=True)
        with clear_col:
            clear_draft = st.button("Clear Input", use_container_width=True)
        with info_col:
            st.caption(f"Searching top {top_k} statutory passages across 72+ books")
        st.markdown("</div>", unsafe_allow_html=True)

    if clear_draft:
        st.session_state.query = ""
        st.rerun()

    if send:
        cleaned = question.strip()
        if not cleaned:
            st.warning("Please enter or record a legal research question first.")
        elif not API_KEY:
            st.error("API_KEY is not configured in .env file.")
        else:
            st.markdown("<div style='height:20px'></div><div class='section-title'>Live Legal Research Generation</div>", unsafe_allow_html=True)
            
            with st.chat_message("user", avatar="👤"):
                st.markdown(f"**Question:** {html.escape(cleaned)}")

            with st.chat_message("assistant", avatar="⚖️"):
                started = time.perf_counter()
                incoming_sources: list[dict[str, Any]] = []
                done_metadata: dict[str, Any] = {}

                if stream_enabled:
                    try:
                        response = requests.post(
                            STREAM_URL,
                            headers={"key": API_KEY, "Content-Type": "application/json"},
                            json={
                                "message": cleaned,
                                "top_k": top_k,
                                "session_id": st.session_state.session_id,
                            },
                            stream=True,
                            timeout=180
                        )

                        if not response.ok:
                            st.error(f"Request failed: {error_detail(response)}")
                        else:
                            def token_stream():
                                for line in response.iter_lines(decode_unicode=True):
                                    if not line:
                                        continue
                                    if line.startswith("data: "):
                                        raw_data = line[6:].strip()
                                        if raw_data == "[DONE]":
                                            break
                                        try:
                                            payload = json.loads(raw_data)
                                            event_type = payload.get("type")
                                            if event_type == "sources":
                                                incoming_sources.extend(payload.get("sources", []))
                                            elif event_type == "token":
                                                yield payload.get("token", "")
                                            elif event_type == "done":
                                                done_metadata["conversation_id"] = payload.get("conversation_id")
                                                done_metadata["latency_sec"] = payload.get("latency_sec")
                                            elif event_type == "error":
                                                yield f"\n\n**Error:** {payload.get('error')}"
                                        except Exception:
                                            continue

                            # ChatGPT-style real time typewriter streaming
                            full_answer = st.write_stream(token_stream)

                            latency = done_metadata.get("latency_sec", round(time.perf_counter() - started, 2))
                            st.markdown(
                                f'<div class="answer-meta">Language: {html.escape(LANGUAGE_NAMES.get(language, "English"))} · Processing Time: {latency:.1f}s · Supporting Sources: {len(incoming_sources)}</div>',
                                unsafe_allow_html=True,
                            )

                            if incoming_sources:
                                with st.expander(f"📚 Retrieved Legal Sources ({len(incoming_sources)})"):
                                    for s in incoming_sources:
                                        src_id = f"[{s.get('id')}] " if s.get("id") else ""
                                        doc_name = s.get("document", "Statute")
                                        sec_info = f" | Section/Article: {s.get('section')}" if s.get("section") else ""
                                        page_info = f" | Page: {s.get('page')}" if s.get("page") else ""
                                        file_info = f" | File: {s.get('file')}" if s.get("file") else ""
                                        url_val = s.get("url")
                                        url_info = f' | <a href="{html.escape(str(url_val))}" target="_blank" style="color:var(--accent);text-decoration:underline;">Source Link</a>' if url_val else ""
                                        st.markdown(
                                            f'<div class="source">🏛️ {html.escape(src_id)}<strong>{html.escape(str(doc_name))}</strong>{html.escape(sec_info)}{html.escape(page_info)}{html.escape(file_info)}{url_info}</div>',
                                            unsafe_allow_html=True,
                                        )

                            if voice_enabled:
                                render_voice_player(full_answer, language, autoplay=auto_play)

                            st.session_state.history.append({
                                "conversation_id": done_metadata.get("conversation_id"),
                                "question": cleaned,
                                "answer": full_answer,
                                "sources": incoming_sources,
                                "language": language,
                                "latency": latency,
                                "timestamp": datetime.now().isoformat(timespec="seconds"),
                            })

                    except requests.Timeout:
                        st.error("Request timed out. Please try a more specific question.")
                    except requests.RequestException:
                        st.error("Cannot connect to FastAPI backend on port 8000. Ensure run_server.py is running.")
                else:
                    # Non-streaming fallback
                    with st.spinner("Analyzing legal issues and searching FAISS knowledge base…"):
                        try:
                            response = requests.post(
                                ASK_URL,
                                headers={"key": API_KEY, "Content-Type": "application/json"},
                                json={
                                    "message": cleaned,
                                    "top_k": top_k,
                                    "session_id": st.session_state.session_id,
                                },
                                timeout=180
                            )
                            if response.ok:
                                result = response.json()
                                full_answer = result["answer"]
                                incoming_sources = result.get("sources", [])
                                latency = result.get("latency_sec", round(time.perf_counter() - started, 2))

                                st.markdown(full_answer)
                                st.markdown(
                                    f'<div class="answer-meta">Language: {html.escape(LANGUAGE_NAMES.get(language, "English"))} · Processing Time: {latency:.1f}s · Supporting Sources: {len(incoming_sources)}</div>',
                                    unsafe_allow_html=True,
                                )
                                if incoming_sources:
                                    with st.expander(f"📚 Retrieved Legal Sources ({len(incoming_sources)})"):
                                        for s in incoming_sources:
                                            src_id = f"[{s.get('id')}] " if s.get("id") else ""
                                            doc_name = s.get("document", "Statute")
                                            sec_info = f" | Section/Article: {s.get('section')}" if s.get("section") else ""
                                            page_info = f" | Page: {s.get('page')}" if s.get("page") else ""
                                            file_info = f" | File: {s.get('file')}" if s.get("file") else ""
                                            st.markdown(f'<div class="source">🏛️ {html.escape(src_id)}<strong>{html.escape(str(doc_name))}</strong>{html.escape(sec_info)}{html.escape(page_info)}{html.escape(file_info)}</div>', unsafe_allow_html=True)
                                if voice_enabled:
                                    render_voice_player(full_answer, language, autoplay=auto_play)

                                st.session_state.history.append({
                                    "conversation_id": result.get("conversation_id"),
                                    "question": cleaned,
                                    "answer": full_answer,
                                    "sources": incoming_sources,
                                    "language": language,
                                    "latency": latency,
                                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                                })
                            else:
                                st.error(f"Request failed: {error_detail(response)}")
                        except requests.Timeout:
                            st.error("Request timed out. Please try a more specific question.")
                        except requests.RequestException:
                            st.error("Cannot connect to FastAPI backend on port 8000. Ensure run_server.py is running.")

    # Display historical records (excluding the one just live-streamed in this turn)
    past_items = st.session_state.history[:-1] if (send and st.session_state.history) else st.session_state.history
    if past_items:
        st.markdown("<div style='height:24px'></div><div class='section-title'>Previous Research Records</div><div class='section-copy'>Complete research trajectory with full statutory citations.</div>", unsafe_allow_html=True)
        for item_index, item in enumerate(reversed(past_items)):
            with st.chat_message("user", avatar="👤"):
                st.markdown(f"**Question:** {item['question']}")
            with st.chat_message("assistant", avatar="⚖️"):
                st.markdown(item["answer"])
                st.markdown(f'<div class="answer-meta">Language: {html.escape(LANGUAGE_NAMES.get(item["language"], "English"))} · Processing Time: {item["latency"]:.1f}s · Supporting Sources: {len(item["sources"])}</div>', unsafe_allow_html=True)
                if item["sources"]:
                    with st.expander(f"📚 Retrieved Legal Sources ({len(item['sources'])})"):
                        for s in item["sources"]:
                            src_id = f"[{s.get('id')}] " if s.get("id") else ""
                            doc_name = s.get("document", "Statute")
                            sec_info = f" | Section/Article: {s.get('section')}" if s.get("section") else ""
                            page_info = f" | Page: {s.get('page')}" if s.get("page") else ""
                            file_info = f" | File: {s.get('file')}" if s.get("file") else ""
                            url_val = s.get("url")
                            url_info = f' | <a href="{html.escape(str(url_val))}" target="_blank" style="color:var(--accent);text-decoration:underline;">Source Link</a>' if url_val else ""
                            st.markdown(f'<div class="source">🏛️ {html.escape(src_id)}<strong>{html.escape(str(doc_name))}</strong>{html.escape(sec_info)}{html.escape(page_info)}{html.escape(file_info)}{url_info}</div>', unsafe_allow_html=True)
                if voice_enabled:
                    autoplay = bool(auto_play and item_index == 0 and not item.get("autoplayed", False))
                    render_voice_player(item["answer"], item["language"], autoplay=autoplay)
                    if autoplay:
                        item["autoplayed"] = True

# ==============================================================================
# TAB 2: COURT PLEADING & STATUTORY NOTICE DRAFTER STUDIO
# ==============================================================================
with tab_drafter:
    st.markdown("""
    <div class="panel" style="margin-top:4px;">
      <div class="panel-title">⚖️ Senior Advocate Court Pleading & Statutory Notice Studio</div>
      <div class="panel-copy">Draft court-standard legal pleadings, bail applications, 22-A CrPC petitions, and 14-day statutory notices formatted according to High Court Rules & Orders with direct Microsoft Word (.docx) export.</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    col_d1, col_d2 = st.columns([1, 1.25], gap="large")
    
    with col_d1:
        st.markdown('<div class="section-title">1. Document Type & Forum</div>', unsafe_allow_html=True)
        doc_type_choice = st.selectbox(
            "Select Document Type",
            options=[
                ("bail_application", "⚖️ Bail Application (Section 497 / 498 CrPC)"),
                ("legal_notice", "📜 14-Day Statutory Legal Notice (Cheque 489-F / Contract)"),
                ("petition_22a", "🏛️ Petition u/s 22-A & 22-B CrPC (Direction for FIR)"),
                ("stay_application", "🛡️ Application for Temporary Injunction (Order 39 CPC)"),
                ("custom_pleading", "📝 Custom Court Pleading / Written Statement"),
            ],
            format_func=lambda x: x[1],
            key="draft_doc_type"
        )[0]
        
        c_court, c_city = st.columns(2)
        with c_court:
            court_name_input = st.text_input("Court / Authority", value="In the Court of Sessions Judge", key="draft_court_name")
        with c_city:
            city_input = st.text_input("City / District", value="Lahore", key="draft_city")
            
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            petitioner_input = st.text_input("Petitioner / Complainant", placeholder="e.g. Muhammad Ali S/O Ahmad Ali", key="draft_petitioner")
        with c_p2:
            respondent_input = st.text_input("Respondent / Accused", placeholder="e.g. The State / Imran Khan", key="draft_respondent")
            
        c_extra1, c_extra2 = st.columns(2)
        with c_extra1:
            fir_input = st.text_input("FIR No. & Police Station", placeholder="e.g. 45/2026, P.S. Civil Lines", key="draft_fir")
        with c_extra2:
            sections_input = st.text_input("Offence Sections / Amount", placeholder="e.g. 489-F PPC or Rs. 1,000,000", key="draft_sections")
            
        draft_lang = st.radio("Pleading Language", options=["English (Court Standard)", "Urdu (اردو عدالتی فارمیٹ)"], horizontal=True, key="draft_lang")
        is_urdu_draft = "Urdu" in draft_lang

    with col_d2:
        st.markdown('<div class="section-title">2. Case Facts & Specific Grounds</div>', unsafe_allow_html=True)
        sample_placeholder = (
            "تفصیل درج کریں: ملزم کے خلاف بلاجواز اور رنجش کی بنا پر جھوٹی ایف آئی آر درج کی گئی ہے۔ "
            "کوئی آزاد گواہ موقع پر موجود نہیں تھا اور تفتیش میں کوئی ریکوری نہیں ہوئی۔ "
            "ملزم معزز شہری ہے اور ضمانت کی شرائط پوری کرنے کو تیار ہے۔"
            if is_urdu_draft else
            "State essential facts: Accused was falsely implicated due to business rivalry. "
            "Unexplained 4-day delay in lodging FIR. No independent witnesses under Section 103 CrPC. "
            "Investigation is complete, accused is no more required for police custody. Ready to submit solvent bail bonds."
        )
        if st.session_state.get("pending_draft_facts") is not None:
            st.session_state["draft_facts"] = st.session_state.pop("pending_draft_facts")

        facts_input = st.text_area(
            "Case Facts & Submissions",
            height=210,
            placeholder=sample_placeholder,
            key="draft_facts"
        )
        
        btn_draft, btn_clear_draft = st.columns([1.5, 1])
        with btn_draft:
            generate_draft_btn = st.button("⚖️ Generate Court Pleading", type="primary", use_container_width=True)
        with btn_clear_draft:
            if st.button("Clear Form", use_container_width=True):
                st.session_state["draft_facts"] = ""
                st.session_state["current_draft_text"] = None
                st.rerun()

    # Draft Generation & Result Handler
    if generate_draft_btn:
        if not facts_input.strip():
            st.warning("Please provide case facts and instructions to draft the pleading.")
        else:
            with st.status("Analyzing Pakistani court rules, formulating grounds, and drafting formal pleading…", expanded=True) as d_status:
                try:
                    payload = {
                        "doc_type": doc_type_choice,
                        "case_facts": facts_input.strip(),
                        "parties": {
                            "petitioner": petitioner_input.strip() or "Petitioner / Complainant",
                            "respondent": respondent_input.strip() or "The State / Respondent",
                        },
                        "court_info": {
                            "court_name": court_name_input.strip(),
                            "city": city_input.strip(),
                            "fir_no": fir_input.strip(),
                            "sections": sections_input.strip(),
                        },
                        "language": "ur" if is_urdu_draft else "en",
                    }
                    d_resp = requests.post(DRAFT_URL, headers={"key": API_KEY, "Content-Type": "application/json"}, json=payload, timeout=120)
                    if d_resp.ok:
                        draft_data = d_resp.json()
                        st.session_state["current_draft_text"] = draft_data.get("draft", "")
                        st.session_state["current_draft_meta"] = {
                            "title": court_name_input.strip() or "Court Pleading",
                            "court_name": court_name_input.strip(),
                            "city": city_input.strip(),
                            "doc_type": doc_type_choice,
                        }
                        d_status.update(label="Court Pleading successfully drafted!", state="complete")
                    else:
                        d_status.update(label=f"Drafting error: {error_detail(d_resp)}", state="error")
                except Exception as ex:
                    d_status.update(label=f"Error connecting to drafting service: {ex}", state="error")

    # Render Current Draft and Download Buttons
    if st.session_state.get("current_draft_text"):
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">📄 Court-Standard Pleading Output</div>', unsafe_allow_html=True)
        
        # Prepare Word .docx export bytes
        meta = st.session_state.get("current_draft_meta", {})
        docx_bytes = None
        try:
            docx_bytes = LegalDrafter().export_to_docx(
                content=st.session_state["current_draft_text"],
                title=meta.get("title", "Court Pleading"),
                court_name=meta.get("court_name", ""),
                city=meta.get("city", "")
            )
        except Exception as exc:
            st.caption(f"Word export preparation notice: {exc}")
            
        c_actions1, c_actions2, _ = st.columns([1.3, 1.3, 2])
        with c_actions1:
            if docx_bytes:
                st.download_button(
                    label="📥 Download Word (.docx)",
                    data=docx_bytes,
                    file_name=f"{meta.get('doc_type', 'court_pleading')}_{city_input.lower().replace(' ', '_')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    type="primary",
                    use_container_width=True
                )
        with c_actions2:
            st.download_button(
                label="📄 Download Text (.txt)",
                data=st.session_state["current_draft_text"],
                file_name=f"{meta.get('doc_type', 'court_pleading')}.txt",
                mime="text/plain",
                use_container_width=True
            )
            
        st.text_area(
            "Court Pleading (Editable & Ready to Print):",
            value=st.session_state["current_draft_text"],
            height=460,
            key="draft_display_area"
        )

# ==============================================================================
# TAB 3: LEGAL DOCUMENT & FIR OCR SCANNER WITH STATUTORY INTELLIGENCE
# ==============================================================================
with tab_scanner:
    st.markdown("""
    <div class="panel" style="margin-top:4px;">
      <div class="panel-title">📷 Legal Document & FIR OCR Scanner with Automated Intelligence</div>
      <div class="panel-copy">Upload scanned FIRs, police zimnis, court orders, stamped affidavits, or legal deeds in PDF or image format (.png, .jpg). Extracts full text using RapidOCR and produces an in-depth statutory legal audit with direct Word (.docx) export.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    col_s1, col_s2 = st.columns([1, 1.25], gap="large")

    with col_s1:
        st.markdown('<div class="section-title">1. Document Upload & Configuration</div>', unsafe_allow_html=True)
        uploaded_doc = st.file_uploader(
            "Select Scanned Document or Image",
            type=["pdf", "png", "jpg", "jpeg", "webp", "bmp", "tiff"],
            help="Upload clear PDF (scanned or digital) or photo of FIR / Court Order.",
            key="scanner_upload_widget"
        )

        c_lang, c_pages = st.columns(2)
        with c_lang:
            scan_lang_choice = st.selectbox(
                "Audit Output Language",
                options=[("both", "Bilingual (اردو + English)"), ("en", "English (Court Standard)"), ("ur", "Urdu (عدالتی اردو)")],
                format_func=lambda x: x[1],
                key="scan_lang_choice"
            )[0]
        total_doc_pages = 100
        if uploaded_doc is not None and uploaded_doc.name.lower().endswith(".pdf"):
            try:
                import fitz
                doc_tmp = fitz.open(stream=uploaded_doc.getvalue(), filetype="pdf")
                total_doc_pages = max(1, len(doc_tmp))
                doc_tmp.close()
            except Exception:
                pass
        with c_pages:
            slider_max = min(175, max(50, total_doc_pages))
            default_val = min(10, slider_max)
            max_pages_choice = st.slider(
                "Max pages to scan",
                min_value=1,
                max_value=slider_max,
                value=default_val,
                step=1,
                help=f"Number of pages to process (Document contains {total_doc_pages} pages)." if uploaded_doc else "Number of pages to process for multi-page PDFs."
            )

        if uploaded_doc is not None:
            st.caption(f"📁 **Filename:** `{uploaded_doc.name}` ({round(len(uploaded_doc.getvalue()) / 1024, 1)} KB)")
            if uploaded_doc.type.startswith("image/"):
                st.image(uploaded_doc, caption="Document Preview", use_container_width=True)
            elif uploaded_doc.name.lower().endswith(".pdf"):
                st.info("📄 PDF Document ready for digital text extraction & RapidOCR.")

        scan_btn = st.button("⚡ Scan & Generate Legal Intelligence", type="primary", use_container_width=True, disabled=uploaded_doc is None)

    with col_s2:
        st.markdown('<div class="section-title">2. Processing & Action Center</div>', unsafe_allow_html=True)

        if scan_btn and uploaded_doc is not None:
            file_bytes = uploaded_doc.getvalue()
            filename = uploaded_doc.name
            with st.status("Performing RapidOCR text recognition and running Pakistani statutory legal audit…", expanded=True) as s_status:
                try:
                    files_payload = {"file": (filename, file_bytes, uploaded_doc.type or "application/octet-stream")}
                    data_payload = {"language": scan_lang_choice, "max_pages": max_pages_choice}
                    resp = requests.post(SCAN_URL, headers={"key": API_KEY}, files=files_payload, data=data_payload, timeout=180)
                    if resp.ok:
                        st.session_state["current_scan_result"] = resp.json()
                        st.session_state["current_scan_filename"] = filename
                        s_status.update(label="Document successfully scanned and analyzed!", state="complete")
                    else:
                        s_status.update(label=f"Scanning failed: {error_detail(resp)}", state="error")
                except Exception as ex:
                    s_status.update(label=f"Connection error to scanner: {ex}", state="error")

        scan_res = st.session_state.get("current_scan_result")
        if scan_res:
            ext_info = scan_res.get("extraction", {})
            total_pgs = ext_info.get("total_pages", ext_info.get("page_count", 1))
            proc_pgs = ext_info.get("page_count", 1)
            is_partial = bool(total_pgs and proc_pgs < total_pgs)

            c_m1, c_m2, c_m3, c_m4 = st.columns(4)
            with c_m1:
                st.metric("Total Pages", f"{total_pgs} Pgs")
            with c_m2:
                st.metric("Scanned", f"Pgs 1–{proc_pgs}")
            with c_m3:
                st.metric("Words Extracted", f"{ext_info.get('word_count', 0):,}")
            with c_m4:
                st.metric("Method", ext_info.get("method", "OCR").upper())

            if is_partial:
                st.warning(
                    f"⚠️ **Partial Document Scan Notice:** Only pages 1 to {proc_pgs} of this {total_pgs}-page document were processed (based on 'Max pages to scan' slider). "
                    f"The remaining {total_pgs - proc_pgs} pages (which contain the substantive offences) were not processed. "
                    f"To scan further into the document, adjust the 'Max pages to scan' slider on the left."
                )

            analysis_text = scan_res.get("analysis", "")
            raw_text = ext_info.get("extracted_text", "")
            fname = st.session_state.get("current_scan_filename", "document")

            # Action Buttons: Download Word report, Download txt, Transfer to Court Drafter
            docx_report_bytes = None
            try:
                docx_report_bytes = LegalDocumentScanner().export_analysis_docx(
                    analysis_markdown=analysis_text,
                    extracted_text=raw_text,
                    filename=fname
                )
            except Exception as e_docx:
                st.caption(f"Docx preparation notice: {e_docx}")

            c_a1, c_a2, c_a3 = st.columns([1.2, 1.1, 1.4])
            with c_a1:
                if docx_report_bytes:
                    st.download_button(
                        "📥 Download Word Report (.docx)",
                        data=docx_report_bytes,
                        file_name=f"legal_audit_{Path(fname).stem}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        type="primary",
                        use_container_width=True
                    )
            with c_a2:
                st.download_button(
                    "📄 Raw Text (.txt)",
                    data=raw_text,
                    file_name=f"ocr_text_{Path(fname).stem}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with c_a3:
                if st.button("⚖️ Transfer to Court Drafter", use_container_width=True, help="Copies this text into the Court Drafter tab to immediately prepare a bail petition or reply!"):
                    st.session_state["pending_draft_facts"] = raw_text[:2500]
                    st.rerun()
        else:
            st.info("💡 Upload an image of an FIR, court notice, or scanned PDF on the left and click **'⚡ Scan & Generate Legal Intelligence'** to extract text and see an instant legal breakdown.")

    # Results Display area below
    if scan_res:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        tab_v1, tab_v2 = st.tabs(["📋 Statutory Legal Audit & Intelligence", "📝 Full Extracted Text (OCR)"])
        with tab_v1:
            st.markdown(scan_res.get("analysis", ""))
        with tab_v2:
            st.text_area("Extracted Document Text (Editable):", value=scan_res.get("extraction", {}).get("extracted_text", ""), height=400, key="scan_raw_text_display")

# Global Legal Disclaimer
st.markdown("<div style='height:28px'></div><div class='notice'><strong>Disclaimer:</strong> This system provides AI-assisted legal research and educational reference based on Pakistan statutory laws. It does not constitute formal legal counsel or create an advocate-client relationship. Always verify official Gazette notifications and consult a licensed High Court advocate for litigation.</div>", unsafe_allow_html=True)

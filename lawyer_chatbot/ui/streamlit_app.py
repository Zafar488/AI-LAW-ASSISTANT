"""Multilingual text and voice UI for Pakistan LegalBot."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

from app.language import LANGUAGE_NAMES, detect_language
from ui.voice_output import render_voice_player

API_BASE = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
if API_BASE.endswith("/ask"):
    API_BASE = API_BASE[:-4]
ASK_URL = f"{API_BASE}/ask"
TRANSCRIBE_URL = f"{API_BASE}/transcribe"
API_KEY = os.getenv("API_KEY", "")

st.set_page_config(
    page_title="Pakistan LegalBot Voice Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

for key, default in {
    "history": [],
    "draft": "",
    "last_audio_id": None,
    "detected_language": "en",
    "question_editor": "",
    "clear_editor_on_next_run": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.clear_editor_on_next_run:
    st.session_state.question_editor = ""
    st.session_state.draft = ""
    st.session_state.clear_editor_on_next_run = False

st.markdown(
    """
    <style>
      .stApp {background:linear-gradient(180deg,#f8fafc,#eef2ff)}
      .hero {padding:24px 28px;border-radius:18px;color:white;
             background:linear-gradient(135deg,#0f172a,#1e3a5f);margin-bottom:18px}
      .hero h1 {margin:0;font-size:30px}.hero p{margin:7px 0 0;color:#cbd5e1}
      .state {display:inline-flex;align-items:center;gap:8px;padding:6px 11px;
              border-radius:999px;background:#dcfce7;color:#166534;font-weight:700;font-size:12px}
      .dot {width:8px;height:8px;border-radius:50%;background:#22c55e;animation:pulse 1.2s infinite}
      @keyframes pulse {50%{transform:scale(1.6);opacity:.45}}
      [data-testid="stChatMessage"] {border:1px solid #e2e8f0;border-radius:14px;background:white}
      .privacy {font-size:12px;color:#64748b}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("🎙️ Voice settings")
    voice_enabled = st.toggle("Speak AI responses", value=True)
    auto_play = st.toggle("Auto-play new responses", value=False, disabled=not voice_enabled)
    top_k = st.slider("Legal references", 3, 15, 8)
    st.caption("Voice model is selected automatically from voices installed in your browser.")
    st.divider()
    st.subheader("Conversation")
    st.metric("Messages", len(st.session_state.history))
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.draft = ""
        st.rerun()
    st.divider()
    st.markdown(
        '<p class="privacy">🔒 Recording starts only when you press the microphone. '
        "Audio is held in memory for transcription and is not saved by this app.</p>",
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="hero">
      <h1>⚖️ Pakistan LegalBot Voice Assistant</h1>
      <p>Speak or type naturally in English, Urdu, or Hindi. The assistant replies in the same language.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

for item_index, item in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.markdown(item["question"])
    with st.chat_message("assistant"):
        st.markdown(item["answer"])
        st.caption(
            f'{LANGUAGE_NAMES.get(item["language"], "English")} · '
            f'{item["latency"]:.1f}s · {len(item["sources"])} source(s)'
        )
        if voice_enabled:
            should_autoplay = bool(
                auto_play
                and item_index == len(st.session_state.history) - 1
                and not item.get("autoplayed", False)
            )
            render_voice_player(
                item["answer"],
                item["language"],
                autoplay=should_autoplay,
            )
            if should_autoplay:
                item["autoplayed"] = True
        if item["sources"]:
            with st.expander("Legal sources"):
                for source in item["sources"]:
                    st.write(f"• {source}")

st.subheader("Ask a legal question")
audio = st.audio_input(
    "🎤 Record your question",
    help="Your browser requests microphone permission only after you activate recording. Press stop when finished.",
)

if audio is not None:
    audio_bytes = audio.getvalue()
    audio_id = hash(audio_bytes)
    if audio_id != st.session_state.last_audio_id:
        with st.status("🎧 Listening and transcribing…", expanded=True) as status:
            try:
                response = requests.post(
                    TRANSCRIBE_URL,
                    headers={"key": API_KEY},
                    files={"audio": (audio.name or "recording.wav", audio_bytes, audio.type)},
                    timeout=90,
                )
                if response.ok:
                    result = response.json()
                    st.session_state.draft = result["text"]
                    st.session_state.question_editor = result["text"]
                    st.session_state.detected_language = result["language"]
                    st.session_state.last_audio_id = audio_id
                    status.update(label="Transcript ready — edit it before sending", state="complete")
                else:
                    detail = response.json().get("detail", response.text)
                    status.update(label=f"I could not recognize that: {detail}", state="error")
            except requests.RequestException:
                status.update(label="Speech recognition is unavailable. Please type or try again.", state="error")

question = st.text_area(
    "Recognized speech / typed question",
    height=110,
    placeholder="Type here, or record above. You can edit recognized speech before sending.",
    key="question_editor",
)
language = detect_language(question) if question.strip() else st.session_state.detected_language
st.markdown(
    f'<span class="state"><span class="dot"></span>Detected: {LANGUAGE_NAMES.get(language, "English")}</span>',
    unsafe_allow_html=True,
)

send_col, clear_col, _ = st.columns([1.2, 1, 4])
with send_col:
    send = st.button("Send question", type="primary", use_container_width=True)
with clear_col:
    def clear_draft() -> None:
        st.session_state.question_editor = ""
        st.session_state.draft = ""

    st.button("Clear draft", use_container_width=True, on_click=clear_draft)

if send:
    cleaned = question.strip()
    if not cleaned:
        st.warning("Please type or record a question first.")
    elif not API_KEY:
        st.error("API_KEY is not configured.")
    else:
        with st.status("🧠 Searching laws and generating a response…", expanded=True) as status:
            started = time.perf_counter()
            try:
                response = requests.post(
                    ASK_URL,
                    headers={"key": API_KEY, "Content-Type": "application/json"},
                    json={"message": cleaned, "top_k": top_k},
                    timeout=180,
                )
                if response.ok:
                    result = response.json()
                    st.session_state.history.append(
                        {
                            "question": cleaned,
                            "answer": result["answer"],
                            "sources": result.get("sources", []),
                            "language": language,
                            "latency": time.perf_counter() - started,
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                        }
                    )
                    st.session_state.clear_editor_on_next_run = True
                    status.update(label="Response ready", state="complete")
                    st.rerun()
                else:
                    detail = response.json().get("detail", response.text)
                    status.update(label=f"Response failed: {detail}", state="error")
            except requests.Timeout:
                status.update(label="The response timed out. Please try a shorter question.", state="error")
            except requests.RequestException:
                status.update(label="Cannot connect to the legal assistant backend.", state="error")

if st.session_state.history:
    export = json.dumps(st.session_state.history, ensure_ascii=False, indent=2)
    st.download_button("Download conversation", export, "legalbot_conversation.json", "application/json")

st.caption("AI-generated legal information for education only; it is not a substitute for a licensed lawyer.")

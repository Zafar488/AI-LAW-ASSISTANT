"""Accessible browser-native text-to-speech controls for Streamlit."""

from __future__ import annotations

import html
import json

import streamlit.components.v1 as components

from app.language import TTS_LOCALES


def render_voice_player(text: str, language: str, autoplay: bool = False) -> None:
    safe_text = json.dumps(text, ensure_ascii=False)
    locale = json.dumps(TTS_LOCALES.get(language, "en-US"))
    auto = "true" if autoplay else "false"
    components.html(
        f"""
        <div class="voice-controls" aria-label="{html.escape('Voice response controls')}">
          <button id="play" aria-label="Read answer aloud">🔊 Play answer</button>
          <button id="stop" aria-label="Stop voice playback">⏹ Stop</button>
          <span id="status" role="status" aria-live="polite"></span>
        </div>
        <style>
          .voice-controls {{display:flex;gap:8px;align-items:center;font:14px system-ui}}
          button {{border:1px solid #cbd5e1;background:white;border-radius:8px;
                   padding:8px 12px;cursor:pointer;font-weight:600}}
          button:hover {{background:#eff6ff;border-color:#2563eb}}
          #status.speaking::before {{content:"";display:inline-block;width:8px;height:8px;
            margin-right:7px;border-radius:50%;background:#16a34a;animation:pulse 1s infinite}}
          @keyframes pulse {{50% {{transform:scale(1.7);opacity:.45}}}}
        </style>
        <script>
          const answer = {safe_text};
          const locale = {locale};
          const status = document.getElementById("status");
          function stopSpeech() {{
            speechSynthesis.cancel();
            status.textContent = "";
            status.className = "";
          }}
          function speak() {{
            stopSpeech();
            const utterance = new SpeechSynthesisUtterance(answer);
            utterance.lang = locale;
            const voices = speechSynthesis.getVoices();
            utterance.voice = voices.find(v => v.lang.toLowerCase() === locale.toLowerCase())
              || voices.find(v => v.lang.toLowerCase().startsWith(locale.slice(0,2).toLowerCase()));
            utterance.onstart = () => {{status.textContent="Speaking…";status.className="speaking"}};
            utterance.onend = stopSpeech;
            utterance.onerror = () => {{
              stopSpeech();
              status.textContent="Voice unavailable; text remains available.";
            }};
            speechSynthesis.speak(utterance);
          }}
          document.getElementById("play").onclick = speak;
          document.getElementById("stop").onclick = stopSpeech;
          window.addEventListener("beforeunload", stopSpeech);
          if ({auto}) setTimeout(speak, 250);
        </script>
        """,
        height=55,
    )

"""Persistent SQLite Conversation Database for Pakistan LegalBot.

Stores all user queries, legal research answers, retrieved sources,
detected languages, and execution timestamps.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from app.config import BASE_DIR

LOGGER = logging.getLogger("legalbot.db")

DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "conversations.db"


def get_db_connection() -> sqlite3.Connection:
    """Create a connection to the SQLite database with WAL mode."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    """Initialize conversations table and search indexes."""
    conn = get_db_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_query TEXT NOT NULL,
                detected_language TEXT,
                bot_answer TEXT NOT NULL,
                sources_json TEXT,
                top_k INTEGER,
                latency_sec REAL,
                client_ip TEXT
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conv_created ON conversations(created_at DESC);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);"
        )
        conn.commit()
    finally:
        conn.close()
    LOGGER.info("Conversation database initialized at %s", DB_PATH)


def save_conversation(
    user_query: str,
    bot_answer: str,
    sources: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
    detected_language: str | None = None,
    top_k: int = 8,
    latency_sec: float | None = None,
    client_ip: str | None = None,
) -> int:
    """Save a user question and legal answer pair to SQLite database."""
    init_db()
    sources_str = json.dumps(sources or [], ensure_ascii=False)
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO conversations (
                session_id, user_query, detected_language,
                bot_answer, sources_json, top_k, latency_sec, client_ip
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                user_query.strip(),
                detected_language or "en",
                bot_answer.strip(),
                sources_str,
                top_k,
                round(latency_sec, 3) if latency_sec is not None else None,
                client_ip,
            ),
        )
        conn.commit()
        return cursor.lastrowid or 0
    finally:
        conn.close()


def get_recent_conversations(
    limit: int = 50, offset: int = 0, session_id: str | None = None
) -> list[dict[str, Any]]:
    """Fetch recent conversations ordered by most recent first."""
    init_db()
    conn = get_db_connection()
    try:
        if session_id:
            cursor = conn.execute(
                """
                SELECT id, session_id, created_at, user_query, detected_language,
                       bot_answer, sources_json, top_k, latency_sec, client_ip
                FROM conversations
                WHERE session_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (session_id, limit, offset),
            )
        else:
            cursor = conn.execute(
                """
                SELECT id, session_id, created_at, user_query, detected_language,
                       bot_answer, sources_json, top_k, latency_sec, client_ip
                FROM conversations
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )

        rows = cursor.fetchall()
        results = []
        for r in rows:
            try:
                sources = json.loads(r["sources_json"]) if r["sources_json"] else []
            except Exception:
                sources = []
            results.append(
                {
                    "id": r["id"],
                    "session_id": r["session_id"],
                    "created_at": r["created_at"],
                    "user_query": r["user_query"],
                    "detected_language": r["detected_language"],
                    "bot_answer": r["bot_answer"],
                    "sources": sources,
                    "top_k": r["top_k"],
                    "latency_sec": r["latency_sec"],
                    "client_ip": r["client_ip"],
                }
            )
        return results
    finally:
        conn.close()


def get_conversation_count() -> int:
    """Return total number of saved conversations."""
    init_db()
    conn = get_db_connection()
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM conversations")
        return cursor.fetchone()[0]
    finally:
        conn.close()


def search_conversations(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Search conversations by query keyword matching."""
    init_db()
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            """
            SELECT id, session_id, created_at, user_query, detected_language,
                   bot_answer, sources_json, top_k, latency_sec
            FROM conversations
            WHERE user_query LIKE ? OR bot_answer LIKE ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (f"%{query}%", f"%{query}%", limit),
        )
        rows = cursor.fetchall()
        results = []
        for r in rows:
            try:
                sources = json.loads(r["sources_json"]) if r["sources_json"] else []
            except Exception:
                sources = []
            results.append(
                {
                    "id": r["id"],
                    "session_id": r["session_id"],
                    "created_at": r["created_at"],
                    "user_query": r["user_query"],
                    "detected_language": r["detected_language"],
                    "bot_answer": r["bot_answer"],
                    "sources": sources,
                    "top_k": r["top_k"],
                    "latency_sec": r["latency_sec"],
                }
            )
        return results
    finally:
        conn.close()

"""
logos/memory/store.py
─────────────────────
Persistent memory for LOGOS.

Stored at: ~/.logos/memory.db  (survives pip install / reinstall)

Schema
------
  user_profile       key/value personal preferences and identity
  queries            past research sessions with summaries
  tracked_entities   companies, technologies, topics the user returns to
  insights           bookmarked findings from past reports
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


# ── Storage location ──────────────────────────────────────────────────────────

def _logos_home() -> Path:
    """Returns ~/.logos, creating it if needed."""
    home = Path.home() / ".logos"
    home.mkdir(parents=True, exist_ok=True)
    return home


def _default_db() -> Path:
    return _logos_home() / "memory.db"


# ── Schema ────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS user_profile (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS queries (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    query      TEXT NOT NULL,
    summary    TEXT DEFAULT '',
    topics     TEXT DEFAULT '[]',      -- JSON list of strings
    entities   TEXT DEFAULT '[]',      -- JSON list of strings
    path_used  TEXT DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tracked_entities (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL UNIQUE,
    entity_type   TEXT DEFAULT 'general',
    mention_count INTEGER DEFAULT 1,
    last_seen     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS insights (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id   INTEGER REFERENCES queries(id),
    text       TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


# ── MemoryStore ───────────────────────────────────────────────────────────────

class MemoryStore:
    """
    Thread-safe SQLite memory for LOGOS.

    Usage
    -----
        mem = MemoryStore()          # uses ~/.logos/memory.db
        mem.initialize()
        profile = mem.get_user_profile()
        mem.set_profile(name="Raj", role="founder")
        qid = mem.save_query("NLP trends", summary="...", topics=["NLP","AI"])
        mem.save_insight(qid, "RAG adoption grew 3x in 2025")
    """

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else _default_db()
        self._conn: sqlite3.Connection | None = None

    # ── Connection management ─────────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def initialize(self) -> None:
        """Create tables if they don't exist."""
        self._get_conn().executescript(_DDL)
        self._get_conn().commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── User profile ──────────────────────────────────────────────────────────

    def get_user_profile(self) -> dict[str, str]:
        """Returns all profile key/value pairs as a dict."""
        rows = self._get_conn().execute(
            "SELECT key, value FROM user_profile"
        ).fetchall()
        return {r["key"]: r["value"] for r in rows}

    def set_profile(self, **kwargs: str) -> None:
        """Upsert profile fields.  e.g. set_profile(name='Raj', role='analyst')"""
        now = _now()
        conn = self._get_conn()
        for key, value in kwargs.items():
            conn.execute(
                "INSERT INTO user_profile(key, value, updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, str(value), now),
            )
        conn.commit()

    def is_first_run(self) -> bool:
        return len(self.get_user_profile()) == 0

    # ── Queries ───────────────────────────────────────────────────────────────

    def save_query(
        self,
        query: str,
        summary: str = "",
        topics: list[str] | None = None,
        entities: list[str] | None = None,
        path_used: str = "",
    ) -> int:
        """Persist a completed research query.  Returns the new row id."""
        conn = self._get_conn()
        cur = conn.execute(
            "INSERT INTO queries(query, summary, topics, entities, path_used, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (
                query,
                summary,
                json.dumps(topics or []),
                json.dumps(entities or []),
                path_used,
                _now(),
            ),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_recent_queries(self, n: int = 5) -> list[dict[str, Any]]:
        """Return the n most recent queries, newest first."""
        rows = self._get_conn().execute(
            "SELECT id, query, summary, topics, entities, created_at "
            "FROM queries ORDER BY id DESC LIMIT ?",
            (n,),
        ).fetchall()
        result = []
        for r in rows:
            result.append({
                "id":         r["id"],
                "query":      r["query"],
                "summary":    r["summary"],
                "topics":     json.loads(r["topics"] or "[]"),
                "entities":   json.loads(r["entities"] or "[]"),
                "created_at": r["created_at"],
            })
        return result

    def total_query_count(self) -> int:
        return self._get_conn().execute("SELECT COUNT(*) FROM queries").fetchone()[0]

    # ── Tracked entities ──────────────────────────────────────────────────────

    def track_entity(self, name: str, entity_type: str = "general") -> None:
        """Upsert a named entity (company, technology, market, etc.)."""
        self._get_conn().execute(
            "INSERT INTO tracked_entities(name, entity_type, mention_count, last_seen) "
            "VALUES (?,?,1,?) "
            "ON CONFLICT(name) DO UPDATE SET "
            "mention_count=mention_count+1, last_seen=excluded.last_seen",
            (name, entity_type, _now()),
        )
        self._get_conn().commit()

    def get_tracked_entities(self, top: int = 10) -> list[dict[str, Any]]:
        rows = self._get_conn().execute(
            "SELECT name, entity_type, mention_count FROM tracked_entities "
            "ORDER BY mention_count DESC LIMIT ?",
            (top,),
        ).fetchall()
        return [{"name": r["name"], "type": r["entity_type"], "count": r["mention_count"]} for r in rows]

    # ── Insights ──────────────────────────────────────────────────────────────

    def save_insight(self, query_id: int, text: str) -> None:
        """Bookmark a specific finding from a report."""
        self._get_conn().execute(
            "INSERT INTO insights(query_id, text, created_at) VALUES (?,?,?)",
            (query_id, text, _now()),
        )
        self._get_conn().commit()

    def get_recent_insights(self, n: int = 5) -> list[str]:
        rows = self._get_conn().execute(
            "SELECT text FROM insights ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [r["text"] for r in rows]

    # ── Context string for agent prompts ──────────────────────────────────────

    def build_context_string(self) -> str:
        """
        Builds a concise context block to prepend to agent prompts.
        Personalizes the research output based on stored memory.
        """
        profile  = self.get_user_profile()
        recent   = self.get_recent_queries(3)
        entities = self.get_tracked_entities(5)
        insights = self.get_recent_insights(3)

        lines: list[str] = ["=== RESEARCH CONTEXT FROM MEMORY ==="]

        if profile:
            name = profile.get("name", "")
            role = profile.get("role", "")
            org  = profile.get("organization", "")
            domain = profile.get("domain", "")
            depth  = profile.get("depth_preference", "detailed")

            if name:
                lines.append(f"Researcher: {name}" + (f" ({role})" if role else ""))
            if org:
                lines.append(f"Organization: {org}")
            if domain:
                lines.append(f"Primary domain: {domain}")
            lines.append(f"Report depth preference: {depth}")

        if recent:
            lines.append("\nRecent research topics:")
            for q in recent:
                lines.append(f"  - {q['query'][:80]}")

        if entities:
            top = ", ".join(e["name"] for e in entities[:5])
            lines.append(f"\nFrequently researched entities: {top}")

        if insights:
            lines.append("\nPast insights the researcher has bookmarked:")
            for ins in insights:
                lines.append(f"  - {ins[:120]}")

        lines.append("=== END CONTEXT ===\n")
        return "\n".join(lines)

    # ── Session number ────────────────────────────────────────────────────────

    def session_number(self) -> int:
        """Returns number of distinct research sessions so far (total queries + 1)."""
        return self.total_query_count() + 1


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")

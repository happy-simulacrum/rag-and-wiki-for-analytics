"""SQLite: метаданные чанков, FTS5-индекс, состояние индексации, лог вопросов."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeqa.indexer.chunker import Chunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    module TEXT NOT NULL,
    relpath TEXT NOT NULL,
    language TEXT NOT NULL,
    symbol TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_project ON chunks(project, relpath);

CREATE TABLE IF NOT EXISTS project_state (
    project TEXT PRIMARY KEY,
    last_commit TEXT,
    indexed_at REAL,
    chunk_count INTEGER
);

CREATE TABLE IF NOT EXISTS question_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    project TEXT NOT NULL,
    question TEXT NOT NULL,
    embedding TEXT NOT NULL,
    answered_ok INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_question_log_project ON question_log(project);
"""

_FTS_TRIGRAM = (
    "CREATE VIRTUAL TABLE chunks_fts USING fts5("
    "text, symbol, relpath, chunk_id UNINDEXED, tokenize='trigram')"
)
_FTS_UNICODE = (
    "CREATE VIRTUAL TABLE chunks_fts USING fts5("
    "text, symbol, relpath, chunk_id UNINDEXED)"
)


class ChunkStore:
    def __init__(self, db_path: str | Path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: FastAPI обслуживает запросы в других потоках
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.Lock()
        self._db.executescript(_SCHEMA)
        self._fts_tokenizer = self._init_fts()

    def _init_fts(self) -> str:
        exists = self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        ).fetchone()
        if exists:
            row = self._db.execute(
                "SELECT sql FROM sqlite_master WHERE name='chunks_fts'"
            ).fetchone()
            return "trigram" if row and "trigram" in row[0] else "unicode61"
        try:
            self._db.execute(_FTS_TRIGRAM)
            return "trigram"
        except sqlite3.OperationalError:
            self._db.execute(_FTS_UNICODE)
            return "unicode61"

    def close(self) -> None:
        self._db.close()

    # ---- запись ----

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        rows = [
            (c.chunk_id, c.project, c.module, c.relpath, c.language, c.symbol,
             c.start_line, c.end_line, c.text)
            for c in chunks
        ]
        with self._lock, self._db:
            self._db.executemany(
                "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?,?)", rows
            )
            self._db.executemany(
                "DELETE FROM chunks_fts WHERE chunk_id = ?", [(c.chunk_id,) for c in chunks]
            )
            self._db.executemany(
                "INSERT INTO chunks_fts (text, symbol, relpath, chunk_id) VALUES (?,?,?,?)",
                [(c.text, c.symbol, c.relpath, c.chunk_id) for c in chunks],
            )

    def delete_files(self, project: str, relpaths: list[str]) -> list[str]:
        """Удалить все чанки файлов; возвращает удалённые chunk_id."""
        if not relpaths:
            return []
        ids: list[str] = []
        with self._lock, self._db:
            for rel in relpaths:
                found = self._db.execute(
                    "SELECT chunk_id FROM chunks WHERE project=? AND relpath=?",
                    (project, rel),
                ).fetchall()
                ids.extend(r[0] for r in found)
                self._db.execute(
                    "DELETE FROM chunks WHERE project=? AND relpath=?", (project, rel)
                )
            self._db.executemany(
                "DELETE FROM chunks_fts WHERE chunk_id=?", [(i,) for i in ids]
            )
        return ids

    def drop_project(self, project: str) -> None:
        with self._lock, self._db:
            self._db.execute(
                "DELETE FROM chunks_fts WHERE chunk_id IN "
                "(SELECT chunk_id FROM chunks WHERE project=?)", (project,)
            )
            self._db.execute("DELETE FROM chunks WHERE project=?", (project,))
            self._db.execute("DELETE FROM project_state WHERE project=?", (project,))

    # ---- чтение ----

    def get_chunks(self, chunk_ids: list[str]) -> list[dict]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self._db.execute(
            f"SELECT chunk_id, project, module, relpath, symbol, start_line, end_line, text "
            f"FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        by_id = {
            r[0]: {
                "chunk_id": r[0], "project": r[1], "module": r[2], "relpath": r[3],
                "symbol": r[4], "start_line": r[5], "end_line": r[6], "text": r[7],
            }
            for r in rows
        }
        return [by_id[i] for i in chunk_ids if i in by_id]

    def lexical_search(self, project: str, terms: list[str], limit: int = 40) -> list[str]:
        """FTS5 по терминам (идентификаторы/слова). Возвращает chunk_id по релевантности."""
        terms = [t for t in (t.strip() for t in terms) if len(t) >= 3]
        if not terms:
            return []
        query = " OR ".join(f'"{t}"' for t in terms)
        try:
            rows = self._db.execute(
                "SELECT c.chunk_id FROM chunks_fts f "
                "JOIN chunks c ON c.chunk_id = f.chunk_id "
                "WHERE chunks_fts MATCH ? AND c.project = ? "
                "ORDER BY rank LIMIT ?",
                (query, project, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [r[0] for r in rows]

    # ---- состояние индексации ----

    def get_state(self, project: str) -> dict | None:
        row = self._db.execute(
            "SELECT last_commit, indexed_at, chunk_count FROM project_state WHERE project=?",
            (project,),
        ).fetchone()
        if not row:
            return None
        return {"last_commit": row[0], "indexed_at": row[1], "chunk_count": row[2]}

    def set_state(self, project: str, last_commit: str, chunk_count: int) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO project_state VALUES (?,?,?,?)",
                (project, last_commit, time.time(), chunk_count),
            )

    def count_chunks(self, project: str) -> int:
        return self._db.execute(
            "SELECT COUNT(*) FROM chunks WHERE project=?", (project,)
        ).fetchone()[0]

    # ---- лог вопросов (питает FAQ) ----

    def log_question(self, project: str, question: str, embedding: list[float], ok: bool) -> None:
        import json

        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO question_log (ts, project, question, embedding, answered_ok) "
                "VALUES (?,?,?,?,?)",
                (time.time(), project, question, json.dumps(embedding), 1 if ok else 0),
            )

    def get_questions(self, project: str) -> list[dict]:
        import json

        rows = self._db.execute(
            "SELECT id, ts, question, embedding FROM question_log "
            "WHERE project=? AND answered_ok=1 ORDER BY ts",
            (project,),
        ).fetchall()
        return [
            {"id": r[0], "ts": r[1], "question": r[2], "embedding": json.loads(r[3])}
            for r in rows
        ]

    def has_relpath(self, project: str, relpath: str) -> bool:
        return bool(
            self._db.execute(
                "SELECT 1 FROM chunks WHERE project=? AND relpath=? LIMIT 1",
                (project, relpath),
            ).fetchone()
        )

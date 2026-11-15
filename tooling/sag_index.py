"""Transactional SQLite FTS5 storage for source passages, without embeddings."""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import unicodedata
from contextlib import contextmanager
from pathlib import Path


STOPWORDS = set("the a an of in on at to for and or is are was were be by with from which what who how when where does do did that this it as 的 了 是 在 和 与 或 请 我 你 什么 如何 怎么 哪个 是否".split())
WORDS = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*|[\u3400-\u9fff]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Normalize Latin words and Chinese bigrams/trigrams deterministically.

    FTS5's unicode61 does not segment Chinese sentences. Indexing these ngrams
    makes Chinese queries usable without a downloaded model or dictionary.
    """
    result = []
    for word in WORDS.findall(unicodedata.normalize("NFKC", text).lower()):
        if "\u3400" <= word[0] <= "\u9fff":
            if len(word) <= 3:
                result.append(word)
            for size in (2, 3):
                result.extend(word[i:i + size] for i in range(len(word) - size + 1))
        else:
            result.append(word)
            result.extend(part for part in re.split(r"[._-]", word) if part != word)
    return [word for word in result if word not in STOPWORDS]


def match_expression(terms: list[str]) -> str:
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in dict.fromkeys(terms))


class SQLiteKnowledgeIndex:
    """Small collection-compatible store, with document transactions and BM25."""

    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=30000")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS passages (
                rowid INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL, content TEXT NOT NULL, metadata TEXT NOT NULL,
                document_id TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS passages_document ON passages(document_id);
            CREATE VIRTUAL TABLE IF NOT EXISTS passage_fts USING fts5(title, body);
            CREATE VIRTUAL TABLE IF NOT EXISTS passage_vocab USING fts5vocab(passage_fts, 'row');
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        self.db.commit()

    @contextmanager
    def transaction(self):
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise

    def upsert(self, *, ids, documents, metadatas):
        # The caller holds transaction() for the complete document replacement.
        for passage_id, content, metadata in zip(ids, documents, metadatas):
            title = metadata.get("title", "")
            self.db.execute(
                "INSERT INTO passages(id,title,content,metadata,document_id) VALUES(?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET title=excluded.title,content=excluded.content,"
                "metadata=excluded.metadata,document_id=excluded.document_id",
                (passage_id, title, content, json.dumps(metadata, ensure_ascii=False), metadata.get("document_id", "")),
            )
            rowid = self.db.execute("SELECT rowid FROM passages WHERE id=?", (passage_id,)).fetchone()[0]
            self.db.execute("DELETE FROM passage_fts WHERE rowid=?", (rowid,))
            self.db.execute("INSERT INTO passage_fts(rowid,title,body) VALUES(?,?,?)",
                            (rowid, " ".join(tokenize(title)), " ".join(tokenize(content))))

    def get(self, where=None, include=None, *, limit=None, offset=0):
        where = where or {}
        with self.lock:
            # All predicate names originate in KnowledgeBase, never in SQL input.
            field = "document_id" if "document_id" in where else "title"
            rows = self.db.execute(f"SELECT * FROM passages WHERE {field}=?", (where[field],)).fetchall() if where else self.db.execute("SELECT * FROM passages ORDER BY rowid").fetchall()
            selected = [(row, json.loads(row["metadata"])) for row in rows]
            selected = [(row, meta) for row, meta in selected if all(meta.get(key) == value for key, value in where.items())]
            if limit is not None:
                selected = selected[offset:offset + limit]
            return {"ids": [row["id"] for row, _ in selected], "documents": [row["content"] for row, _ in selected], "metadatas": [meta for _, meta in selected]}

    def delete(self, *, ids):
        for passage_id in ids:
            self.db.execute("DELETE FROM passage_fts WHERE rowid=(SELECT rowid FROM passages WHERE id=?)", (passage_id,))
            self.db.execute("DELETE FROM passages WHERE id=?", (passage_id,))

    def count(self):
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM passages").fetchone()[0]

    def retrieve(self, terms: list[str], limit: int, excluded=()) -> list[dict]:
        expression = match_expression(terms[:80])
        if not expression:
            return []
        with self.lock:
            # Seen passages count towards LIMIT only in the SQL exclusion clause.
            excluded = list(excluded)
            exclusion = " AND p.id NOT IN (" + ",".join("?" for _ in excluded) + ")" if excluded else ""
            rows = self.db.execute(
                "SELECT p.*, bm25(passage_fts, 2.0, 1.0) AS rank FROM passage_fts "
                "JOIN passages p ON p.rowid=passage_fts.rowid WHERE passage_fts MATCH ?"
                + exclusion + " ORDER BY rank,p.id LIMIT ?", [expression, *excluded, limit],
            ).fetchall()
        result = []
        for index, row in enumerate(rows):
            meta = json.loads(row["metadata"])
            result.append({"id": row["id"], "title": row["title"], "content": row["content"],
                           "document_id": meta.get("document_id", ""), "version": meta.get("version", ""),
                           "chunk": meta.get("chunk_index", 0), "score": 1 / (index + 1),
                           "lexical_score": -row["rank"], "fusion_score": 1 / (61 + index)})
        return result

    def frequencies(self, terms: list[str]) -> dict[str, int]:
        with self.lock:
            return {term: (self.db.execute("SELECT doc FROM passage_vocab WHERE term=?", (term,)).fetchone() or [0])[0] for term in terms}

    def setting(self, key: str):
        with self.lock:
            row = self.db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
            return json.loads(row[0]) if row else None

    def set_setting(self, key: str, value):
        self.db.execute("INSERT INTO settings VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, json.dumps(value)))

    def close(self):
        with self.lock:
            self.db.close()

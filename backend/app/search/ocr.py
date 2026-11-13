"""Keyword search over OCR text (SQLite FTS5 with a LIKE fallback).

User text is never passed to FTS5 verbatim: it is split into word tokens,
each token is quoted and prefix-matched (``"grad"*``) and tokens are OR-ed,
so punctuation or FTS operators in the query cannot cause syntax errors.
BM25 relevance (more negative = better in SQLite) is mapped to [0, 1] by
dividing by the best match of the query. When FTS finds nothing, a
case-insensitive substring ``LIKE`` match catches fragments inside words
with a fixed score of :data:`LIKE_SCORE`.
"""

from __future__ import annotations

import re

from ..db.schema import get_db

LIKE_SCORE = 0.5
MAX_MATCHES = 1000
SNIPPET_CHARS = 120

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens of ``text`` (deduplicated, order kept)."""
    return list(dict.fromkeys(t.lower() for t in _TOKEN_RE.findall(text)))


def build_fts_query(text: str) -> str:
    """Sanitised FTS5 MATCH expression: quoted prefix tokens joined by OR."""
    return " OR ".join(f'"{t}"*' for t in tokenize(text))


def search_ocr(text: str, limit: int = MAX_MATCHES) -> dict[int, float]:
    """Return ``{image_id: score in [0, 1]}`` for COMPLETED images matching ``text``."""
    fts = build_fts_query(text)
    if not fts:
        return {}
    with get_db() as conn:
        rows = conn.execute(
            """SELECT f.rowid AS image_id, bm25(image_ocr_fts) AS rank
               FROM image_ocr_fts f JOIN images i ON i.image_id = f.rowid
               WHERE image_ocr_fts MATCH ? AND i.status = 'COMPLETED'
               ORDER BY rank LIMIT ?""",
            (fts, limit),
        ).fetchall()
        if rows:
            best = max(-r["rank"] for r in rows) or 1.0
            return {int(r["image_id"]): max(0.0, min(1.0, -r["rank"] / best)) for r in rows}
        like = f"%{text.strip()}%"
        rows = conn.execute(
            """SELECT o.image_id FROM image_ocr o JOIN images i ON i.image_id = o.image_id
               WHERE o.raw_text LIKE ? AND i.status = 'COMPLETED' LIMIT ?""",
            (like, limit),
        ).fetchall()
    return {int(r["image_id"]): LIKE_SCORE for r in rows}


def make_snippet(raw_text: str | None, query: str | None = None, width: int = SNIPPET_CHARS) -> str:
    """A ``width``-char excerpt of ``raw_text``, centred on the first query hit if any."""
    text = raw_text or ""
    if len(text) <= width:
        return text
    start = 0
    if query:
        lower = text.lower()
        hits = [p for p in (lower.find(t) for t in tokenize(query)) if p >= 0]
        if hits:
            start = max(0, min(min(hits) - width // 3, len(text) - width))
    snippet = text[start : start + width]
    return ("…" if start > 0 else "") + snippet + ("…" if start + width < len(text) else "")

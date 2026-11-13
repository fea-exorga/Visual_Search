"""
SQLite3 database schema and connection manager.

Uses WAL mode for concurrent reads and FTS5 for full-text search on OCR text.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from ..config import get_settings

# ── SQL DDL ───────────────────────────────────────────────────────────

_SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS watched_folders (
    folder_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT    NOT NULL UNIQUE,
    enabled     BOOLEAN NOT NULL DEFAULT 1,
    last_scan_at TIMESTAMP,
    created_at  TIMESTAMP NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS images (
    image_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path       TEXT    NOT NULL UNIQUE,
    folder_id       INTEGER NOT NULL REFERENCES watched_folders(folder_id),
    sha256          TEXT,
    width           INTEGER,
    height          INTEGER,
    file_size       INTEGER,
    mtime_ns        INTEGER,
    format          TEXT,
    status          TEXT    NOT NULL DEFAULT 'PENDING'
                        CHECK (status IN (
                            'PENDING', 'SCANNING', 'COLOR_DONE',
                            'OCR_DONE', 'EMBEDDING_DONE', 'COMPLETED', 'ERROR'
                        )),
    error_message   TEXT,
    embedding_row_id INTEGER,
    created_at      TIMESTAMP NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at      TIMESTAMP NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS image_colors (
    color_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id    INTEGER NOT NULL REFERENCES images(image_id) ON DELETE CASCADE,
    rank        INTEGER NOT NULL CHECK (rank BETWEEN 1 AND 5),
    hex_code    TEXT    NOT NULL,
    r           INTEGER NOT NULL,
    g           INTEGER NOT NULL,
    b           INTEGER NOT NULL,
    weight      REAL    NOT NULL,
    UNIQUE (image_id, rank)
);

CREATE TABLE IF NOT EXISTS image_ocr (
    image_id        INTEGER PRIMARY KEY REFERENCES images(image_id) ON DELETE CASCADE,
    raw_text        TEXT,
    mean_confidence REAL,
    word_count      INTEGER
);

-- FTS5 virtual table backed by image_ocr
CREATE VIRTUAL TABLE IF NOT EXISTS image_ocr_fts USING fts5(
    raw_text,
    content='image_ocr',
    content_rowid='image_id'
);

-- Triggers to keep FTS5 index in sync with image_ocr
CREATE TRIGGER IF NOT EXISTS image_ocr_ai AFTER INSERT ON image_ocr BEGIN
    INSERT INTO image_ocr_fts(rowid, raw_text) VALUES (new.image_id, new.raw_text);
END;

CREATE TRIGGER IF NOT EXISTS image_ocr_ad AFTER DELETE ON image_ocr BEGIN
    INSERT INTO image_ocr_fts(image_ocr_fts, rowid, raw_text)
        VALUES ('delete', old.image_id, old.raw_text);
END;

CREATE TRIGGER IF NOT EXISTS image_ocr_au AFTER UPDATE ON image_ocr BEGIN
    INSERT INTO image_ocr_fts(image_ocr_fts, rowid, raw_text)
        VALUES ('delete', old.image_id, old.raw_text);
    INSERT INTO image_ocr_fts(rowid, raw_text) VALUES (new.image_id, new.raw_text);
END;

-- Dense image embeddings (one row per image, L2-normalised float32 blob).
CREATE TABLE IF NOT EXISTS image_embeddings (
    image_id    INTEGER PRIMARY KEY REFERENCES images(image_id) ON DELETE CASCADE,
    model       TEXT    NOT NULL,
    dim         INTEGER NOT NULL,
    vector      BLOB    NOT NULL,
    updated_at  TIMESTAMP NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Perceptual hashes for exact / near duplicate detection (Milestone 5).
CREATE TABLE IF NOT EXISTS image_hashes (
    image_id    INTEGER PRIMARY KEY REFERENCES images(image_id) ON DELETE CASCADE,
    phash       TEXT,
    dhash       TEXT
);

-- 2D/3D projections + cluster assignment (Milestone 3).
CREATE TABLE IF NOT EXISTS image_projections (
    image_id    INTEGER NOT NULL REFERENCES images(image_id) ON DELETE CASCADE,
    method      TEXT    NOT NULL,   -- 'umap' | 'tsne' | 'pca'
    dims        INTEGER NOT NULL,   -- 2 | 3
    x REAL NOT NULL, y REAL NOT NULL, z REAL,
    cluster_id  INTEGER,
    PRIMARY KEY (image_id, method, dims)
);

CREATE TABLE IF NOT EXISTS clusters (
    cluster_id  INTEGER NOT NULL,
    method      TEXT    NOT NULL,
    dims        INTEGER NOT NULL,
    label       TEXT,
    size        INTEGER NOT NULL,
    color       TEXT,
    cx REAL, cy REAL, cz REAL,
    PRIMARY KEY (cluster_id, method, dims)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_images_sha256    ON images(sha256);
CREATE INDEX IF NOT EXISTS idx_images_folder_id ON images(folder_id);
CREATE INDEX IF NOT EXISTS idx_images_status    ON images(status);
CREATE INDEX IF NOT EXISTS idx_colors_image_id  ON image_colors(image_id);
"""


# ── Connection helpers ────────────────────────────────────────────────

def _make_connection(db_path: str | Path) -> sqlite3.Connection:
    """Create a new SQLite connection with recommended pragmas."""
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str | Path | None = None) -> None:
    """
    Initialise the database: create all tables, indexes, triggers, and the
    FTS5 virtual table if they do not yet exist.
    """
    if db_path is None:
        db_path = get_settings().DB_PATH
    conn = _make_connection(db_path)
    try:
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


@contextmanager
def get_db(db_path: str | Path | None = None) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager that yields a SQLite connection.

    Usage::

        with get_db() as conn:
            conn.execute("SELECT ...")
    """
    if db_path is None:
        db_path = get_settings().DB_PATH
    conn = _make_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

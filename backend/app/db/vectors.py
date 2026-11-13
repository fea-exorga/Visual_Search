"""Embedding persistence in SQLite.

Vectors live in ``image_embeddings`` keyed by ``image_id`` so there is never
any row-offset drift between the DB and a side-car ``.npy`` file. The FAISS
index (Milestone 2) is a cache rebuilt from this table.
"""

from __future__ import annotations

import numpy as np

from .schema import get_db


def save_vectors(image_ids: list[int], vectors: np.ndarray, model: str) -> None:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    assert len(image_ids) == vectors.shape[0], "ids / vectors length mismatch"
    dim = int(vectors.shape[1])
    with get_db() as conn:
        conn.executemany(
            """INSERT INTO image_embeddings (image_id, model, dim, vector)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(image_id) DO UPDATE SET model=excluded.model, dim=excluded.dim,
                   vector=excluded.vector,
                   updated_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now')""",
            [(int(i), model, dim, v.tobytes()) for i, v in zip(image_ids, vectors)],
        )


def load_vectors(model: str | None = None, image_ids: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ids int64 (N,), vectors float32 (N, dim))`` ordered by image_id.

    Only images whose status is COMPLETED are returned. ``model`` filters to
    vectors produced by one encoder (mixing encoders is meaningless).
    """
    sql = """SELECT e.image_id, e.dim, e.vector FROM image_embeddings e
             JOIN images i ON i.image_id = e.image_id
             WHERE i.status = 'COMPLETED'"""
    params: list = []
    if model is not None:
        sql += " AND e.model = ?"
        params.append(model)
    if image_ids is not None:
        sql += f" AND e.image_id IN ({','.join('?' * len(image_ids))})"
        params.extend(int(i) for i in image_ids)
    sql += " ORDER BY e.image_id"
    with get_db() as conn:
        rows = conn.execute(sql, params).fetchall()
    if not rows:
        return np.empty((0,), dtype=np.int64), np.empty((0, 0), dtype=np.float32)
    ids = np.array([r["image_id"] for r in rows], dtype=np.int64)
    vecs = np.stack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
    return ids, vecs


def get_vector(image_id: int) -> np.ndarray | None:
    with get_db() as conn:
        row = conn.execute("SELECT vector FROM image_embeddings WHERE image_id = ?", (image_id,)).fetchone()
    return None if row is None else np.frombuffer(row["vector"], dtype=np.float32).copy()


def current_models() -> list[tuple[str, int]]:
    """``[(model, count), …]`` most common first."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT model, COUNT(*) AS n FROM image_embeddings GROUP BY model ORDER BY n DESC"
        ).fetchall()
    return [(r["model"], r["n"]) for r in rows]

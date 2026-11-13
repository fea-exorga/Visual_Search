"""Galaxy orchestration: load vectors → project → cluster → label → persist.

Persistence: ``image_projections`` (one row per point) and ``clusters`` (one
row per cluster) are replaced atomically per ``(method, dims)``. Run metadata
(``computed_at``, model, parameters) lives in a small side table
``galaxy_runs`` created lazily by this module.

Staleness is evaluated on read (no post-index hook needed): a stored galaxy
is ``stale`` when the set of embedded, COMPLETED image ids for the current
model differs from the projected ids, when the dominant model changed, or
when any embedding was rewritten after ``computed_at``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from typing import Any

import numpy as np

from ..db.schema import get_db
from ..db.vectors import current_models, load_vectors
from .cluster import cluster_points
from .labels import label_clusters
from .reduce import METHODS, project

logger = logging.getLogger(__name__)

_compute_lock = threading.Lock()

_META_SQL = """
CREATE TABLE IF NOT EXISTS galaxy_runs (
    method       TEXT    NOT NULL,
    dims         INTEGER NOT NULL,
    model        TEXT,
    n_points     INTEGER NOT NULL,
    params       TEXT,
    computed_at  TEXT    NOT NULL,
    PRIMARY KEY (method, dims)
)
"""


class GalaxyError(ValueError):
    """Invalid request (``status`` 400) or not enough data (``status`` 409)."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def _validate(method: str, dims: int) -> None:
    if method not in METHODS:
        raise GalaxyError(f"method must be one of {list(METHODS)}", 400)
    if dims not in (2, 3):
        raise GalaxyError("dims must be 2 or 3", 400)


def _ensure_meta(conn: sqlite3.Connection) -> None:
    conn.execute(_META_SQL)


def _dominant_model() -> str | None:
    models = current_models()
    return models[0][0] if models else None


def _load_ocr(ids: np.ndarray) -> list[str | None]:
    if len(ids) == 0:
        return []
    with get_db() as conn:
        rows = conn.execute("SELECT image_id, raw_text FROM image_ocr").fetchall()
    by_id = {int(r["image_id"]): r["raw_text"] for r in rows}
    return [by_id.get(int(i)) for i in ids]


def _cluster_names(mat: np.ndarray, labels: np.ndarray, model: str | None,
                   ocr: list[str | None]) -> list[str]:
    """Zero-shot names; generic ``theme N`` names if the encoder is unusable."""
    k = int(labels.max()) + 1 if labels.size else 0
    fallback = [f"theme {c + 1}" for c in range(k)]
    try:
        from ..embeddings import get_encoder

        enc = get_encoder()
        if model is not None and getattr(enc, "name", None) != model:
            logger.info("Encoder %s != vector model %s; using generic cluster names",
                        getattr(enc, "name", "?"), model)
            return fallback
        if int(getattr(enc, "dim", mat.shape[1])) != mat.shape[1]:
            return fallback
        return label_clusters(mat, labels, enc, ocr)
    except Exception:
        logger.exception("Zero-shot cluster labelling failed; using generic names")
        return fallback


def _is_stale(conn: sqlite3.Connection, method: str, dims: int,
              run: sqlite3.Row | None) -> bool:
    model = _dominant_model()
    if run is not None and model != run["model"]:
        return True
    projected = {int(r[0]) for r in conn.execute(
        "SELECT image_id FROM image_projections WHERE method = ? AND dims = ?", (method, dims))}
    sql = ("SELECT e.image_id FROM image_embeddings e JOIN images i ON i.image_id = e.image_id "
           "WHERE i.status = 'COMPLETED'")
    params: list[Any] = []
    if model is not None:
        sql += " AND e.model = ?"
        params.append(model)
    embedded = {int(r[0]) for r in conn.execute(sql, params)}
    if embedded != projected:
        return True
    if run is not None:
        newer = conn.execute(
            "SELECT 1 FROM image_embeddings WHERE updated_at > ? LIMIT 1", (run["computed_at"],)
        ).fetchone()
        if newer is not None:
            return True
    return False


def get_galaxy(method: str = "umap", dims: int = 2) -> dict[str, Any] | None:
    """Return the stored galaxy payload (PRD §6.4) or ``None`` if never computed."""
    _validate(method, dims)
    with get_db() as conn:
        _ensure_meta(conn)
        run = conn.execute(
            "SELECT * FROM galaxy_runs WHERE method = ? AND dims = ?", (method, dims)
        ).fetchone()
        prow = conn.execute(
            "SELECT image_id, x, y, z, cluster_id FROM image_projections "
            "WHERE method = ? AND dims = ? ORDER BY image_id", (method, dims)
        ).fetchall()
        if run is None and not prow:
            return None
        crow = conn.execute(
            "SELECT cluster_id, label, size, color, cx, cy, cz FROM clusters "
            "WHERE method = ? AND dims = ? ORDER BY cluster_id", (method, dims)
        ).fetchall()
        stale = _is_stale(conn, method, dims, run)

    three = dims == 3
    points = []
    for r in prow:
        p: dict[str, Any] = {"image_id": int(r["image_id"]), "x": float(r["x"]), "y": float(r["y"])}
        if three:
            p["z"] = float(r["z"] or 0.0)
        p["cluster_id"] = None if r["cluster_id"] is None else int(r["cluster_id"])
        points.append(p)
    clusters = []
    for r in crow:
        centroid = [float(r["cx"] or 0.0), float(r["cy"] or 0.0)]
        if three:
            centroid.append(float(r["cz"] or 0.0))
        clusters.append({
            "cluster_id": int(r["cluster_id"]),
            "label": r["label"],
            "size": int(r["size"]),
            "color": r["color"],
            "centroid": centroid,
        })
    return {
        "method": method,
        "dims": dims,
        "points": points,
        "clusters": clusters,
        "computed_at": run["computed_at"] if run is not None else None,
        "stale": stale,
    }


def compute_galaxy(method: str = "umap", dims: int = 2, *, n_neighbors: int | None = None,
                   min_dist: float | None = None, perplexity: float | None = None,
                   n_clusters: int | None = None, **extra: Any) -> dict[str, Any]:
    """Project, cluster, label and persist the library; return the GET payload.

    Raises:
        GalaxyError: invalid method/dims (400) or fewer than 2 embedded images (409).
    """
    _validate(method, dims)
    if n_clusters is not None and int(n_clusters) < 1:
        raise GalaxyError("n_clusters must be >= 1", 400)
    with _compute_lock:
        model = _dominant_model()
        ids, mat = load_vectors(model=model)
        if len(ids) < 2:
            raise GalaxyError(f"need at least 2 embedded images, have {len(ids)}", 409)

        proj_params = {"n_neighbors": n_neighbors, "min_dist": min_dist, "perplexity": perplexity, **extra}
        coords = project(mat, method, dims, **proj_params)
        result = cluster_points(coords, n_clusters=n_clusters)
        names = _cluster_names(mat, result.labels, model, _load_ocr(ids))

        z_of = (lambda i: float(coords[i, 2])) if dims == 3 else (lambda i: None)
        point_rows = [
            (int(ids[i]), method, dims, float(coords[i, 0]), float(coords[i, 1]), z_of(i),
             int(result.labels[i]))
            for i in range(len(ids))
        ]
        cluster_rows = [
            (c, method, dims, names[c], int(result.sizes[c]), result.colors[c],
             float(result.centroids[c, 0]), float(result.centroids[c, 1]),
             float(result.centroids[c, 2]) if dims == 3 else None)
            for c in range(result.k)
        ]
        stored_params = {k: v for k, v in {**proj_params, "n_clusters": n_clusters}.items() if v is not None}
        with get_db() as conn:  # single transaction: commit on exit, rollback on error
            _ensure_meta(conn)
            conn.execute("DELETE FROM image_projections WHERE method = ? AND dims = ?", (method, dims))
            conn.execute("DELETE FROM clusters WHERE method = ? AND dims = ?", (method, dims))
            conn.executemany(
                "INSERT INTO image_projections (image_id, method, dims, x, y, z, cluster_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)", point_rows)
            conn.executemany(
                "INSERT INTO clusters (cluster_id, method, dims, label, size, color, cx, cy, cz) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", cluster_rows)
            conn.execute(
                "INSERT OR REPLACE INTO galaxy_runs (method, dims, model, n_points, params, computed_at) "
                "VALUES (?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))",
                (method, dims, model, len(ids), json.dumps(stored_params)))
        logger.info("Galaxy %s/%dD: %d points, %d clusters (silhouette=%s)",
                    method, dims, len(ids), result.k, result.silhouette)

    payload = get_galaxy(method, dims)
    assert payload is not None
    return payload

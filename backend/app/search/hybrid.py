"""Hybrid multimodal search: semantic (FAISS) + OCR keywords + colour palette.

Fusion (PRD §4, milestone-2)::

    S = (w_sem·sem + w_ocr·ocr + w_col·col) / (w_sem + w_ocr + w_col)

summed over the modes the query actually uses, every component in [0, 1].

* **semantic** — the query vector is the L2-normalised sum of every available
  text / image / upload / sketch vector. Raw cosine similarities are
  min-max normalised over the candidate set: CLIP cosines live in a narrow,
  model-dependent band (≈0.15–0.35 for text→image), so raw values would be
  drowned out by the OCR and colour components. Consequently the best
  semantic candidate always scores 1.0.
* **ocr** — BM25 relative to the query's best match (see :mod:`.ocr`).
* **color** — Lab palette match (see :mod:`.colors`). Stroke colours of a
  sketch are added as a weak colour hint.

Candidates are the union of the FAISS top ``k·5``, every OCR match and the
top ``k·5`` colour matches; missing semantic scores of non-FAISS candidates
are filled in with an exact restricted FAISS search. Only COMPLETED images
are returned.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

import numpy as np
from PIL import Image

from ..db.schema import get_db
from ..db.vectors import get_vector
from ..embeddings import get_encoder
from ..embeddings.base import l2_normalize
from .colors import QuerySwatch, hex_to_rgb, load_palettes, score_palettes
from .faiss_index import VectorIndex, get_index
from .ocr import make_snippet, search_ocr
from .schemas import ProjectionRef, SearchRequest
from .sketch import prepare_sketch, to_rgb_on_white

DEFAULT_WEIGHTS = {"semantic": 1.0, "ocr": 0.6, "color": 0.5}
CANDIDATE_FACTOR = 5
# Sketch stroke colours are only a hint: their weight relative to explicit swatches,
# and the colour-mode weight factor when the sketch is the only colour source.
SKETCH_COLOR_FACTOR = 0.3
_SWATCH_FIELDS = ("rank", "hex_code", "r", "g", "b", "weight")


class IndexMismatchError(RuntimeError):
    """The FAISS index was built with a different encoder than the active one."""


def hybrid_search(
    req: SearchRequest,
    upload: Image.Image | None = None,
    exclude_ids: Iterable[int] = (),
) -> dict:
    """Run a hybrid query and return the ``/search`` response payload.

    Args:
        req: the query (PRD §6.4).
        upload: an uploaded query image, embedded with the image tower.
        exclude_ids: image ids never returned (e.g. the query image itself).

    Raises:
        ValueError: empty or malformed query (HTTP 400).
        LookupError: ``image_id`` unknown or without an embedding (HTTP 404).
        IndexMismatchError: encoder and index disagree (HTTP 409).
    """
    t0 = time.perf_counter()
    exclude = {int(i) for i in exclude_ids}
    modes, vectors, swatches, sketch_only_colors = _parse_query(req, upload)
    ocr_text = (req.ocr or "").strip()
    projection = req.projection or ProjectionRef()
    n_cand = req.k * CANDIDATE_FACTOR + len(exclude)
    allowed = _allowed_ids(req, projection)

    # ── semantic ──────────────────────────────────────────────────────
    sem_raw: dict[int, float] = {}
    faiss_ms = 0.0
    index: VectorIndex | None = None
    query_vec: np.ndarray | None = None
    if vectors:
        index = get_index()
        if index.model and any(m in modes for m in ("text", "upload", "sketch")) and index.model != get_encoder().name:
            raise IndexMismatchError(
                f"index holds '{index.model}' vectors but the active encoder is '{get_encoder().name}'; re-index"
            )
        query_vec = l2_normalize(np.sum(vectors, axis=0))[0]
        ids, scores, faiss_ms = index.search(query_vec, n_cand, allowed)
        sem_raw = dict(zip(ids[0].tolist(), scores[0].tolist()))

    # ── OCR ───────────────────────────────────────────────────────────
    ocr_scores: dict[int, float] = {}
    if "ocr" in modes:
        ocr_scores = search_ocr(ocr_text)
        if allowed is not None:
            ocr_scores = {i: s for i, s in ocr_scores.items() if i in allowed}

    # ── colour ────────────────────────────────────────────────────────
    color_scores: dict[int, float] = {}
    color_top: list[int] = []
    if swatches:
        palettes = load_palettes(sorted(allowed) if allowed is not None else None)
        col = score_palettes(swatches, palettes)
        color_scores = dict(zip(palettes.ids.tolist(), col.tolist()))
        color_top = [int(palettes.ids[i]) for i in np.argsort(-col)[:n_cand] if col[i] > 0]

    candidates = (set(sem_raw) | set(ocr_scores) | set(color_top)) - exclude
    if modes == ["browse"]:
        candidates = (allowed or set()) - exclude

    # Fill in semantic scores for candidates found only by OCR / colour.
    if index is not None and query_vec is not None:
        missing = [i for i in candidates if i not in sem_raw]
        if missing:
            ids, scores, ms = index.search(query_vec, len(missing), missing)
            faiss_ms += ms
            sem_raw.update(zip(ids[0].tolist(), scores[0].tolist()))

    # ── fusion ────────────────────────────────────────────────────────
    weights = _effective_weights(req, bool(vectors), "ocr" in modes, bool(swatches), sketch_only_colors)
    sem_norm = _minmax({i: sem_raw[i] for i in candidates if i in sem_raw})
    fused: list[tuple[float, float, int, dict[str, float]]] = []
    for iid in candidates:
        parts: dict[str, float] = {}
        if "semantic" in weights:
            parts["semantic"] = sem_norm.get(iid, 0.0)
        if "ocr" in weights:
            parts["ocr"] = ocr_scores.get(iid, 0.0)
        if "color" in weights:
            parts["color"] = color_scores.get(iid, 0.0)
        score = sum(weights[m] * s for m, s in parts.items()) if parts else 1.0
        fused.append((score, sem_raw.get(iid, 0.0), iid, parts))
    fused.sort(key=lambda t: (-t[0], -t[1], t[2]))

    results, total = _hydrate(fused, req.k, projection, ocr_text or None)
    return {
        "results": results,
        "took_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        "faiss_ms": round(faiss_ms, 4),
        "modes": modes,
        "total_candidates": total,
    }


# ── query parsing ─────────────────────────────────────────────────────


def _parse_query(
    req: SearchRequest, upload: Image.Image | None
) -> tuple[list[str], list[np.ndarray], list[QuerySwatch], bool]:
    """Return ``(modes, query vectors, colour swatches, colours-come-only-from-sketch)``."""
    modes: list[str] = []
    vectors: list[np.ndarray] = []
    text = (req.text or "").strip()
    if text:
        modes.append("text")
        vectors.append(get_encoder().encode_texts([text])[0])
    if req.image_id is not None:
        vec = _image_vector(req.image_id)
        modes.append("image")
        vectors.append(vec)
    if upload is not None:
        modes.append("upload")
        vectors.append(get_encoder().encode_images([to_rgb_on_white(upload)])[0])
    sketch_colors: list[QuerySwatch] = []
    if req.sketch_png:
        sketch = prepare_sketch(req.sketch_png)
        modes.append("sketch")
        vectors.append(sketch.vector)
        sketch_colors = sketch.colors
    swatches = [QuerySwatch(hex_to_rgb(h)) for h in (req.colors or [])]
    if swatches:
        modes.append("color")
        sketch_colors = [QuerySwatch(s.rgb, s.weight * SKETCH_COLOR_FACTOR) for s in sketch_colors]
    if (req.ocr or "").strip():
        modes.append("ocr")
    if not modes and (req.cluster_ids or req.folder_id is not None):
        modes.append("browse")  # filter-only query: list the cluster / folder
    if not modes:
        raise ValueError("empty query: provide text, image_id, sketch_png, colors, ocr or an uploaded file")
    return modes, vectors, swatches + sketch_colors, bool(sketch_colors) and not swatches


def _image_vector(image_id: int) -> np.ndarray:
    with get_db() as conn:
        row = conn.execute("SELECT 1 FROM images WHERE image_id = ?", (image_id,)).fetchone()
    if row is None:
        raise LookupError(f"image {image_id} not found")
    vec = get_vector(image_id)
    if vec is None:
        raise LookupError(f"image {image_id} has no embedding yet")
    return vec


def _effective_weights(
    req: SearchRequest, semantic: bool, ocr: bool, color: bool, sketch_only_colors: bool
) -> dict[str, float]:
    """Weights of the active modes, renormalised to sum to 1."""
    w = dict(DEFAULT_WEIGHTS)
    if req.weights is not None:
        w.update({m: v for m, v in req.weights.model_dump().items() if v is not None})
    if sketch_only_colors:
        w["color"] *= SKETCH_COLOR_FACTOR
    active = {m: w[m] for m, on in (("semantic", semantic), ("ocr", ocr), ("color", color)) if on}
    total = sum(active.values())
    if total <= 0:
        return {m: 1.0 / len(active) for m in active}
    return {m: v / total for m, v in active.items()}


def _minmax(values: dict[int, float]) -> dict[int, float]:
    """Min-max normalise to [0, 1]; a degenerate range keeps clipped raw values."""
    if not values:
        return {}
    arr = np.fromiter(values.values(), dtype=np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        return {i: float(np.clip(v, 0.0, 1.0)) for i, v in values.items()}
    return {i: (v - lo) / (hi - lo) for i, v in values.items()}


# ── filters & hydration ───────────────────────────────────────────────


def _allowed_ids(req: SearchRequest, projection: ProjectionRef) -> set[int] | None:
    """Ids passing ``folder_id`` / ``cluster_ids`` filters, or ``None`` if unfiltered.

    The cluster filter is ignored when the requested projection has not been
    computed yet (``image_projections`` has no rows for it).
    """
    where = ["i.status = 'COMPLETED'"]
    params: list[object] = []
    if req.folder_id is not None:
        where.append("i.folder_id = ?")
        params.append(req.folder_id)
    with get_db() as conn:
        if req.cluster_ids:
            computed = conn.execute(
                "SELECT 1 FROM image_projections WHERE method = ? AND dims = ? LIMIT 1",
                (projection.method, projection.dims),
            ).fetchone()
            if computed:
                where.append(
                    f"""i.image_id IN (SELECT image_id FROM image_projections
                        WHERE method = ? AND dims = ? AND cluster_id IN ({','.join('?' * len(req.cluster_ids))}))"""
                )
                params.extend([projection.method, projection.dims, *req.cluster_ids])
        if len(where) == 1:
            return None
        rows = conn.execute(f"SELECT i.image_id FROM images i WHERE {' AND '.join(where)}", params).fetchall()
    return {int(r["image_id"]) for r in rows}


def _hydrate(
    fused: list[tuple[float, float, int, dict[str, float]]],
    k: int,
    projection: ProjectionRef,
    ocr_query: str | None,
) -> tuple[list[dict], int]:
    """Attach metadata to the top-``k`` COMPLETED hits; returns ``(hits, n_candidates)``."""
    if not fused:
        return [], 0
    ids = [iid for _, _, iid, _ in fused]
    marks = ",".join("?" * len(ids))
    with get_db() as conn:
        meta = {
            r["image_id"]: r
            for r in conn.execute(
                f"""SELECT i.image_id, i.file_path, i.width, i.height, o.raw_text, p.cluster_id
                    FROM images i
                    LEFT JOIN image_ocr o ON o.image_id = i.image_id
                    LEFT JOIN image_projections p
                           ON p.image_id = i.image_id AND p.method = ? AND p.dims = ?
                    WHERE i.status = 'COMPLETED' AND i.image_id IN ({marks})""",
                (projection.method, projection.dims, *ids),
            )
        }
        top = [t for t in fused if t[2] in meta][:k]
        top_ids = [t[2] for t in top]
        colors: dict[int, list[dict]] = {i: [] for i in top_ids}
        if top_ids:
            for c in conn.execute(
                f"""SELECT image_id, rank, hex_code, r, g, b, weight FROM image_colors
                    WHERE image_id IN ({','.join('?' * len(top_ids))}) ORDER BY image_id, rank""",
                top_ids,
            ):
                colors[c["image_id"]].append({f: c[f] for f in _SWATCH_FIELDS})
    hits = []
    for score, _, iid, parts in top:
        m = meta[iid]
        hits.append(
            {
                "image_id": iid,
                "score": round(float(np.clip(score, 0.0, 1.0)), 6),
                "scores": {name: round(float(np.clip(v, 0.0, 1.0)), 6) for name, v in parts.items()},
                "file_path": m["file_path"],
                "thumbnail_url": f"/api/v1/thumbnails/{iid}",
                "width": m["width"],
                "height": m["height"],
                "colors": colors[iid],
                "ocr_snippet": make_snippet(m["raw_text"], ocr_query),
                "cluster_id": m["cluster_id"],
            }
        )
    return hits, len(meta)

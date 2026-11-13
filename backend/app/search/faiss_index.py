"""FAISS vector index: an id-mapped, rebuildable cache over ``image_embeddings``.

SQLite stays the single source of truth; the FAISS index is derived from it
and persisted next to the database (``Settings.FAISS_INDEX_PATH`` plus a JSON
side-car describing what it was built from). A process-wide singleton is
kept fresh lazily: every :func:`get_index` call compares a cheap signature of
the vector table (model, vector count, last update) with the one the cached
index was built from and rebuilds when they differ. The indexer's post-index
hook triggers an eager rebuild after each job.

Index type: exact ``IndexFlatIP`` up to :data:`HNSW_THRESHOLD` vectors,
``IndexHNSWFlat`` above. Vectors are L2-normalised, so inner product equals
cosine similarity.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass

import faiss
import numpy as np

from ..config import get_settings
from ..db.schema import get_db
from ..db.vectors import current_models, load_vectors
from ..indexer.hooks import register_post_index_hook

logger = logging.getLogger(__name__)

HNSW_THRESHOLD = 50_000
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 80
HNSW_EF_SEARCH = 64
# Above this many allowed ids an HNSW filtered search uses an ID selector;
# below it the allowed vectors are scored exactly (selectors hurt HNSW recall).
_EXACT_SUBSET_MAX = 4096


@dataclass(frozen=True)
class IndexSignature:
    """What an index was built from; equal signatures mean the index is fresh."""

    model: str
    n_vectors: int
    last_update: str


class VectorIndex:
    """Immutable id-mapped FAISS index over L2-normalised vectors.

    Build directly from arrays (``VectorIndex(ids, mat)``) or from the
    database with :meth:`from_db`. Searching is thread-safe; instances are
    never mutated after construction (a rebuild creates a new instance).
    """

    def __init__(
        self,
        ids: np.ndarray,
        mat: np.ndarray,
        model: str = "",
        signature: IndexSignature | None = None,
        index: faiss.Index | None = None,
    ) -> None:
        ids = np.ascontiguousarray(ids, dtype=np.int64)  # ascending image_id order
        mat = np.ascontiguousarray(mat, dtype=np.float32)
        self.model = model
        self.signature = signature
        self.ids = ids
        self.dim = int(mat.shape[1]) if mat.ndim == 2 and mat.shape[0] else 0
        if index is None:
            t0 = time.perf_counter()
            index = self._build(ids, mat, self.dim)
            self.build_ms = (time.perf_counter() - t0) * 1000.0
        else:
            self.dim = index.d
            self.build_ms = 0.0
        self._index = index

    # ── construction ──────────────────────────────────────────────────
    @staticmethod
    def _build(ids: np.ndarray, mat: np.ndarray, dim: int) -> faiss.Index | None:
        if len(ids) == 0:
            return None
        if len(ids) > HNSW_THRESHOLD:
            inner = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
            inner.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
            inner.hnsw.efSearch = HNSW_EF_SEARCH
        else:
            inner = faiss.IndexFlatIP(dim)
        index = faiss.IndexIDMap2(inner)
        index.add_with_ids(mat, ids)
        return index

    @classmethod
    def from_db(cls, signature: IndexSignature) -> VectorIndex:
        """Build from every COMPLETED vector of ``signature.model``."""
        if signature.model:
            ids, mat = load_vectors(model=signature.model)
        else:
            ids, mat = np.empty(0, np.int64), np.empty((0, 0), np.float32)
        return cls(ids, mat, model=signature.model, signature=signature)

    # ── properties ────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.ids)

    @property
    def is_hnsw(self) -> bool:
        return len(self) > HNSW_THRESHOLD

    @property
    def index_type(self) -> str:
        if self._index is None:
            return "empty"
        return "IndexIDMap2(IndexHNSWFlat)" if self.is_hnsw else "IndexIDMap2(IndexFlatIP)"

    # ── search ────────────────────────────────────────────────────────
    def search(
        self,
        query_vecs: np.ndarray,
        k: int,
        allowed_ids: np.ndarray | list[int] | set[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Top-``k`` inner-product search.

        Args:
            query_vecs: ``(nq, dim)`` or ``(dim,)`` float32 L2-normalised query.
            k: number of neighbours per query.
            allowed_ids: optional candidate set; only these ids are returned.

        Returns:
            ``(ids (nq, k') int64, scores (nq, k') float32, faiss_ms)`` where
            ``k' <= k`` and padding entries (``-1`` ids) are removed when ``nq == 1``.
        """
        q = np.ascontiguousarray(np.atleast_2d(query_vecs), dtype=np.float32)
        empty = (np.empty((q.shape[0], 0), np.int64), np.empty((q.shape[0], 0), np.float32), 0.0)
        if self._index is None or k <= 0:
            return empty
        if q.shape[1] != self.dim:
            raise ValueError(f"query dim {q.shape[1]} != index dim {self.dim}")

        allowed: np.ndarray | None = None
        if allowed_ids is not None:
            allowed = np.fromiter((int(i) for i in allowed_ids), dtype=np.int64)
            allowed = allowed[np.isin(allowed, self.ids)]
            if allowed.size == 0:
                return empty
        k_eff = min(k, len(self) if allowed is None else int(allowed.size))

        t0 = time.perf_counter()
        if allowed is not None and self.is_hnsw and allowed.size <= _EXACT_SUBSET_MAX:
            ids, scores = self._exact_subset(q, k_eff, allowed)
        else:
            params = None
            if allowed is not None:
                sel = faiss.IDSelectorBatch(allowed)
                params = (
                    faiss.SearchParametersHNSW(sel=sel, efSearch=max(HNSW_EF_SEARCH, 4 * k_eff))
                    if self.is_hnsw
                    else faiss.SearchParameters(sel=sel)
                )
            elif self.is_hnsw:
                params = faiss.SearchParametersHNSW(efSearch=max(HNSW_EF_SEARCH, 2 * k_eff))
            scores, ids = self._index.search(q, k_eff, params=params)
        faiss_ms = (time.perf_counter() - t0) * 1000.0

        if q.shape[0] == 1:
            keep = ids[0] >= 0
            ids, scores = ids[:, keep], scores[:, keep]
        return ids, scores, faiss_ms

    def _exact_subset(self, q: np.ndarray, k: int, allowed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vecs = self._index.reconstruct_batch(allowed)
        sims = q @ vecs.T
        order = np.argsort(-sims, axis=1)[:, :k]
        return allowed[order], np.take_along_axis(sims, order, axis=1).astype(np.float32)

    # ── persistence ───────────────────────────────────────────────────
    def save(self) -> None:
        """Write the index and its signature side-car (best effort)."""
        if self._index is None or self.signature is None:
            return
        path = get_settings().FAISS_INDEX_PATH
        try:
            faiss.write_index(self._index, str(path))
            _meta_path().write_text(json.dumps(asdict(self.signature)))
        except Exception:
            logger.exception("Could not persist FAISS index to %s", path)

    @classmethod
    def load(cls, signature: IndexSignature) -> VectorIndex | None:
        """Load the persisted index if it was built from ``signature``."""
        path, meta = get_settings().FAISS_INDEX_PATH, _meta_path()
        if not (path.is_file() and meta.is_file()):
            return None
        try:
            if IndexSignature(**json.loads(meta.read_text())) != signature:
                return None
            index = faiss.read_index(str(path))
            ids = faiss.vector_to_array(faiss.downcast_index(index).id_map).astype(np.int64)
            if len(ids) != signature.n_vectors:
                return None
            order = np.argsort(ids)
            return cls(ids[order], np.empty((0, 0), np.float32), signature.model, signature, index=index)
        except Exception:
            logger.exception("Ignoring unreadable FAISS index at %s", path)
            return None


# ── signature ─────────────────────────────────────────────────────────


def _meta_path():
    return get_settings().FAISS_INDEX_PATH.with_suffix(".json")


def current_signature() -> IndexSignature:
    """Signature of the vectors the index should be built from right now.

    The model is the most common one in ``image_embeddings``.
    """
    models = current_models()
    model = models[0][0] if models else ""
    with get_db() as conn:
        row = conn.execute(
            """SELECT COUNT(*) AS n, COALESCE(MAX(e.updated_at), '') AS last
               FROM image_embeddings e JOIN images i ON i.image_id = e.image_id
               WHERE i.status = 'COMPLETED' AND e.model = ?""",
            (model,),
        ).fetchone()
    return IndexSignature(model=model, n_vectors=int(row["n"]), last_update=str(row["last"]))


# ── singleton ─────────────────────────────────────────────────────────

_index: VectorIndex | None = None
_lock = threading.Lock()


def get_index() -> VectorIndex:
    """Return the process-wide index, loading or rebuilding it when stale."""
    global _index
    sig = current_signature()
    current = _index
    if current is not None and current.signature == sig:
        return current
    with _lock:
        if _index is not None and _index.signature == sig:
            return _index
        loaded = VectorIndex.load(sig)
        if loaded is not None:
            _index = loaded
        else:
            _index = VectorIndex.from_db(sig)
            _index.save()
            logger.info("Built %s with %d vectors in %.1f ms", _index.index_type, len(_index), _index.build_ms)
        return _index


def rebuild() -> VectorIndex:
    """Force a rebuild from the database and persist it."""
    global _index
    with _lock:
        _index = VectorIndex.from_db(current_signature())
        _index.save()
        logger.info("Rebuilt %s with %d vectors in %.1f ms", _index.index_type, len(_index), _index.build_ms)
        return _index


def reset_index() -> None:
    """Drop the in-memory index (tests switch data dirs between runs)."""
    global _index
    with _lock:
        _index = None


register_post_index_hook("faiss_rebuild", rebuild)

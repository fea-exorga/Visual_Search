"""Dimensionality reduction of L2-normalised embeddings to 2D / 3D.

``project`` is a pure function: (N, D) float matrix in, (N, dims) float32
coordinates out, every axis centred and scaled into ``[-1, 1]``. All methods
are deterministic (``random_state=42``).

* ``pca``  — plain PCA to ``dims`` components.
* ``umap`` — PCA pre-reduction to ``min(50, N-1, D)`` then UMAP (cosine).
* ``tsne`` — PCA pre-reduction then Barnes-Hut t-SNE with PCA init and a
  perplexity clamped below ``(N-1)/3``.

Tiny inputs (N < 4) cannot be embedded meaningfully by UMAP / t-SNE, so they
fall back to PCA (N >= 2) or the origin (N == 1).
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

METHODS: tuple[str, ...] = ("umap", "tsne", "pca")
RANDOM_STATE = 42
PCA_PRE_DIMS = 50
MIN_MANIFOLD_POINTS = 4  # below this UMAP / t-SNE fall back to PCA


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    """Centre each axis on its mean and divide by the largest absolute value.

    The result lies in ``[-1, 1]`` per axis; constant axes become all zeros.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.size == 0:
        return coords.astype(np.float32)
    coords = coords - coords.mean(axis=0, keepdims=True)
    scale = np.abs(coords).max(axis=0, keepdims=True)
    scale[scale < 1e-12] = 1.0
    out = np.clip(coords / scale, -1.0, 1.0)
    out[np.abs(out) < 1e-12] = 0.0
    return out.astype(np.float32)


def _pca(mat: np.ndarray, n_components: int) -> np.ndarray:
    """PCA to ``n_components``, zero-padding when the data has fewer dims."""
    from sklearn.decomposition import PCA

    n, d = mat.shape
    k = max(0, min(n_components, n - 1, d)) if n > 1 else 0
    out = np.zeros((n, n_components), dtype=np.float64)
    if k > 0:
        out[:, :k] = PCA(n_components=k, random_state=RANDOM_STATE).fit_transform(mat)
    return out


def _pre_reduce(mat: np.ndarray) -> np.ndarray:
    """PCA pre-reduction to ``min(50, N-1, D)`` dims (skipped when already small)."""
    n, d = mat.shape
    k = min(PCA_PRE_DIMS, n - 1, d)
    if k >= d:
        return mat
    return _pca(mat, k)


def _umap(x: np.ndarray, dims: int, n_neighbors: int = 15, min_dist: float = 0.1,
          n_epochs: int | None = None, **_: Any) -> np.ndarray:
    import umap  # heavy import (numba) — keep lazy

    n = x.shape[0]
    nn = int(max(2, min(int(n_neighbors), n - 1)))
    # Spectral init needs a reasonably sized graph; random init is robust for tiny N.
    init = "spectral" if n > dims + 10 else "random"
    reducer = umap.UMAP(
        n_components=dims,
        n_neighbors=nn,
        min_dist=float(min_dist),
        metric="cosine",
        init=init,
        n_epochs=n_epochs,
        random_state=RANDOM_STATE,
        n_jobs=1,  # random_state forces single-threaded anyway; silences the warning
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(reducer.fit_transform(x), dtype=np.float64)


def _tsne(x: np.ndarray, dims: int, perplexity: float = 30.0,
          max_iter: int | None = None, **_: Any) -> np.ndarray:
    from sklearn.manifold import TSNE

    n = x.shape[0]
    # sklearn requires perplexity < N; the usual guidance is N > 3 * perplexity.
    upper = (n - 1) / 3.0
    perp = float(min(float(perplexity), upper - 1e-3))
    perp = max(perp, min(1.0, upper - 1e-3))
    kwargs: dict[str, Any] = dict(
        n_components=dims,
        perplexity=perp,
        init="pca",
        learning_rate="auto",
        random_state=RANDOM_STATE,
        method="barnes_hut" if dims <= 3 else "exact",
    )
    if max_iter is not None:
        kwargs["max_iter"] = int(max(250, max_iter))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(TSNE(**kwargs).fit_transform(x), dtype=np.float64)


def project(mat: np.ndarray, method: str = "umap", dims: int = 2, **params: Any) -> np.ndarray:
    """Project ``mat`` (N, D) to ``(N, dims)`` coordinates normalised to [-1, 1].

    Args:
        mat: embedding matrix (ideally L2-normalised).
        method: ``"umap"`` | ``"tsne"`` | ``"pca"``.
        dims: 2 or 3.
        **params: method-specific options — UMAP ``n_neighbors``, ``min_dist``,
            ``n_epochs``; t-SNE ``perplexity``, ``max_iter``. ``None`` values
            are ignored so API bodies can be forwarded verbatim.

    Raises:
        ValueError: unknown method / dims or a non-2-D matrix.
    """
    if method not in METHODS:
        raise ValueError(f"unknown projection method {method!r}; expected one of {METHODS}")
    if dims not in (2, 3):
        raise ValueError(f"dims must be 2 or 3, got {dims!r}")
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError("mat must be a 2-D (N, D) array")
    n = mat.shape[0]
    if n == 0:
        return np.zeros((0, dims), dtype=np.float32)
    if n == 1:
        return np.zeros((1, dims), dtype=np.float32)

    params = {k: v for k, v in params.items() if v is not None}
    if method == "pca" or n < MIN_MANIFOLD_POINTS:
        coords = _pca(mat.astype(np.float64), dims)
    else:
        x = _pre_reduce(mat.astype(np.float64))
        coords = _umap(x, dims, **params) if method == "umap" else _tsne(x, dims, **params)
    if not np.all(np.isfinite(coords)):
        logger.warning("%s produced non-finite coordinates; falling back to PCA", method)
        coords = _pca(mat.astype(np.float64), dims)
    return normalize_coords(coords)

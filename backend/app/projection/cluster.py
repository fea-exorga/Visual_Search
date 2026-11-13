"""KMeans theme clustering of projected galaxy coordinates.

Design choice: we cluster the **projected 2D/3D coordinates** rather than the
PCA-50 embeddings. UMAP / t-SNE already concentrate the semantic
neighbourhood structure into visible islands, so clusters computed on the
coordinates are exactly the "vibes" a user sees on the map (contiguous,
non-overlapping colour regions). Clustering in PCA-50 space often produces
clusters that are interleaved on the 2D map, which reads as noise in the
galaxy view. It is also much cheaper (silhouette on 2-3 dims).

``k`` is chosen by silhouette score over ``[min(2, N-1) .. min(24, N // 3)]``
unless given. The lower bound is 2 (not 4 as in the PRD sketch) so a library
that genuinely has 2-3 themes is not over-split. Silhouette is computed on a
random sample of at most 3000 points for speed.

Cluster ids are re-numbered by descending size (0 = largest) and coloured
from a fixed, perceptually distinct palette, so colours are stable across
recomputes of the same library.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RANDOM_STATE = 42
MAX_K = 24
SILHOUETTE_SAMPLE = 3000

# 24 perceptually distinct colours (Glasbey-style, tuned for a dark canvas).
PALETTE: tuple[str, ...] = (
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948",
    "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC", "#17BECF", "#BCBD22",
    "#E377C2", "#8C6D31", "#7F3C8D", "#11A579", "#3969AC", "#F2B701",
    "#E73F74", "#80BA5A", "#E68310", "#008695", "#CF1C90", "#A5AA99",
)


def color_for(cluster_id: int) -> str:
    """Stable colour for a cluster id."""
    return PALETTE[int(cluster_id) % len(PALETTE)]


@dataclass
class ClusterResult:
    """Output of :func:`cluster_points`.

    Attributes:
        labels: (N,) int cluster id per point, ids ordered by descending size.
        centroids: (k, dims) mean coordinate of each cluster.
        sizes: (k,) number of points per cluster.
        colors: hex colour per cluster.
        silhouette: silhouette score of the chosen k (``None`` if k < 2).
    """

    labels: np.ndarray
    centroids: np.ndarray
    sizes: np.ndarray
    colors: list[str]
    silhouette: float | None

    @property
    def k(self) -> int:
        return int(len(self.sizes))


def k_range(n: int) -> range:
    """Candidate cluster counts for ``n`` points (may be a single value)."""
    lo = min(2, n - 1)
    hi = min(MAX_K, n // 3)
    hi = max(hi, lo)
    return range(max(lo, 1), max(hi, 1) + 1)


def _kmeans(x: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, n_init=4, random_state=RANDOM_STATE)
    labels = km.fit_predict(x)
    return labels.astype(np.int64), float(km.inertia_)


def _silhouette(x: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> float | None:
    """Silhouette score (sampled for large N); ``None`` when undefined for this labelling."""
    from sklearn.metrics import silhouette_score

    if not 2 <= len(np.unique(labels)) < x.shape[0]:
        return None
    if x.shape[0] > SILHOUETTE_SAMPLE:
        idx = rng.choice(x.shape[0], SILHOUETTE_SAMPLE, replace=False)
        x, labels = x[idx], labels[idx]
        if not 2 <= len(np.unique(labels)) < x.shape[0]:
            return None
    return float(silhouette_score(x, labels))


def _relabel_by_size(labels: np.ndarray) -> np.ndarray:
    """Renumber cluster ids 0..k-1 by descending size (ties: lowest old id)."""
    uniq, counts = np.unique(labels, return_counts=True)
    order = sorted(range(len(uniq)), key=lambda i: (-counts[i], uniq[i]))
    mapping = {int(uniq[old]): new for new, old in enumerate(order)}
    return np.array([mapping[int(lab)] for lab in labels], dtype=np.int64)


def cluster_points(coords: np.ndarray, n_clusters: int | None = None) -> ClusterResult:
    """Cluster ``coords`` (N, dims) with KMeans; every point gets a cluster.

    Args:
        coords: projected coordinates.
        n_clusters: fixed k; ``None`` selects k by silhouette score.
    """
    x = np.asarray(coords, dtype=np.float64)
    n = x.shape[0]
    dims = x.shape[1] if x.ndim == 2 else 0
    if n == 0:
        return ClusterResult(np.zeros(0, np.int64), np.zeros((0, dims)), np.zeros(0, np.int64), [], None)

    n_distinct = len(np.unique(np.round(x, 9), axis=0))
    best_labels: np.ndarray
    best_score: float | None = None
    if n_clusters is not None:
        k = int(max(1, min(int(n_clusters), n_distinct)))
        best_labels = _kmeans(x, k)[0] if k > 1 else np.zeros(n, np.int64)
        if k > 1:
            best_score = _silhouette(x, best_labels, np.random.default_rng(RANDOM_STATE))
    elif n < 3 or n_distinct < 2:
        best_labels = np.zeros(n, np.int64)
    else:
        rng = np.random.default_rng(RANDOM_STATE)
        best_labels = np.zeros(n, np.int64)
        best_score = -2.0
        for k in k_range(n):
            if k < 2 or k > n_distinct:
                continue
            labels, _ = _kmeans(x, k)
            score = _silhouette(x, labels, rng)
            if score is not None and score > best_score + 1e-9:
                best_score, best_labels = score, labels
        if best_score <= -2.0:
            best_score = None

    labels = _relabel_by_size(best_labels)
    k = int(labels.max()) + 1
    sizes = np.bincount(labels, minlength=k).astype(np.int64)
    centroids = np.stack([x[labels == c].mean(axis=0) for c in range(k)])
    return ClusterResult(
        labels=labels,
        centroids=centroids,
        sizes=sizes,
        colors=[color_for(c) for c in range(k)],
        silhouette=best_score,
    )

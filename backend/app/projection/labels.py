"""Zero-shot cluster names.

Each cluster's mean image embedding is compared (cosine) with text
embeddings of a fixed "vibe" vocabulary; prompts are encoded once per
encoder with a few CLIP-style templates and averaged. Labels are assigned
greedily so that no two clusters share a name when the vocabulary allows.
If the cluster has OCR text, its most frequent non-stopword keyword
(>= 4 chars) is appended: ``"lecture slide with text · gradient"``.
"""

from __future__ import annotations

import re
import threading
from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np

VIBE_VOCABULARY: tuple[str, ...] = (
    # documents & screens
    "lecture slide with text", "presentation slide", "code screenshot", "terminal window",
    "website screenshot", "mobile app screenshot", "document", "scanned page", "handwritten notes",
    "receipt", "spreadsheet", "chart or graph", "diagram", "map", "infographic", "poster",
    "meme with caption", "comic", "logo", "icon", "user interface",
    # art & style
    "hand-drawn sketch", "painting", "watercolor", "pixel art", "3d render", "anime illustration",
    "abstract art", "geometric shapes", "circle", "pattern", "gradient", "minimalist",
    "black and white photo", "dark moody", "pastel", "neon", "vintage film", "colorful",
    # nature & places
    "nature landscape", "mountains", "beach", "ocean", "forest", "sunset", "sky and clouds",
    "snow", "desert", "flowers", "city at night", "city skyline", "street", "architecture",
    "interior room", "road trip",
    # people & animals
    "portrait photo", "selfie", "group of people", "wedding", "concert", "sports",
    "baby", "cat", "dog", "bird", "wildlife",
    # things
    "food", "dessert", "drinks", "car", "airplane", "computer", "phone", "book",
    "clothing", "shoes", "toys", "plants",
    # colours
    "solid red", "solid blue", "solid green", "solid yellow", "solid purple", "solid black",
    "solid white", "solid orange", "solid pink", "solid gray",
)

TEMPLATES: tuple[str, ...] = ("a photo of {}", "{}", "an image of {}")

STOPWORDS: frozenset[str] = frozenset(
    """
    about above after again against also because been before being below between both
    cannot could does doing down during each from further have having here hers herself
    himself into itself just more most myself once only other ours ourselves over same
    should some such than that their theirs them themselves then there these they this
    those through under until very were what when where which while whom will with would
    your yours yourself yourselves http https www com page click image photo
    """.split()
)

_cache: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}
_cache_lock = threading.Lock()


def _normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def vocabulary_embeddings(encoder, vocabulary: Sequence[str] = VIBE_VOCABULARY) -> np.ndarray:
    """Template-averaged, L2-normalised text embeddings (V, dim), cached per encoder."""
    key = (str(getattr(encoder, "name", type(encoder).__name__)), tuple(vocabulary))
    with _cache_lock:
        hit = _cache.get(key)
    if hit is not None:
        return hit
    texts = [t.format(v) for v in vocabulary for t in TEMPLATES]
    emb = np.asarray(encoder.encode_texts(texts), dtype=np.float32)
    emb = _normalize(emb.reshape(len(vocabulary), len(TEMPLATES), -1).mean(axis=1))
    with _cache_lock:
        _cache[key] = emb
    return emb


def clear_cache() -> None:
    """Forget cached vocabulary embeddings (tests / encoder switch)."""
    with _cache_lock:
        _cache.clear()


def top_keyword(texts: Iterable[str | None]) -> str | None:
    """Most frequent non-stopword (>= 4 letters) across ``texts``; ties → alphabetical."""
    counts: Counter[str] = Counter()
    for t in texts:
        if not t:
            continue
        for w in re.findall(r"[^\W\d_]{4,}", t.lower()):
            if w not in STOPWORDS:
                counts[w] += 1
    if not counts:
        return None
    return min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def assign_labels(centroids: np.ndarray, vocab_emb: np.ndarray,
                  vocabulary: Sequence[str] = VIBE_VOCABULARY) -> list[str]:
    """Greedy unique assignment of vocabulary entries to cluster centroids.

    The highest-similarity (cluster, prompt) pairs are taken first, so every
    cluster gets its best still-unused prompt. Falls back to reuse when there
    are more clusters than prompts.
    """
    k = centroids.shape[0]
    if k == 0:
        return []
    sims = _normalize(centroids) @ vocab_emb.T  # (k, V)
    out: list[str | None] = [None] * k
    used: set[int] = set()
    for flat in np.argsort(-sims, axis=None, kind="stable"):
        c, v = divmod(int(flat), sims.shape[1])
        if out[c] is None and v not in used:
            out[c] = vocabulary[v]
            used.add(v)
            if all(o is not None for o in out):
                break
    return [o if o is not None else vocabulary[int(np.argmax(sims[i]))] for i, o in enumerate(out)]


def label_clusters(mat: np.ndarray, labels: np.ndarray, encoder,
                   ocr_texts: Sequence[str | None] | None = None,
                   vocabulary: Sequence[str] = VIBE_VOCABULARY) -> list[str]:
    """Name each cluster ``0..k-1``.

    Args:
        mat: (N, dim) image embeddings, row-aligned with ``labels``.
        labels: (N,) cluster id per image.
        encoder: object with ``encode_texts`` and ``name`` (``app.embeddings``).
        ocr_texts: optional OCR text per image (row-aligned).
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        return []
    k = int(labels.max()) + 1
    centroids = np.stack([np.asarray(mat[labels == c], dtype=np.float32).mean(axis=0) for c in range(k)])
    names = assign_labels(centroids, vocabulary_embeddings(encoder, vocabulary), vocabulary)
    if ocr_texts is not None:
        for c in range(k):
            kw = top_keyword(t for t, lab in zip(ocr_texts, labels) if lab == c)
            if kw:
                names[c] = f"{names[c]} · {kw}"
    return names

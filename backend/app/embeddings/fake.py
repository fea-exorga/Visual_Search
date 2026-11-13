"""Deterministic, dependency-free encoder for tests and offline smoke runs.

Image vectors are a fixed random projection of a colour histogram plus a
16x16 grayscale thumbnail, so visually similar images really are close.
Text vectors are hashed bag-of-words projections, and colour words ("red",
"blue", ...) are mapped onto the same colour-histogram features so
text-to-image search behaves sensibly in tests.
"""

from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image

from .base import l2_normalize

_COLOR_WORDS = {
    "red": (220, 30, 30), "green": (30, 180, 60), "blue": (30, 60, 220),
    "yellow": (240, 220, 40), "orange": (250, 140, 20), "purple": (140, 40, 180),
    "pink": (250, 150, 200), "black": (10, 10, 10), "white": (245, 245, 245),
    "gray": (128, 128, 128), "grey": (128, 128, 128), "cyan": (40, 220, 220),
}

_HIST_BINS = 4  # per channel → 64 colour bins
_THUMB = 16


class FakeEncoder:
    def __init__(self, dim: int = 512, seed: int = 1234) -> None:
        self.dim = dim
        self.name = f"fake-{dim}"
        rng = np.random.default_rng(seed)
        self._feat_dim = _HIST_BINS**3 + _THUMB * _THUMB
        self._proj = rng.standard_normal((self._feat_dim, dim)).astype(np.float32)
        self._proj /= np.sqrt(self._feat_dim)

    # ── image ──────────────────────────────────────────────────────────
    def _hist(self, rgb: np.ndarray) -> np.ndarray:
        q = (rgb // (256 // _HIST_BINS)).reshape(-1, 3)
        idx = q[:, 0] * _HIST_BINS * _HIST_BINS + q[:, 1] * _HIST_BINS + q[:, 2]
        h = np.bincount(idx, minlength=_HIST_BINS**3).astype(np.float32)
        return h / max(h.sum(), 1.0)

    def _features(self, img: Image.Image) -> np.ndarray:
        rgb = np.asarray(img.convert("RGB").resize((32, 32)), dtype=np.int64)
        gray = np.asarray(img.convert("L").resize((_THUMB, _THUMB)), dtype=np.float32) / 255.0
        gray = gray.flatten() - gray.mean()
        return np.concatenate([self._hist(rgb) * 4.0, gray * 0.5])

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, self.dim), dtype=np.float32)
        feats = np.stack([self._features(im) for im in images])
        return l2_normalize(feats @ self._proj)

    # ── text ───────────────────────────────────────────────────────────
    def _text_features(self, text: str) -> np.ndarray:
        feat = np.zeros(self._feat_dim, dtype=np.float32)
        for word in text.lower().split():
            word = word.strip(".,!?;:'\"")
            if word in _COLOR_WORDS:
                r, g, b = (c // (256 // _HIST_BINS) for c in _COLOR_WORDS[word])
                feat[r * _HIST_BINS * _HIST_BINS + g * _HIST_BINS + b] += 4.0
            else:
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                feat[_HIST_BINS**3 + h % (_THUMB * _THUMB)] += 0.5
        return feat

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        feats = np.stack([self._text_features(t) for t in texts])
        return l2_normalize(feats @ self._proj)

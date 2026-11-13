"""Common encoder interface shared by every backend (ONNX, torch, fake)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation returning float32."""
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat[None, :]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


@runtime_checkable
class Encoder(Protocol):
    """A contrastive image/text encoder mapping both modalities into one space.

    Every method returns an L2-normalised ``float32`` matrix of shape ``(N, dim)``
    so cosine similarity == inner product (what FAISS ``IndexFlatIP`` expects).
    """

    name: str  # e.g. "clip-vit-b32/onnx-fp32" — stored next to every vector
    dim: int

    def encode_images(self, images: list[Image.Image]) -> np.ndarray: ...

    def encode_texts(self, texts: list[str]) -> np.ndarray: ...

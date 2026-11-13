"""Process-wide encoder singleton selected by ``Settings.EMBED_BACKEND``.

Backends:
  * ``onnx``  — CLIP ViT-B/32 on ONNX Runtime (default; fp32/fp16/int8)
  * ``torch`` — Hugging Face transformers CLIP/SigLIP (optional, needs torch)
  * ``fake``  — deterministic test encoder (no downloads)
"""

from __future__ import annotations

import logging
import threading

from ..config import get_settings
from .base import Encoder

logger = logging.getLogger(__name__)

_encoder: Encoder | None = None
_lock = threading.Lock()


def _build() -> Encoder:
    s = get_settings()
    backend = s.EMBED_BACKEND
    if backend == "fake":
        from .fake import FakeEncoder
        return FakeEncoder(dim=s.EMBEDDING_DIM)
    if backend == "torch":
        from .torch_hf import TorchHFEncoder
        return TorchHFEncoder(s.TORCH_MODEL_NAME)
    from .onnx_clip import OnnxClipEncoder
    return OnnxClipEncoder(precision=s.ONNX_PRECISION, repo_id=s.ONNX_REPO_ID)


def get_encoder() -> Encoder:
    global _encoder
    if _encoder is None:
        with _lock:
            if _encoder is None:
                _encoder = _build()
    return _encoder


def reset_encoder() -> None:
    """Drop the cached encoder (tests / precision switch)."""
    global _encoder
    with _lock:
        _encoder = None

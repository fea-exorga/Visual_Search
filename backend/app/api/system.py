"""System info: which encoder / OCR engine / precision is active."""

from __future__ import annotations

from fastapi import APIRouter

from ..config import get_settings
from ..db.vectors import current_models

router = APIRouter(tags=["system"])


@router.get("/system/info")
def system_info():
    s = get_settings()
    from ..indexer.ocr_engine import get_ocr_engine

    return {
        "embed_backend": s.EMBED_BACKEND,
        "onnx_repo": s.ONNX_REPO_ID,
        "onnx_precision": s.ONNX_PRECISION,
        "embedding_dim": s.EMBEDDING_DIM,
        "ocr_engine": get_ocr_engine().engine,
        "vector_models": [{"model": m, "count": n} for m, n in current_models()],
    }

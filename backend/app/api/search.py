"""Search API (Milestone 2): hybrid, upload, find-similar and index rebuild.

Importing this module registers the FAISS rebuild post-index hook (via
:mod:`app.search.faiss_index`).
"""

from __future__ import annotations

import io

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from ..search import faiss_index
from ..search.hybrid import IndexMismatchError, hybrid_search
from ..search.schemas import SearchRequest

router = APIRouter(tags=["search"])


def _run(req: SearchRequest, upload: Image.Image | None = None, exclude: tuple[int, ...] = ()) -> dict:
    """Run a search, translating domain errors into HTTP errors."""
    try:
        return hybrid_search(req, upload=upload, exclude_ids=exclude)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except IndexMismatchError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/search")
def search(body: SearchRequest) -> dict:
    """Hybrid search over any combination of text, image, sketch, colours and OCR."""
    return _run(body)


@router.post("/search/upload")
def search_upload(
    file: UploadFile = File(...),
    k: int = Form(60, ge=1, le=1000),
    text: str | None = Form(None),
) -> dict:
    """Image-to-image search with an uploaded file (optionally combined with text)."""
    try:
        img = Image.open(io.BytesIO(file.file.read()))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="uploaded file is not a readable image") from exc
    return _run(SearchRequest(text=text, k=k), upload=img)


@router.get("/search/similar/{image_id}")
def search_similar(image_id: int, k: int = Query(24, ge=1, le=1000)) -> dict:
    """Images most similar to an indexed image (the image itself is excluded)."""
    return _run(SearchRequest(image_id=image_id, k=k), exclude=(image_id,))


@router.post("/search/rebuild")
def rebuild_index() -> dict:
    """Rebuild the FAISS index from the embeddings table."""
    index = faiss_index.rebuild()
    return {"n_vectors": len(index), "index_type": index.index_type, "build_ms": round(index.build_ms, 3)}

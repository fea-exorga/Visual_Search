"""Request models for the ``/search`` endpoints (PRD §6.4)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SearchWeights(BaseModel):
    """Per-mode fusion weights; ``None`` means the default."""

    semantic: float | None = Field(None, ge=0)
    ocr: float | None = Field(None, ge=0)
    color: float | None = Field(None, ge=0)


class ProjectionRef(BaseModel):
    """Which projection's clusters ``cluster_ids`` refer to."""

    method: Literal["umap", "tsne", "pca"] = "umap"
    dims: Literal[2, 3] = 2


class SearchRequest(BaseModel):
    """Hybrid query: any combination of the modes below (at least one)."""

    text: str | None = None
    image_id: int | None = None
    sketch_png: str | None = None
    colors: list[str] | None = None
    ocr: str | None = None
    cluster_ids: list[int] | None = None
    folder_id: int | None = None
    projection: ProjectionRef | None = None
    k: int = Field(60, ge=1, le=1000)
    weights: SearchWeights | None = None

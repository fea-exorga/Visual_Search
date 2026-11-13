"""Galaxy explorer API (Milestone 3) — see docs/PRD.md §6.4.

``GET  /galaxy?method=umap&dims=2``  stored projection + clusters (404 if none)
``POST /galaxy/compute``             (re)compute synchronously and return it
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from ..projection.service import GalaxyError, compute_galaxy, get_galaxy

router = APIRouter(tags=["galaxy"])


class GalaxyComputeRequest(BaseModel):
    """Body of ``POST /galaxy/compute``; method/dims are validated by the service (400)."""

    method: str = "umap"
    dims: int = 2
    n_neighbors: int | None = Field(None, ge=2, le=200)
    min_dist: float | None = Field(None, ge=0.0, le=1.0)
    perplexity: float | None = Field(None, gt=0.0, le=200.0)
    n_clusters: int | None = Field(None, ge=1, le=64)


def _raise(e: GalaxyError) -> None:
    raise HTTPException(status_code=e.status, detail=str(e)) from e


@router.get("/galaxy")
async def read_galaxy(
    method: str = Query("umap", description="umap | tsne | pca"),
    dims: int = Query(2, description="2 | 3"),
) -> dict[str, Any]:
    """Return the persisted galaxy for ``(method, dims)``; 400 if invalid, 404 until computed."""
    try:
        payload = await run_in_threadpool(get_galaxy, method, dims)
    except GalaxyError as e:
        _raise(e)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"galaxy {method}/{dims}D not computed yet")
    return payload


@router.post("/galaxy/compute")
async def compute(body: GalaxyComputeRequest | None = None) -> dict[str, Any]:
    """Project + cluster + label the library (CPU-heavy; runs in a worker thread)."""
    body = body or GalaxyComputeRequest()
    try:
        return await run_in_threadpool(
            compute_galaxy, body.method, body.dims,
            n_neighbors=body.n_neighbors, min_dist=body.min_dist,
            perplexity=body.perplexity, n_clusters=body.n_clusters,
        )
    except GalaxyError as e:
        _raise(e)
        raise  # unreachable; keeps type-checkers happy

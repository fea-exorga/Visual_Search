"""Duplicate detection API (Milestone 5, PRD §6.4)."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Query

from ..dedup import DedupParams, find_duplicates

router = APIRouter(tags=["dedup"])


@router.get("/duplicates")
async def get_duplicates(
    threshold: float = Query(0.95, ge=0.0, le=1.0, description="cosine similarity for near-duplicates"),
    max_hamming: int = Query(6, ge=0, le=64, description="pHash Hamming distance for near-duplicates"),
    folder_id: int | None = Query(None),
):
    """Exact (SHA-256) + near (pHash / embedding cosine) duplicate groups.

    Missing perceptual hashes are back-filled lazily on the first call.
    """
    params = DedupParams(threshold=threshold, max_hamming=max_hamming, folder_id=folder_id)
    return await asyncio.to_thread(find_duplicates, params)

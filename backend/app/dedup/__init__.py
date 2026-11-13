"""Duplicate detection (Milestone 5): exact (SHA-256) + near (pHash / cosine)."""

from .service import (  # noqa: F401
    DedupParams,
    backfill_hashes,
    find_duplicates,
    hamming_pairs,
    UnionFind,
)
from ..indexer.hooks import register_post_index_hook

# Hashes are computed by the indexer; this only fills gaps (e.g. images
# linked to an exact-duplicate donor by the scanner, or legacy indexes).
register_post_index_hook("dedup_hashes", lambda: backfill_hashes())

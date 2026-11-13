"""Post-index hooks: other milestones register work to run after a job.

Milestone 2 registers the FAISS rebuild, Milestone 5 the duplicate hashes.
Hooks must be idempotent and must never raise into the indexer.
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)

_HOOKS: list[tuple[str, Callable[[], None]]] = []


def register_post_index_hook(name: str, fn: Callable[[], None]) -> None:
    if all(n != name for n, _ in _HOOKS):
        _HOOKS.append((name, fn))


def run_post_index_hooks() -> None:
    for name, fn in list(_HOOKS):
        try:
            fn()
        except Exception:
            logger.exception("Post-index hook %s failed", name)

"""
Pipeline orchestrator that ties Scanner → Color Extractor → OCR Engine → Embedder
into a batch loop with pause / cancel / progress reporting.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from PIL import Image

from ..config import get_settings
from ..db.schema import get_db
from .scanner import scan_directory, validate_and_prepare_image
from .color_extractor import extract_color_palette, save_color_palette
from .ocr_engine import get_ocr_engine
from .embedder import get_embedder

logger = logging.getLogger(__name__)


# ── Progress tracking ─────────────────────────────────────────────────


@dataclass
class IndexProgress:
    """Snapshot of indexing progress."""

    total_files: int = 0
    scanned: int = 0
    color_done: int = 0
    ocr_done: int = 0
    embedded: int = 0
    skipped: int = 0
    failed: int = 0
    current_stage: str = "idle"
    images_per_sec: float = 0.0
    is_paused: bool = False
    is_running: bool = False
    eta_seconds: float = 0.0
    errors: list[dict] = field(default_factory=list)


# ── Worker ────────────────────────────────────────────────────────────


class IndexWorker:
    """
    Asynchronous pipeline worker that processes images through four stages:
    color extraction, OCR, and batched embedding generation.

    ``scan_directory`` (from scanner.py) handles file discovery, validation,
    SHA-256 hashing, deduplication, and thumbnail creation and returns the
    ``image_id`` values that still need feature extraction.
    """

    def __init__(self, progress_callback: Callable[[IndexProgress], None] | None = None) -> None:
        self._callback = progress_callback
        self._progress = IndexProgress()
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # not paused initially
        self._cancelled = False

    # ── public control ────────────────────────────────────────────────

    def pause(self) -> None:
        self._pause_event.clear()
        self._progress.is_paused = True
        self._emit()

    def resume(self) -> None:
        self._pause_event.set()
        self._progress.is_paused = False
        self._emit()

    def cancel(self) -> None:
        self._cancelled = True
        # Also release pause so the loop can exit
        self._pause_event.set()
        self._progress.is_paused = False

    def get_progress(self) -> IndexProgress:
        return self._progress

    # ── main entry point ──────────────────────────────────────────────

    async def start(self, folder_ids: list[int], force_reindex: bool = False) -> None:
        """Run the full indexing pipeline for the given folders."""
        settings = get_settings()
        p = self._progress
        p.is_running = True
        p.current_stage = "scanning"
        self._emit()

        # 1. Collect image_ids that need processing from all folders
        pending_ids: list[int] = []
        for fid in folder_ids:
            folder_path = self._folder_path(fid)
            if folder_path is None:
                logger.warning("Folder id %d not found or disabled, skipping", fid)
                continue

            # scan_directory does validation/thumb/dedup and returns PENDING ids
            ids = await asyncio.to_thread(scan_directory, folder_path, fid)
            pending_ids.extend(ids)

        # If force_reindex, also include already-processed images
        if force_reindex:
            for fid in folder_ids:
                with get_db() as conn:
                    rows = conn.execute(
                        "SELECT image_id FROM images WHERE folder_id = ? AND status != 'PENDING'",
                        (fid,),
                    ).fetchall()
                    for r in rows:
                        if r["image_id"] not in pending_ids:
                            # Reset status so pipeline re-runs
                            conn.execute(
                                "UPDATE images SET status = 'PENDING', updated_at = CURRENT_TIMESTAMP WHERE image_id = ?",
                                (r["image_id"],),
                            )
                            pending_ids.append(r["image_id"])

        p.total_files = len(pending_ids)
        p.scanned = p.total_files  # scanning phase already done
        p.current_stage = "processing"
        self._emit()

        if not pending_ids:
            p.current_stage = "done"
            p.is_running = False
            self._emit()
            return

        # 2. Process each image through color → OCR → (batch embed)
        ocr_engine = get_ocr_engine()
        embedder = get_embedder()
        batch_size = settings.EMBEDDING_BATCH_SIZE

        embed_queue_ids: list[int] = []
        embed_queue_imgs: list[Image.Image] = []
        start_time = time.monotonic()
        processed = 0

        for image_id in pending_ids:
            # Respect pause
            await self._pause_event.wait()
            if self._cancelled:
                p.current_stage = "cancelled"
                p.is_running = False
                self._emit()
                return

            try:
                await self._process_single(
                    image_id, ocr_engine, embedder,
                    embed_queue_ids, embed_queue_imgs, batch_size,
                )
            except Exception as exc:
                logger.error("Failed processing image %d: %s", image_id, exc, exc_info=True)
                p.failed += 1
                p.errors.append({"image_id": image_id, "error": str(exc)})
                self._set_error(image_id, str(exc))

            processed += 1
            elapsed = time.monotonic() - start_time
            p.images_per_sec = round(processed / elapsed, 2) if elapsed > 0 else 0.0
            remaining = p.total_files - processed
            p.eta_seconds = round(remaining / p.images_per_sec, 1) if p.images_per_sec > 0 else 0.0
            self._emit()

        # 3. Flush remaining embedding batch
        if embed_queue_ids:
            p.current_stage = "embedding"
            self._emit()
            await self._flush_embeddings(embedder, embed_queue_ids, embed_queue_imgs)

        p.current_stage = "done"
        p.is_running = False
        self._emit()

    # ── per-image pipeline ────────────────────────────────────────────

    async def _process_single(
        self,
        image_id: int,
        ocr_engine,
        embedder,
        embed_queue_ids: list[int],
        embed_queue_imgs: list[Image.Image],
        batch_size: int,
    ) -> None:
        p = self._progress

        # Load image path from DB
        file_path = self._image_path(image_id)
        if file_path is None:
            p.skipped += 1
            return

        # a. Validate & prepare (re-opens the image for feature extraction)
        p.current_stage = "validating"
        img, width, height, fmt = await asyncio.to_thread(validate_and_prepare_image, file_path)

        # b. Color extraction
        p.current_stage = "color"
        palette = await asyncio.to_thread(extract_color_palette, img)
        await asyncio.to_thread(save_color_palette, image_id, palette)
        p.color_done += 1
        self._emit()

        # c. OCR
        p.current_stage = "ocr"
        ocr_result = await asyncio.to_thread(ocr_engine.extract_text, img)
        await asyncio.to_thread(ocr_engine.save_ocr_result, image_id, ocr_result)
        p.ocr_done += 1
        self._emit()

        # d. Queue for embedding batch
        embed_queue_ids.append(image_id)
        embed_queue_imgs.append(img)

        if len(embed_queue_ids) >= batch_size:
            p.current_stage = "embedding"
            self._emit()
            await self._flush_embeddings(embedder, embed_queue_ids, embed_queue_imgs)

    async def _flush_embeddings(
        self,
        embedder,
        queue_ids: list[int],
        queue_imgs: list[Image.Image],
    ) -> None:
        """Embed a batch and persist, then clear the queues."""
        p = self._progress
        ids = list(queue_ids)
        imgs = list(queue_imgs)
        queue_ids.clear()
        queue_imgs.clear()

        embeddings = await asyncio.to_thread(embedder.embed_batch, imgs)
        await asyncio.to_thread(embedder.save_embeddings, ids, embeddings)

        # Mark each image COMPLETED
        with get_db() as conn:
            for iid in ids:
                conn.execute(
                    "UPDATE images SET status = 'COMPLETED', updated_at = CURRENT_TIMESTAMP WHERE image_id = ?",
                    (iid,),
                )

        p.embedded += len(ids)
        self._emit()

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _folder_path(folder_id: int) -> str | None:
        with get_db() as conn:
            row = conn.execute(
                "SELECT path FROM watched_folders WHERE folder_id = ? AND enabled = 1",
                (folder_id,),
            ).fetchone()
        return row["path"] if row else None

    @staticmethod
    def _image_path(image_id: int) -> str | None:
        with get_db() as conn:
            row = conn.execute(
                "SELECT file_path FROM images WHERE image_id = ?",
                (image_id,),
            ).fetchone()
        return row["file_path"] if row else None

    @staticmethod
    def _set_error(image_id: int, message: str) -> None:
        with get_db() as conn:
            conn.execute(
                "UPDATE images SET status = 'ERROR', error_message = ?, updated_at = CURRENT_TIMESTAMP WHERE image_id = ?",
                (message, image_id),
            )

    def _emit(self) -> None:
        if self._callback is not None:
            try:
                self._callback(self._progress)
            except Exception:
                logger.debug("Progress callback failed", exc_info=True)

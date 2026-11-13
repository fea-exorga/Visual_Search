"""
Pipelined indexing worker (Milestone 1 → optimised in Milestone 5).

``scan_directory`` (scanner.py) discovers files, validates them, hashes them
(SHA-256), links exact duplicates and writes thumbnails. Everything left
``PENDING`` is processed here in chunks of ``EMBEDDING_BATCH_SIZE``:

    chunk N   : [decode → downscale → colour palette → OCR → pHash/dHash
                 → encoder preprocessing]   (thread pool, ≈ cpu_count workers)
                 → one DB transaction for colours / OCR / hashes
    chunk N-1 : [batch encode → vectors + COMPLETED]  (single embed thread)

The embed stage of chunk N-1 runs while chunk N is being prepared, so the
CPU-bound per-image work overlaps with ONNX inference. Pause / cancel are
honoured between chunks; progress is emitted only from the event-loop thread
(the SSE queue in main.py is not thread-safe).
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageOps

from ..config import get_settings
from ..db.schema import get_db
from ..embeddings import get_encoder
from .color_extractor import extract_color_palette
from .ocr_engine import get_ocr_engine
from .scanner import scan_directory

logger = logging.getLogger(__name__)

_COLOR_LOCK = threading.Lock()

# Longest side kept in memory for colour / hashes / embedding.
WORK_MAX_SIDE = 1024
# OCR benefits from more pixels (matches ocr_engine._MAX_SIDE).
OCR_MAX_SIDE = 1600


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


# ── Per-image stage (runs in the thread pool) ─────────────────────────


@dataclass
class _Prepared:
    image_id: int
    ok: bool = False
    skipped: bool = False
    error: str | None = None
    palette: list[dict] | None = None
    ocr: dict | None = None
    phash: str | None = None
    dhash: str | None = None
    embed_input: Any = None  # np.ndarray (pre-processed pixels) or PIL image
    timings: dict[str, float] = field(default_factory=dict)


def load_image(file_path: str | Path, max_side: int = OCR_MAX_SIDE) -> Image.Image:
    """Decode an image as RGB, cheaply capped to ``max_side``.

    JPEGs are decoded directly at a reduced DCT scale (``draft``), other
    formats are shrunk with ``reduce``-backed resampling. EXIF orientation is
    applied; alpha is composited over white (same as the scanner).
    """
    img = Image.open(file_path)
    if img.format == "JPEG":
        try:
            img.draft("RGB", (max_side, max_side))
        except Exception:  # pragma: no cover - draft is best effort
            pass
    img.load()
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "PA", "P"):
        if img.mode == "P":
            img = img.convert("RGBA")
        if "A" in img.mode:
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        else:
            img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.BILINEAR, reducing_gap=2.0)
    return img


def _downscale(img: Image.Image, max_side: int) -> Image.Image:
    if max(img.size) <= max_side:
        return img
    out = img.copy()
    out.thumbnail((max_side, max_side), Image.BILINEAR, reducing_gap=2.0)
    return out


def _prepare_one(image_id: int, file_path: str | None, ocr_engine, embedder) -> _Prepared:
    """Decode + colour + OCR + perceptual hashes + encoder preprocessing."""
    import imagehash

    res = _Prepared(image_id)
    if file_path is None:
        res.skipped = True
        return res
    t = time.perf_counter
    try:
        t0 = t()
        try:
            big = load_image(file_path, OCR_MAX_SIDE)
        except Exception as exc:
            raise ValueError(f"Invalid image {file_path}: {exc}") from exc
        small = _downscale(big, WORK_MAX_SIDE)
        t1 = t()
        # extract_color_palette() works on a 64×64 LANCZOS thumbnail; doing
        # that resize here keeps the identical result while the GIL-bound
        # MiniBatchKMeans runs serialised (threads only thrash on it).
        tiny = small.resize((64, 64), Image.LANCZOS, reducing_gap=3.0)
        with _COLOR_LOCK:
            res.palette = extract_color_palette(tiny)
        t2 = t()
        res.ocr = ocr_engine.extract_text(big) if ocr_engine.engine != "none" else {
            "raw_text": "", "mean_confidence": 0.0, "word_count": 0}
        t3 = t()
        res.phash = str(imagehash.phash(small))
        res.dhash = str(imagehash.dhash(small))
        t4 = t()
        if hasattr(embedder, "preprocess_uint8") and hasattr(embedder, "normalize_batch"):
            res.embed_input = embedder.preprocess_uint8(small)  # (H, W, 3) uint8
        elif hasattr(embedder, "preprocess") and hasattr(embedder, "encode_pixel_batch"):
            res.embed_input = embedder.preprocess(small)  # (3, H, W) float
        else:
            res.embed_input = small
        t5 = t()
        res.timings = {"decode": t1 - t0, "color": t2 - t1, "ocr": t3 - t2,
                       "hash": t4 - t3, "preprocess": t5 - t4}
        res.ok = True
    except Exception as exc:
        logger.warning("Failed processing image %d: %s", image_id, exc)
        res.error = str(exc)
    return res


def _write_features(prepared: list[_Prepared]) -> None:
    """Persist colours, OCR text, perceptual hashes and errors in ONE transaction."""
    ok = [r for r in prepared if r.ok]
    bad = [r for r in prepared if r.error is not None]
    with get_db() as conn:
        if ok:
            ids = [(r.image_id,) for r in ok]
            conn.executemany("DELETE FROM image_colors WHERE image_id = ?", ids)
            conn.executemany(
                """INSERT INTO image_colors (image_id, rank, hex_code, r, g, b, weight)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [(r.image_id, c["rank"], c["hex_code"], c["r"], c["g"], c["b"], c["weight"])
                 for r in ok for c in r.palette or []],
            )
            conn.executemany(
                """INSERT INTO image_ocr (image_id, raw_text, mean_confidence, word_count)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(image_id) DO UPDATE SET raw_text=excluded.raw_text,
                       mean_confidence=excluded.mean_confidence, word_count=excluded.word_count""",
                [(r.image_id, r.ocr["raw_text"], r.ocr["mean_confidence"], r.ocr["word_count"])
                 for r in ok],
            )
            conn.executemany(
                """INSERT INTO image_hashes (image_id, phash, dhash) VALUES (?, ?, ?)
                   ON CONFLICT(image_id) DO UPDATE SET phash=excluded.phash, dhash=excluded.dhash""",
                [(r.image_id, r.phash, r.dhash) for r in ok],
            )
            conn.executemany(
                "UPDATE images SET status = 'OCR_DONE', error_message = NULL, "
                "updated_at = CURRENT_TIMESTAMP WHERE image_id = ?",
                ids,
            )
        if bad:
            conn.executemany(
                "UPDATE images SET status = 'ERROR', error_message = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE image_id = ?",
                [(r.error, r.image_id) for r in bad],
            )


def _embed_and_store(embedder, prepared: list[_Prepared]) -> float:
    """Batch-encode a chunk and persist vectors + COMPLETED in one transaction."""
    t0 = time.perf_counter()
    inputs = [r.embed_input for r in prepared]
    if inputs and isinstance(inputs[0], np.ndarray):
        batch = np.stack(inputs)
        if batch.dtype == np.uint8:
            batch = embedder.normalize_batch(batch)
        vecs = embedder.encode_pixel_batch(batch)
    else:
        vecs = embedder.encode_images(inputs)
    vecs = np.ascontiguousarray(vecs, dtype=np.float32)
    dim = int(vecs.shape[1])
    with get_db() as conn:
        conn.executemany(
            """INSERT INTO image_embeddings (image_id, model, dim, vector)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(image_id) DO UPDATE SET model=excluded.model, dim=excluded.dim,
                   vector=excluded.vector,
                   updated_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now')""",
            [(r.image_id, embedder.name, dim, v.tobytes()) for r, v in zip(prepared, vecs)],
        )
        conn.executemany(
            "UPDATE images SET status = 'COMPLETED', updated_at = CURRENT_TIMESTAMP WHERE image_id = ?",
            [(r.image_id,) for r in prepared],
        )
    return time.perf_counter() - t0


def _limit_native_threads():
    """Cap OpenMP / BLAS pools (sklearn, numpy) at 1 thread for the job."""
    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=1)
    except Exception:  # pragma: no cover - threadpoolctl ships with sklearn
        return None


def _mark_error(image_ids: list[int], message: str) -> None:
    with get_db() as conn:
        conn.executemany(
            "UPDATE images SET status = 'ERROR', error_message = ?, "
            "updated_at = CURRENT_TIMESTAMP WHERE image_id = ?",
            [(message, i) for i in image_ids],
        )


# ── Worker ────────────────────────────────────────────────────────────


class IndexWorker:
    """
    Asynchronous, pipelined worker: per-image CPU stages (decode, colour,
    OCR, hashes, preprocessing) run on a thread pool while the previous
    chunk is being embedded on a dedicated thread.
    """

    def __init__(self, progress_callback: Callable[[IndexProgress], None] | None = None) -> None:
        self._callback = progress_callback
        self._progress = IndexProgress()
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # not paused initially
        self._cancelled = False
        # Aggregated stage timings (seconds, summed over images / batches);
        # used by scripts/benchmark.py.
        self.stage_times: dict[str, float] = {}

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
        t_scan = time.perf_counter()

        # 1. Collect image_ids that need processing from all folders
        pending_ids: list[int] = []
        for fid in folder_ids:
            folder_path = self._folder_path(fid)
            if folder_path is None:
                logger.warning("Folder id %d not found or disabled, skipping", fid)
                continue
            ids = await asyncio.to_thread(scan_directory, folder_path, fid)
            pending_ids.extend(ids)

        if force_reindex:
            seen = set(pending_ids)
            with get_db() as conn:
                for fid in folder_ids:
                    rows = conn.execute(
                        "SELECT image_id FROM images WHERE folder_id = ? AND status != 'PENDING'",
                        (fid,),
                    ).fetchall()
                    reset = [r["image_id"] for r in rows if r["image_id"] not in seen]
                    conn.executemany(
                        "UPDATE images SET status = 'PENDING', updated_at = CURRENT_TIMESTAMP WHERE image_id = ?",
                        [(i,) for i in reset],
                    )
                    pending_ids.extend(reset)
                    seen.update(reset)
        self.stage_times["scan_wall"] = time.perf_counter() - t_scan

        p.total_files = len(pending_ids)
        p.scanned = p.total_files  # scanning phase already done
        p.current_stage = "processing"
        self._emit()

        if not pending_ids:
            await self._finalize()
            return

        try:
            await self._run_pipeline(pending_ids, max(1, settings.EMBEDDING_BATCH_SIZE))
        except _Cancelled:
            p.current_stage = "cancelled"
            p.is_running = False
            self._emit()
            return
        except Exception as exc:
            p.current_stage = "error"
            p.is_running = False
            p.errors.append({"image_id": None, "error": str(exc)})
            self._emit()
            raise
        await self._finalize()

    async def _finalize(self) -> None:
        p = self._progress
        p.current_stage = "finalizing"
        self._emit()
        from .hooks import run_post_index_hooks
        try:  # registers the dedup-hash backfill hook (Milestone 5)
            from .. import dedup  # noqa: F401
        except Exception:  # pragma: no cover
            logger.debug("dedup hook unavailable", exc_info=True)
        t0 = time.perf_counter()
        await asyncio.to_thread(run_post_index_hooks)
        self.stage_times["hooks_wall"] = time.perf_counter() - t0
        p.current_stage = "done"
        p.is_running = False
        self._emit()

    # ── pipeline ──────────────────────────────────────────────────────

    async def _run_pipeline(self, pending_ids: list[int], batch_size: int) -> None:
        p = self._progress
        loop = asyncio.get_running_loop()
        ocr_engine = get_ocr_engine()
        embedder = await asyncio.to_thread(get_encoder)  # first call may load a model
        n_workers = int(os.environ.get("VS_INDEX_WORKERS", "0")) or max(1, os.cpu_count() or 1)
        paths = self._image_paths(pending_ids)
        chunks = [pending_ids[i:i + batch_size] for i in range(0, len(pending_ids), batch_size)]

        # Parallelism comes from running images concurrently, so every
        # per-image library must stay single-threaded: OpenMP inside several
        # concurrent Tesseract processes (and sklearn's KMeans) oversubscribes
        # the CPU so badly that a 0.3 s OCR call can take minutes.
        os.environ.setdefault("OMP_THREAD_LIMIT", "1")
        limiter = _limit_native_threads()

        start_time = time.monotonic()
        prep_pool = ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="vs-prep")
        embed_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vs-embed")
        inflight: tuple[asyncio.Future, list[_Prepared]] | None = None
        st = self.stage_times

        async def drain() -> None:
            nonlocal inflight
            if inflight is None:
                return
            fut, batch = inflight
            inflight = None
            try:
                st["embed"] = st.get("embed", 0.0) + await fut
                p.embedded += len(batch)
            except Exception as exc:
                logger.error("Embedding batch failed: %s", exc, exc_info=True)
                msg = f"embedding failed: {exc}"
                await asyncio.to_thread(_mark_error, [r.image_id for r in batch], msg)
                p.failed += len(batch)
                p.errors.extend({"image_id": r.image_id, "error": msg} for r in batch)
            self._update_rate(start_time)
            self._emit()

        try:
            for chunk in chunks:
                await self._pause_event.wait()
                if self._cancelled:
                    await drain()
                    raise _Cancelled

                # a. per-image CPU stage, concurrently (overlaps the in-flight embed)
                p.current_stage = "processing"
                futs = [
                    loop.run_in_executor(prep_pool, _prepare_one, iid, paths.get(iid), ocr_engine, embedder)
                    for iid in chunk
                ]
                prepared: list[_Prepared] = []
                for fut in asyncio.as_completed(futs):
                    r = await fut
                    prepared.append(r)
                    if r.ok:
                        p.color_done += 1
                        p.ocr_done += 1
                        for k, v in r.timings.items():
                            st[k] = st.get(k, 0.0) + v
                    elif r.skipped:
                        p.skipped += 1
                    else:
                        p.failed += 1
                        p.errors.append({"image_id": r.image_id, "error": r.error})
                    self._update_rate(start_time)
                    self._emit()
                order = {iid: i for i, iid in enumerate(chunk)}
                prepared.sort(key=lambda r: order[r.image_id])

                # b. one transaction for colours / OCR / hashes / errors
                t0 = time.perf_counter()
                await asyncio.to_thread(_write_features, prepared)
                st["db_write"] = st.get("db_write", 0.0) + time.perf_counter() - t0

                # c. hand the chunk to the embed thread (at most one in flight)
                await drain()
                ready = [r for r in prepared if r.ok]
                if ready:
                    p.current_stage = "embedding"
                    self._emit()
                    inflight = (loop.run_in_executor(embed_pool, _embed_and_store, embedder, ready), ready)
            await drain()
        finally:
            prep_pool.shutdown(wait=False, cancel_futures=True)
            embed_pool.shutdown(wait=True)
            if limiter is not None:
                limiter.restore_original_limits()

    def _update_rate(self, start_time: float) -> None:
        p = self._progress
        done = p.embedded + p.failed + p.skipped
        elapsed = time.monotonic() - start_time
        p.images_per_sec = round(done / elapsed, 2) if elapsed > 0 else 0.0
        remaining = max(0, p.total_files - done)
        p.eta_seconds = round(remaining / p.images_per_sec, 1) if p.images_per_sec > 0 else 0.0

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
    def _image_paths(image_ids: list[int]) -> dict[int, str]:
        out: dict[int, str] = {}
        with get_db() as conn:
            for i in range(0, len(image_ids), 900):
                part = image_ids[i:i + 900]
                rows = conn.execute(
                    f"SELECT image_id, file_path FROM images WHERE image_id IN ({','.join('?' * len(part))})",
                    part,
                ).fetchall()
                out.update({r["image_id"]: r["file_path"] for r in rows})
        return out

    @staticmethod
    def _set_error(image_id: int, message: str) -> None:
        _mark_error([image_id], message)

    def _emit(self) -> None:
        if self._callback is not None:
            try:
                self._callback(self._progress)
            except Exception:
                logger.debug("Progress callback failed", exc_info=True)


class _Cancelled(Exception):
    pass

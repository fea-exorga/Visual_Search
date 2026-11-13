"""
FastAPI application — REST endpoints, SSE progress streaming, and image browsing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .config import get_settings
from .db.schema import get_db, init_db
from .indexer.worker import IndexProgress, IndexWorker

logger = logging.getLogger(__name__)

# ── Lifespan ──────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("Database initialised")
    yield


# ── App ───────────────────────────────────────────────────────────────

app = FastAPI(title="Zero-Text Multimodal Desktop Search", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Feature routers (one module per milestone) ───────────────────────
from .api import dedup as _dedup_api  # noqa: E402  Milestone 5
from .api import galaxy as _galaxy_api  # noqa: E402  Milestone 3
from .api import search as _search_api  # noqa: E402  Milestone 2
from .api import system as _system_api  # noqa: E402

for _r in (_search_api.router, _galaxy_api.router, _dedup_api.router, _system_api.router):
    app.include_router(_r, prefix="/api/v1")

# ── Pydantic models ──────────────────────────────────────────────────


class FolderCreate(BaseModel):
    path: str


class IndexStartRequest(BaseModel):
    folder_ids: list[int] = []
    force_reindex: bool = False


class JobPauseRequest(BaseModel):
    job_id: str


# ── Global state ─────────────────────────────────────────────────────

active_jobs: dict[str, IndexWorker] = {}
progress_queues: dict[str, asyncio.Queue] = {}

# ── Folders ───────────────────────────────────────────────────────────


@app.post("/api/v1/folders", status_code=201)
def create_folder(body: FolderCreate):
    """Register a folder to watch."""
    p = Path(body.path).resolve()
    if not p.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a readable directory: {body.path}")
    if not os.access(p, os.R_OK):
        raise HTTPException(status_code=400, detail=f"Path is not readable: {body.path}")

    with get_db() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO watched_folders (path) VALUES (?)",
                (str(p),),
            )
        except Exception:
            raise HTTPException(status_code=409, detail="Folder already registered")

        folder_id = cur.lastrowid
        row = conn.execute(
            "SELECT * FROM watched_folders WHERE folder_id = ?", (folder_id,)
        ).fetchone()

    return dict(row)


@app.get("/api/v1/folders")
def list_folders():
    """List all watched folders with image count breakdown."""
    with get_db() as conn:
        folders = conn.execute("SELECT * FROM watched_folders ORDER BY folder_id").fetchall()
        result = []
        for f in folders:
            fid = f["folder_id"]
            counts = conn.execute(
                """
                SELECT
                    COUNT(*)                                           AS total,
                    SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END)   AS completed,
                    SUM(CASE WHEN status = 'PENDING'   THEN 1 ELSE 0 END)   AS pending,
                    SUM(CASE WHEN status = 'ERROR'     THEN 1 ELSE 0 END)   AS error
                FROM images WHERE folder_id = ?
                """,
                (fid,),
            ).fetchone()
            row = dict(f)
            row["image_counts"] = {
                "total": counts["total"] or 0,
                "completed": counts["completed"] or 0,
                "pending": counts["pending"] or 0,
                "error": counts["error"] or 0,
            }
            result.append(row)
    return result


@app.delete("/api/v1/folders/{folder_id}")
def delete_folder(folder_id: int):
    """Soft-delete a folder (set enabled=0)."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT folder_id FROM watched_folders WHERE folder_id = ?", (folder_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Folder not found")
        conn.execute(
            "UPDATE watched_folders SET enabled = 0 WHERE folder_id = ?", (folder_id,)
        )
    return {"status": "deleted", "folder_id": folder_id}


# ── Indexing ──────────────────────────────────────────────────────────


@app.post("/api/v1/index/start", status_code=202)
async def start_indexing(body: IndexStartRequest):
    """Start an indexing job and return its id."""
    job_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    progress_queues[job_id] = queue

    def on_progress(progress: IndexProgress):
        try:
            queue.put_nowait(progress)
        except asyncio.QueueFull:
            pass

    worker = IndexWorker(progress_callback=on_progress)
    active_jobs[job_id] = worker

    folder_ids = body.folder_ids
    if not folder_ids:
        # Default: all enabled folders
        with get_db() as conn:
            rows = conn.execute(
                "SELECT folder_id FROM watched_folders WHERE enabled = 1"
            ).fetchall()
            folder_ids = [r["folder_id"] for r in rows]

    async def _run():
        try:
            await worker.start(folder_ids, force_reindex=body.force_reindex)
        except Exception:
            logger.exception("Indexing job %s failed", job_id)
        finally:
            # Signal SSE stream that the job is finished
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

    asyncio.create_task(_run())
    return {"job_id": job_id}


@app.post("/api/v1/index/pause")
def pause_indexing(body: JobPauseRequest):
    worker = active_jobs.get(body.job_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Job not found")
    worker.pause()
    return {"status": "paused", "job_id": body.job_id}


@app.post("/api/v1/index/resume")
def resume_indexing(body: JobPauseRequest):
    worker = active_jobs.get(body.job_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Job not found")
    worker.resume()
    return {"status": "resumed", "job_id": body.job_id}


# ── SSE Stream ────────────────────────────────────────────────────────

KEEPALIVE_INTERVAL = 15  # seconds


@app.get("/api/v1/index/stream")
async def stream_progress(job_id: str = Query(...)):
    """Server-Sent Events stream for indexing progress."""
    queue = progress_queues.get(job_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="Job not found")

    async def _generate():
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_INTERVAL)
            except asyncio.TimeoutError:
                # keepalive
                yield ": keepalive\n\n"
                continue

            if item is None:
                # Job finished
                yield "event: done\ndata: {}\n\n"
                break

            p: IndexProgress = item
            data = json.dumps(
                {
                    "scanned": p.scanned,
                    "color_done": p.color_done,
                    "ocr_done": p.ocr_done,
                    "embedded": p.embedded,
                    "skipped": p.skipped,
                    "failed": p.failed,
                    "total": p.total_files,
                    "stage": p.current_stage,
                    "eta_sec": p.eta_seconds,
                    "images_per_sec": p.images_per_sec,
                },
            )
            yield f"data: {data}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


# ── Images ────────────────────────────────────────────────────────────


@app.get("/api/v1/images")
def list_images(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    folder_id: int | None = Query(None),
    status: str | None = Query(None),
):
    """Paginated image list with palette preview and OCR snippet."""
    with get_db() as conn:
        where_clauses = []
        params: list = []

        if folder_id is not None:
            where_clauses.append("i.folder_id = ?")
            params.append(folder_id)
        if status is not None:
            where_clauses.append("i.status = ?")
            params.append(status)

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        rows = conn.execute(
            f"""
            SELECT i.*,
                   o.raw_text AS ocr_text,
                   o.word_count AS ocr_word_count
            FROM images i
            LEFT JOIN image_ocr o ON o.image_id = i.image_id
            {where_sql}
            ORDER BY i.image_id DESC
            LIMIT ? OFFSET ?
            """,
            (*params, limit, offset),
        ).fetchall()

        total = conn.execute(
            f"SELECT COUNT(*) AS cnt FROM images i {where_sql}", params
        ).fetchone()["cnt"]

        result = []
        for r in rows:
            img = dict(r)
            iid = img["image_id"]

            # Dominant colors (top 5 for proportional bar)
            colors = conn.execute(
                "SELECT rank, hex_code, r, g, b, weight FROM image_colors WHERE image_id = ? ORDER BY rank",
                (iid,),
            ).fetchall()
            img["colors"] = [dict(c) for c in colors]
            img["palette_preview"] = [dict(c) for c in colors[:3]]

            # OCR snippet
            ocr_text = img.pop("ocr_text", None) or ""
            img["ocr_snippet"] = ocr_text[:120] + ("…" if len(ocr_text) > 120 else "")
            img.pop("ocr_word_count", None)

            # Thumbnail url
            img["thumbnail_url"] = f"/api/v1/thumbnails/{iid}"

            result.append(img)

    return {"images": result, "total": total, "limit": limit, "offset": offset}


@app.get("/api/v1/images/{image_id}")
def get_image_detail(image_id: int):
    """Full image metadata, colors, OCR text, and embedding info."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM images WHERE image_id = ?", (image_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Image not found")

        img = dict(row)

        # All 5 colors
        colors = conn.execute(
            "SELECT rank, hex_code, r, g, b, weight FROM image_colors WHERE image_id = ? ORDER BY rank",
            (image_id,),
        ).fetchall()
        img["colors"] = [dict(c) for c in colors]

        # Full OCR text
        ocr = conn.execute(
            "SELECT raw_text, mean_confidence, word_count FROM image_ocr WHERE image_id = ?",
            (image_id,),
        ).fetchone()
        img["ocr"] = dict(ocr) if ocr else None

        # Embedding info
        emb = conn.execute(
            "SELECT model, dim FROM image_embeddings WHERE image_id = ?", (image_id,)
        ).fetchone()
        img["has_embedding"] = emb is not None
        img["embedding_dim"] = emb["dim"] if emb else None
        img["model_name"] = emb["model"] if emb else None
        img["l2_norm"] = 1.0 if emb else None
        img["thumbnail_url"] = f"/api/v1/thumbnails/{image_id}"

    return img


# ── Thumbnails ────────────────────────────────────────────────────────


@app.get("/api/v1/thumbnails/{image_id}")
def serve_thumbnail(image_id: int):
    """Serve the JPEG thumbnail for an image."""
    settings = get_settings()
    thumb = settings.THUMBNAILS_DIR / f"{image_id}.jpg"
    if not thumb.is_file():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(thumb, media_type="image/jpeg")


# ── Stats ─────────────────────────────────────────────────────────────


@app.get("/api/v1/stats")
def dashboard_stats():
    """Dashboard overview stats."""
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) AS cnt FROM images").fetchone()["cnt"]

        by_status = conn.execute(
            "SELECT status, COUNT(*) AS cnt FROM images GROUP BY status"
        ).fetchall()
        status_map = {r["status"]: r["cnt"] for r in by_status}

        folder_count = conn.execute(
            "SELECT COUNT(*) AS cnt FROM watched_folders WHERE enabled = 1"
        ).fetchone()["cnt"]

        disk_bytes = conn.execute(
            "SELECT COALESCE(SUM(file_size), 0) AS total FROM images"
        ).fetchone()["total"]

    return {
        "total_images": total,
        "by_status": status_map,
        "total_folders": folder_count,
        "disk_usage_bytes": disk_bytes,
    }

"""
File-system scanner for the local directory indexer.

Walks watched directories, discovers images, computes hashes, detects
deltas, validates images, deduplicates by SHA-256, and generates thumbnails.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageOps

from ..config import get_settings
from ..db.schema import get_db

logger = logging.getLogger(__name__)

# Thumbnail size (longest edge)
_THUMB_MAX = 256


# ── Hashing ───────────────────────────────────────────────────────────

def compute_sha256(file_path: str | Path) -> str:
    """Return the hex-encoded SHA-256 digest of *file_path*."""
    settings = get_settings()
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(settings.HASH_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ── Delta detection ───────────────────────────────────────────────────

def check_delta(file_path: str | Path, file_size: int, mtime_ns: int) -> bool:
    """
    Return ``True`` if the file at *file_path* has changed compared to the
    stored *file_size* and *mtime_ns*.  If the file no longer exists on disk
    the image is also considered changed (caller should handle removal).
    """
    try:
        stat = os.stat(file_path)
    except FileNotFoundError:
        return True
    return stat.st_size != file_size or stat.st_mtime_ns != mtime_ns


# ── Image validation ─────────────────────────────────────────────────

def validate_and_prepare_image(file_path: str | Path) -> Tuple[Image.Image, int, int, str]:
    """
    Open and verify an image file.

    Returns:
        (PIL.Image, width, height, format_string)

    Raises:
        ValueError: if the file is not a valid image.
    """
    file_path = Path(file_path)
    try:
        img = Image.open(file_path)
        img.verify()  # lightweight check
        # Re-open after verify (verify may close internal state)
        img = Image.open(file_path)
        img.load()
    except Exception as exc:
        raise ValueError(f"Invalid image {file_path}: {exc}") from exc

    # Fix rotated phone photos
    img = ImageOps.exif_transpose(img)

    fmt = (img.format or file_path.suffix.lstrip(".")).upper()

    # Convert RGBA/P to RGB (composite over white background)
    if img.mode in ("RGBA", "LA", "PA", "P"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[-1] if "A" in img.mode else None)
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    return img, img.width, img.height, fmt


# ── De-duplication by hash ────────────────────────────────────────────

def get_or_link_by_hash(sha256: str, image_id: int) -> int | None:
    """
    Check whether another image with the same SHA-256 already exists.

    If a duplicate is found whose processing is further along, copy its
    results (colors, OCR, embedding row) to *image_id* and return the
    donor ``image_id``.  Otherwise return ``None``.
    """
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT image_id, status, embedding_row_id
            FROM images
            WHERE sha256 = ? AND image_id != ?
            ORDER BY
                CASE status
                    WHEN 'COMPLETED'       THEN 0
                    WHEN 'EMBEDDING_DONE'  THEN 1
                    WHEN 'OCR_DONE'        THEN 2
                    WHEN 'COLOR_DONE'      THEN 3
                    ELSE 4
                END
            LIMIT 1
            """,
            (sha256, image_id),
        ).fetchone()

        if row is None:
            return None

        donor_id: int = row["image_id"]
        donor_status: str = row["status"]

        # Nothing useful to copy from a PENDING/SCANNING donor
        if donor_status in ("PENDING", "SCANNING", "ERROR"):
            return None

        # Copy color rows
        colors = conn.execute(
            "SELECT rank, hex_code, r, g, b, weight FROM image_colors WHERE image_id = ?",
            (donor_id,),
        ).fetchall()
        for c in colors:
            conn.execute(
                """
                INSERT OR IGNORE INTO image_colors
                    (image_id, rank, hex_code, r, g, b, weight)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (image_id, c["rank"], c["hex_code"], c["r"], c["g"], c["b"], c["weight"]),
            )

        # Copy OCR row
        ocr = conn.execute(
            "SELECT raw_text, mean_confidence, word_count FROM image_ocr WHERE image_id = ?",
            (donor_id,),
        ).fetchone()
        if ocr:
            conn.execute(
                """
                INSERT OR IGNORE INTO image_ocr
                    (image_id, raw_text, mean_confidence, word_count)
                VALUES (?, ?, ?, ?)
                """,
                (image_id, ocr["raw_text"], ocr["mean_confidence"], ocr["word_count"]),
            )

        # Link embedding
        embedding_row_id = row["embedding_row_id"]
        conn.execute(
            """
            UPDATE images
            SET status = ?, embedding_row_id = ?,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE image_id = ?
            """,
            (donor_status, embedding_row_id, image_id),
        )

        return donor_id


# ── Thumbnail generation ─────────────────────────────────────────────

def generate_thumbnail(image: Image.Image, image_id: int) -> Path:
    """
    Create a JPEG thumbnail for *image* and save it under THUMBNAILS_DIR.

    Returns the path to the thumbnail file.
    """
    settings = get_settings()
    thumb = image.copy()
    thumb.thumbnail((_THUMB_MAX, _THUMB_MAX), Image.LANCZOS)

    if thumb.mode in ("RGBA", "P"):
        thumb = thumb.convert("RGB")

    thumb_path = settings.THUMBNAILS_DIR / f"{image_id}.jpg"
    thumb.save(thumb_path, "JPEG", quality=85)
    return thumb_path


# ── Directory scanner ─────────────────────────────────────────────────


def _scandir_recursive(
    root: Path, extensions: frozenset[str],
) -> List[str]:
    """Recursively discover image files using os.scandir(), skipping hidden dirs."""
    result: List[str] = []
    try:
        with os.scandir(root) as it:
            for entry in it:
                if entry.name.startswith("."):
                    continue
                if entry.is_dir(follow_symlinks=False):
                    result.extend(_scandir_recursive(Path(entry.path), extensions))
                elif entry.is_file(follow_symlinks=True):
                    if Path(entry.name).suffix.lower() in extensions:
                        result.append(str(Path(entry.path).resolve()))
    except PermissionError:
        logger.warning("Permission denied: %s", root)
    return result


def scan_directory(folder_path: str | Path, folder_id: int) -> List[int]:
    """
    Walk *folder_path*, discover supported images, and upsert them into the
    database.

    For each image file:
    1. Stat for size / mtime.
    2. Skip if already tracked and unchanged (delta check).
    3. Validate and open the image.
    4. Compute SHA-256.
    5. Check for duplicates by hash and link if possible.
    6. Generate a thumbnail.
    7. Insert / update the ``images`` row.

    Returns a list of ``image_id`` values that need further processing
    (status == 'PENDING' or reset to 'PENDING' after a change).
    """
    settings = get_settings()
    folder_path = Path(folder_path).resolve()
    new_ids: List[int] = []

    if not folder_path.is_dir():
        logger.error("Folder does not exist: %s", folder_path)
        return new_ids

    discovered_paths = _scandir_recursive(folder_path, settings.SUPPORTED_EXTENSIONS)

    for abs_path in discovered_paths:
        fpath = Path(abs_path)

        try:
            stat = fpath.stat()
        except OSError as exc:
            logger.warning("Cannot stat %s: %s", fpath, exc)
            continue

        file_size = stat.st_size
        mtime_ns = stat.st_mtime_ns

        with get_db() as conn:
            existing = conn.execute(
                "SELECT image_id, file_size, mtime_ns FROM images WHERE file_path = ?",
                (abs_path,),
            ).fetchone()

            if existing and not check_delta(abs_path, existing["file_size"], existing["mtime_ns"]):
                # File unchanged — skip
                continue

            # Validate
            try:
                img, width, height, fmt = validate_and_prepare_image(fpath)
            except ValueError as exc:
                logger.warning("Skipping invalid image %s: %s", fpath, exc)
                if existing:
                    conn.execute(
                        """
                        UPDATE images
                        SET status = 'ERROR', error_message = ?,
                            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                        WHERE image_id = ?
                        """,
                        (str(exc), existing["image_id"]),
                    )
                continue

            sha256 = compute_sha256(fpath)

            if existing:
                image_id = existing["image_id"]
                conn.execute(
                    """
                    UPDATE images
                    SET sha256 = ?, width = ?, height = ?, file_size = ?,
                        mtime_ns = ?, format = ?, status = 'PENDING',
                        error_message = NULL,
                        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    WHERE image_id = ?
                    """,
                    (sha256, width, height, file_size, mtime_ns, fmt, image_id),
                )
            else:
                cur = conn.execute(
                    """
                    INSERT INTO images
                        (file_path, folder_id, sha256, width, height,
                         file_size, mtime_ns, format)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (abs_path, folder_id, sha256, width, height,
                     file_size, mtime_ns, fmt),
                )
                image_id = cur.lastrowid

            # Try dedup by hash
            donor = get_or_link_by_hash(sha256, image_id)
            if donor is not None:
                logger.info(
                    "Image %d linked to duplicate donor %d", image_id, donor
                )

            # Generate thumbnail
            try:
                generate_thumbnail(img, image_id)
            except Exception as exc:
                logger.warning("Thumbnail failed for %s: %s", fpath, exc)

            # If still PENDING after hash-link check, it needs processing
            row = conn.execute(
                "SELECT status FROM images WHERE image_id = ?",
                (image_id,),
            ).fetchone()
            if row and row["status"] == "PENDING":
                new_ids.append(image_id)

    # Update last_scan_at on the folder
    with get_db() as conn:
        conn.execute(
            """
            UPDATE watched_folders
            SET last_scan_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE folder_id = ?
            """,
            (folder_id,),
        )

    logger.info(
        "Scan of %s complete: %d image(s) need processing", folder_path, len(new_ids)
    )
    return new_ids

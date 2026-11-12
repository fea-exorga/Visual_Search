"""OCR text extraction wrapper using EasyOCR."""

import logging
import re

import numpy as np
from PIL import Image

from ..config import get_settings
from ..db.schema import get_db

logger = logging.getLogger(__name__)


class OCREngine:
    """EasyOCR-based text extraction with lazy initialization and singleton reader."""

    _reader = None  # Class-level singleton to avoid re-initialization

    def __init__(self, engine: str = "easyocr", gpu: bool = True) -> None:
        self._engine = engine
        self._gpu = gpu
        self._available = True

    def _ensure_reader(self):
        """Lazy-load the EasyOCR Reader on first use."""
        if OCREngine._reader is not None:
            return
        try:
            import easyocr
            OCREngine._reader = easyocr.Reader(["en"], gpu=self._gpu)
            logger.info("EasyOCR reader initialized (gpu=%s)", self._gpu)
        except Exception as exc:
            logger.warning("EasyOCR not available, OCR disabled: %s", exc)
            self._available = False

    def extract_text(self, image: Image.Image) -> dict:
        """
        Extract text from a PIL Image.

        Returns:
            Dict with raw_text, mean_confidence, and word_count.
        """
        empty = {"raw_text": "", "mean_confidence": 0.0, "word_count": 0}

        try:
            self._ensure_reader()
            if not self._available or OCREngine._reader is None:
                return empty

            settings = get_settings()
            img_array = np.array(image.convert("RGB"))

            # Run EasyOCR
            results = OCREngine._reader.readtext(img_array)

            # Filter by confidence threshold
            threshold = settings.OCR_CONFIDENCE_THRESHOLD
            filtered = [
                (bbox, text, conf)
                for bbox, text, conf in results
                if conf >= threshold
            ]

            if not filtered:
                return empty

            # Sort by reading order: y (top-to-bottom) then x (left-to-right)
            # bbox[0] is the top-left corner [x, y]
            filtered.sort(key=lambda r: (r[0][0][1], r[0][0][0]))

            # Concatenate text blocks
            raw_text = " ".join(text for _, text, _ in filtered)

            # Normalize whitespace
            raw_text = re.sub(r"\s+", " ", raw_text).strip()

            # Compute mean confidence
            mean_conf = sum(conf for _, _, conf in filtered) / len(filtered)

            # Word count
            word_count = len(raw_text.split()) if raw_text else 0

            return {
                "raw_text": raw_text,
                "mean_confidence": round(mean_conf, 4),
                "word_count": word_count,
            }

        except Exception as exc:
            logger.error("OCR extraction failed: %s", exc, exc_info=True)
            return empty

    def save_ocr_result(self, image_id: int, result: dict) -> None:
        """
        Persist OCR result to the database, update FTS5 index, mark OCR_DONE.

        Args:
            image_id: Primary key of the image row.
            result: Output of extract_text().
        """
        with get_db() as db:
            db.execute(
                """
                INSERT OR REPLACE INTO image_ocr
                    (image_id, raw_text, mean_confidence, word_count)
                VALUES (?, ?, ?, ?)
                """,
                (
                    image_id,
                    result["raw_text"],
                    result["mean_confidence"],
                    result["word_count"],
                ),
            )

            # Update FTS5 index — delete old entry then insert new
            db.execute(
                "DELETE FROM image_ocr_fts WHERE rowid = ?",
                (image_id,),
            )
            if result["raw_text"]:
                db.execute(
                    "INSERT INTO image_ocr_fts (rowid, raw_text) VALUES (?, ?)",
                    (image_id, result["raw_text"]),
                )

            db.execute(
                "UPDATE images SET status = 'OCR_DONE', updated_at = CURRENT_TIMESTAMP WHERE image_id = ?",
                (image_id,),
            )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_ocr_engine: OCREngine | None = None


def get_ocr_engine() -> OCREngine:
    """Return the module-level OCREngine singleton."""
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = OCREngine()
    return _ocr_engine

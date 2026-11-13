"""OCR text extraction with pluggable backends (Tesseract / EasyOCR).

``Settings.OCR_ENGINE``:
  * ``auto``      — Tesseract if the binary is present, else EasyOCR, else none
  * ``tesseract`` — pytesseract (fast on CPU, no model download)
  * ``easyocr``   — EasyOCR (needs torch; GPU accelerated when available)
  * ``none``      — skip OCR
"""

from __future__ import annotations

import logging
import re
import shutil

import numpy as np
from PIL import Image, ImageOps

from ..config import get_settings
from ..db.schema import get_db

logger = logging.getLogger(__name__)

_EMPTY = {"raw_text": "", "mean_confidence": 0.0, "word_count": 0}
_MAX_SIDE = 1600  # downscale huge screenshots before OCR


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class OCREngine:
    """Lazy, backend-agnostic OCR wrapper."""

    def __init__(self, engine: str | None = None) -> None:
        requested = (engine or get_settings().OCR_ENGINE).lower()
        self.engine = self._resolve(requested)
        self._reader = None
        logger.info("OCR engine: %s (requested %s)", self.engine, requested)

    @staticmethod
    def _resolve(requested: str) -> str:
        if requested in ("tesseract", "auto") and shutil.which("tesseract"):
            try:
                import pytesseract  # noqa: F401
                return "tesseract"
            except ImportError:
                pass
        if requested in ("easyocr", "auto"):
            try:
                import easyocr  # noqa: F401
                return "easyocr"
            except Exception:
                pass
        return "none"

    # ── backends ──────────────────────────────────────────────────────
    def _tesseract(self, img: Image.Image, threshold: float) -> dict:
        import pytesseract

        gray = ImageOps.grayscale(img)
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config="--psm 3")
        words, confs = [], []
        for text, conf in zip(data["text"], data["conf"]):
            conf = float(conf)
            text = text.strip()
            if text and conf >= 0 and conf / 100.0 >= threshold and re.search(r"\w", text):
                words.append(text)
                confs.append(conf / 100.0)
        raw = _normalise(" ".join(words))
        return {
            "raw_text": raw,
            "mean_confidence": round(float(np.mean(confs)), 4) if confs else 0.0,
            "word_count": len(raw.split()) if raw else 0,
        }

    def _easyocr(self, img: Image.Image, threshold: float) -> dict:
        if self._reader is None:
            import easyocr
            try:
                import torch
                gpu = torch.cuda.is_available()
            except ImportError:
                gpu = False
            self._reader = easyocr.Reader(["en"], gpu=gpu)
        results = [r for r in self._reader.readtext(np.array(img)) if r[2] >= threshold]
        if not results:
            return dict(_EMPTY)
        results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
        raw = _normalise(" ".join(t for _, t, _ in results))
        return {
            "raw_text": raw,
            "mean_confidence": round(sum(c for *_, c in results) / len(results), 4),
            "word_count": len(raw.split()) if raw else 0,
        }

    # ── public ────────────────────────────────────────────────────────
    def extract_text(self, image: Image.Image) -> dict:
        if self.engine == "none":
            return dict(_EMPTY)
        try:
            img = image.convert("RGB")
            if max(img.size) > _MAX_SIDE:
                img.thumbnail((_MAX_SIDE, _MAX_SIDE))
            threshold = get_settings().OCR_CONFIDENCE_THRESHOLD
            if self.engine == "tesseract":
                return self._tesseract(img, threshold)
            return self._easyocr(img, threshold)
        except Exception as exc:
            logger.error("OCR extraction failed: %s", exc)
            return dict(_EMPTY)

    def save_ocr_result(self, image_id: int, result: dict) -> None:
        """Persist OCR text (FTS5 kept in sync by triggers) and mark OCR_DONE."""
        with get_db() as db:
            db.execute(
                """
                INSERT INTO image_ocr (image_id, raw_text, mean_confidence, word_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(image_id) DO UPDATE SET raw_text=excluded.raw_text,
                    mean_confidence=excluded.mean_confidence, word_count=excluded.word_count
                """,
                (image_id, result["raw_text"], result["mean_confidence"], result["word_count"]),
            )
            db.execute(
                "UPDATE images SET status = 'OCR_DONE', updated_at = CURRENT_TIMESTAMP WHERE image_id = ?",
                (image_id,),
            )


_ocr_engine: OCREngine | None = None


def get_ocr_engine() -> OCREngine:
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = OCREngine()
    return _ocr_engine


def reset_ocr_engine() -> None:
    global _ocr_engine
    _ocr_engine = None

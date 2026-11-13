"""
Application configuration using a Settings dataclass.
All data paths are relative to DATA_DIR (defaults to <project_root>/data/).
Directories are created on first access via get_settings().
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet

def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # backend/../..


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    # ── base directories ──────────────────────────────────────────────
    DATA_DIR: Path = field(default_factory=lambda: _PROJECT_ROOT / "data")

    # ── derived paths (computed in __post_init__) ─────────────────────
    DB_PATH: Path = field(init=False)
    THUMBNAILS_DIR: Path = field(init=False)
    MODELS_DIR: Path = field(init=False)
    FAISS_INDEX_PATH: Path = field(init=False)
    PROJECTIONS_DIR: Path = field(init=False)

    # ── image handling ────────────────────────────────────────────────
    SUPPORTED_EXTENSIONS: FrozenSet[str] = frozenset(
        {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    )

    # ── embedding ─────────────────────────────────────────────────────
    # backend: "onnx" (default, torch-free) | "torch" | "fake" (tests)
    EMBED_BACKEND: str = field(default_factory=lambda: _env("VS_EMBED_BACKEND", "onnx"))
    ONNX_REPO_ID: str = field(default_factory=lambda: _env("VS_ONNX_REPO", "Xenova/clip-vit-base-patch32"))
    # "fp32" | "fp16" | "int8"
    ONNX_PRECISION: str = field(default_factory=lambda: _env("VS_ONNX_PRECISION", "fp32"))
    TORCH_MODEL_NAME: str = field(default_factory=lambda: _env("VS_TORCH_MODEL", "google/siglip-base-patch16-224"))
    EMBEDDING_DIM: int = 512  # CLIP ViT-B/32 projection dim (fake encoder uses this too)
    EMBEDDING_BATCH_SIZE: int = field(default_factory=lambda: int(_env("VS_BATCH_SIZE", "32")))

    # ── OCR ───────────────────────────────────────────────────────────
    # "auto" (tesseract → easyocr → none) | "tesseract" | "easyocr" | "none"
    OCR_ENGINE: str = field(default_factory=lambda: _env("VS_OCR_ENGINE", "auto"))
    OCR_CONFIDENCE_THRESHOLD: float = 0.35

    # ── color extraction ──────────────────────────────────────────────
    COLOR_CLUSTERS: int = 5

    # ── hashing ───────────────────────────────────────────────────────
    HASH_CHUNK_SIZE: int = 65_536  # 64 KB

    def __post_init__(self) -> None:
        # frozen=True requires object.__setattr__ for post-init assignment
        object.__setattr__(self, "DB_PATH", self.DATA_DIR / "index.db")
        object.__setattr__(self, "THUMBNAILS_DIR", self.DATA_DIR / "thumbnails")
        object.__setattr__(self, "MODELS_DIR", Path(_env("VS_MODELS_DIR", str(self.DATA_DIR / "models"))))
        object.__setattr__(self, "FAISS_INDEX_PATH", self.DATA_DIR / "faiss.index")
        object.__setattr__(self, "PROJECTIONS_DIR", self.DATA_DIR / "projections")

        # Ensure required directories exist
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.PROJECTIONS_DIR.mkdir(parents=True, exist_ok=True)


# ── singleton ─────────────────────────────────────────────────────────
_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the singleton Settings instance, creating it on first call."""
    global _settings
    if _settings is None:
        data_dir_env = os.environ.get("SEARCH_DATA_DIR")
        if data_dir_env:
            _settings = Settings(DATA_DIR=Path(data_dir_env))
        else:
            _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Forget the cached settings (tests change env vars between runs)."""
    global _settings
    _settings = None

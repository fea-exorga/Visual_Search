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

try:
    import torch
    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    _CUDA_AVAILABLE = False

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # backend/../..


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    # ── base directories ──────────────────────────────────────────────
    DATA_DIR: Path = field(default_factory=lambda: _PROJECT_ROOT / "data")

    # ── derived paths (computed in __post_init__) ─────────────────────
    DB_PATH: Path = field(init=False)
    EMBEDDINGS_PATH: Path = field(init=False)
    THUMBNAILS_DIR: Path = field(init=False)

    # ── image handling ────────────────────────────────────────────────
    SUPPORTED_EXTENSIONS: FrozenSet[str] = frozenset(
        {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    )

    # ── embedding ─────────────────────────────────────────────────────
    EMBEDDING_DIM: int = 768  # SigLIP default
    EMBEDDING_BATCH_SIZE: int = field(init=False)

    # ── OCR ───────────────────────────────────────────────────────────
    OCR_CONFIDENCE_THRESHOLD: float = 0.35

    # ── color extraction ──────────────────────────────────────────────
    COLOR_CLUSTERS: int = 5

    # ── hashing ───────────────────────────────────────────────────────
    HASH_CHUNK_SIZE: int = 65_536  # 64 KB

    def __post_init__(self) -> None:
        # frozen=True requires object.__setattr__ for post-init assignment
        object.__setattr__(self, "DB_PATH", self.DATA_DIR / "index.db")
        object.__setattr__(self, "EMBEDDINGS_PATH", self.DATA_DIR / "embeddings.npy")
        object.__setattr__(self, "THUMBNAILS_DIR", self.DATA_DIR / "thumbnails")
        object.__setattr__(
            self,
            "EMBEDDING_BATCH_SIZE",
            32 if _CUDA_AVAILABLE else 8,
        )

        # Ensure required directories exist
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)


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

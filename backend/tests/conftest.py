"""Shared fixtures.

Every test runs against a throw-away data dir with the deterministic
``fake`` encoder and OCR disabled unless the test opts in, so the suite is
fast and needs no model downloads. ``indexed_client`` returns a FastAPI
TestClient whose library already contains the synthetic dataset.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _reset_singletons() -> None:
    from app import config
    from app.embeddings import reset_encoder
    from app.indexer import ocr_engine

    config.reset_settings()
    reset_encoder()
    ocr_engine.reset_ocr_engine()
    try:  # Milestone 2 cache
        from app.search import faiss_index
        faiss_index.reset_index()
    except Exception:
        pass


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "data"
    monkeypatch.setenv("SEARCH_DATA_DIR", str(d))
    monkeypatch.setenv("VS_EMBED_BACKEND", os.environ.get("VS_TEST_BACKEND", "fake"))
    monkeypatch.setenv("VS_OCR_ENGINE", os.environ.get("VS_TEST_OCR", "none"))
    monkeypatch.setenv("VS_MODELS_DIR", str(Path(__file__).resolve().parents[2] / "data" / "models"))
    _reset_singletons()
    from app.db.schema import init_db
    init_db()
    yield d
    _reset_singletons()


COLORS = {
    "red": (220, 30, 30), "blue": (30, 60, 220), "green": (30, 180, 60),
    "yellow": (240, 220, 40), "purple": (140, 40, 180), "black": (10, 10, 10),
}


def make_dataset(root: Path) -> dict[str, Path]:
    """Small synthetic library: solid colours, shapes, a text slide, a dup."""
    root.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for name, rgb in COLORS.items():
        p = root / f"solid_{name}.png"
        Image.new("RGB", (160, 120), rgb).save(p)
        out[f"solid_{name}"] = p
        img = Image.new("RGB", (160, 160), "white")
        ImageDraw.Draw(img).ellipse((20, 20, 140, 140), fill=rgb)
        p = root / "shapes" / f"circle_{name}.jpg"
        p.parent.mkdir(exist_ok=True)
        img.save(p, quality=92)
        out[f"circle_{name}"] = p
    slide = Image.new("RGB", (640, 360), "white")
    d = ImageDraw.Draw(slide)
    font = ImageFont.load_default(size=36)
    d.text((40, 60), "GRADIENT DESCENT LECTURE", fill="black", font=font)
    d.text((40, 140), "learning rate and loss function", fill="black", font=font)
    p = root / "slides" / "lecture_01.png"
    p.parent.mkdir(exist_ok=True)
    slide.save(p)
    out["slide"] = p
    # exact duplicate (different path, same bytes)
    dup = root / "copy_of_solid_red.png"
    dup.write_bytes(out["solid_red"].read_bytes())
    out["dup_red"] = dup
    # near duplicate (re-encoded + resized)
    near = root / "near_circle_blue.jpg"
    Image.open(out["circle_blue"]).resize((150, 150)).save(near, quality=70)
    out["near_circle_blue"] = near
    return out


@pytest.fixture()
def dataset(tmp_path) -> dict[str, Path]:
    return make_dataset(tmp_path / "images")


def index_folder(path: Path) -> int:
    """Register + fully index a folder synchronously. Returns folder_id."""
    from app.db.schema import get_db
    from app.indexer.worker import IndexWorker

    with get_db() as conn:
        fid = conn.execute("INSERT INTO watched_folders (path) VALUES (?)", (str(path.resolve()),)).lastrowid
    asyncio.run(IndexWorker().start([fid]))
    return fid


@pytest.fixture()
def client(data_dir):
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture()
def indexed_client(data_dir, dataset):
    from fastapi.testclient import TestClient
    from app.main import app

    index_folder(dataset["solid_red"].parent)
    with TestClient(app) as c:
        c.dataset = dataset  # type: ignore[attr-defined]
        yield c


def image_id_for(path: Path) -> int:
    from app.db.schema import get_db

    with get_db() as conn:
        return conn.execute("SELECT image_id FROM images WHERE file_path = ?", (str(path.resolve()),)).fetchone()["image_id"]

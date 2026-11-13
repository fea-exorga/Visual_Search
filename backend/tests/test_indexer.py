"""Milestone 1 — indexer pipeline end-to-end on the synthetic dataset."""

import numpy as np

from app.db.schema import get_db
from app.db.vectors import load_vectors
from tests.conftest import image_id_for, index_folder


def test_index_pipeline(data_dir, dataset):
    index_folder(dataset["solid_red"].parent)
    with get_db() as conn:
        statuses = {r["status"] for r in conn.execute("SELECT status FROM images")}
        n = conn.execute("SELECT COUNT(*) c FROM images").fetchone()["c"]
        n_colors = conn.execute("SELECT COUNT(DISTINCT image_id) c FROM image_colors").fetchone()["c"]
    assert statuses == {"COMPLETED"}
    assert n == len(dataset)
    assert n_colors == n
    ids, vecs = load_vectors()
    assert len(ids) == n and vecs.shape[1] == 512
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)


def test_exact_duplicate_reuses_features(data_dir, dataset):
    index_folder(dataset["solid_red"].parent)
    a, b = image_id_for(dataset["solid_red"]), image_id_for(dataset["dup_red"])
    ids, vecs = load_vectors(image_ids=[a, b])
    assert len(ids) == 2
    assert np.allclose(vecs[0], vecs[1])


def test_dominant_color(data_dir, dataset):
    index_folder(dataset["solid_red"].parent)
    iid = image_id_for(dataset["solid_blue"])
    with get_db() as conn:
        top = conn.execute("SELECT r,g,b FROM image_colors WHERE image_id=? AND rank=1", (iid,)).fetchone()
    assert top["b"] > 180 and top["r"] < 80


def test_reindex_is_incremental(data_dir, dataset):
    fid = index_folder(dataset["solid_red"].parent)
    from app.indexer.scanner import scan_directory
    assert scan_directory(dataset["solid_red"].parent, fid) == []


def test_api_lists_images(indexed_client):
    r = indexed_client.get("/api/v1/images?limit=100")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == len(indexed_client.dataset)
    detail = indexed_client.get(f"/api/v1/images/{body['images'][0]['image_id']}").json()
    assert detail["has_embedding"] and detail["embedding_dim"] == 512
    info = indexed_client.get("/api/v1/system/info").json()
    assert info["vector_models"][0]["count"] == len(indexed_client.dataset)


def test_tesseract_ocr(data_dir, dataset, monkeypatch):
    import shutil
    import pytest
    if not shutil.which("tesseract"):
        pytest.skip("tesseract binary not installed")
    from app.indexer.ocr_engine import OCREngine
    from PIL import Image
    eng = OCREngine("tesseract")
    out = eng.extract_text(Image.open(dataset["slide"]))
    assert "gradient" in out["raw_text"].lower()
    assert out["word_count"] >= 4

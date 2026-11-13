"""Milestone 2: FAISS index and hybrid cross-modal search."""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from tests.conftest import image_id_for

SEARCH = "/api/v1/search"


def _paths(resp_json: dict) -> list[str]:
    return [Path(h["file_path"]).name for h in resp_json["results"]]


def _png_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _insert_ocr(image_id: int, text: str) -> None:
    from app.db.schema import get_db

    with get_db() as conn:  # FTS5 is kept in sync by triggers
        conn.execute(
            "INSERT OR REPLACE INTO image_ocr (image_id, raw_text, mean_confidence, word_count) VALUES (?, ?, ?, ?)",
            (image_id, text, 0.9, len(text.split())),
        )


# ── contract ──────────────────────────────────────────────────────────


def test_text_red_ranks_red_first(indexed_client):
    r = indexed_client.post(SEARCH, json={"text": "red", "k": 10})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"results", "took_ms", "faiss_ms", "modes", "total_candidates"}
    assert body["modes"] == ["text"]
    top2 = _paths(body)[:2]
    assert any("red" in name for name in top2), top2
    hit = body["results"][0]
    assert set(hit) == {
        "image_id", "score", "scores", "file_path", "thumbnail_url",
        "width", "height", "colors", "ocr_snippet", "cluster_id",
    }
    assert set(hit["scores"]) == {"semantic"}
    assert 0.0 <= hit["score"] <= 1.0
    assert hit["thumbnail_url"] == f"/api/v1/thumbnails/{hit['image_id']}"
    assert hit["colors"] and hit["colors"][0]["hex_code"].startswith("#")
    scores = [h["score"] for h in body["results"]]
    assert scores == sorted(scores, reverse=True)


def test_similar_returns_near_duplicate(indexed_client):
    ds = indexed_client.dataset
    qid = image_id_for(ds["circle_blue"])
    r = indexed_client.get(f"{SEARCH}/similar/{qid}", params={"k": 5})
    assert r.status_code == 200
    ids = [h["image_id"] for h in r.json()["results"]]
    assert qid not in ids
    assert image_id_for(ds["near_circle_blue"]) in ids[:3]


def test_similar_unknown_image_404(indexed_client):
    assert indexed_client.get(f"{SEARCH}/similar/999999").status_code == 404


def test_color_query_ranks_blue_first(indexed_client):
    r = indexed_client.post(SEARCH, json={"colors": ["#1E3CDC"], "k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["modes"] == ["color"]
    assert all("blue" in name for name in _paths(body)[:3]), _paths(body)


def test_ocr_query_returns_slide_first(indexed_client):
    ds = indexed_client.dataset
    slide = image_id_for(ds["slide"])
    _insert_ocr(slide, "GRADIENT DESCENT LECTURE learning rate and loss function")
    _insert_ocr(image_id_for(ds["solid_black"]), "unrelated caption text")
    r = indexed_client.post(SEARCH, json={"ocr": "gradient", "k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["results"][0]["image_id"] == slide
    assert body["results"][0]["scores"]["ocr"] == pytest.approx(1.0)
    assert "GRADIENT" in body["results"][0]["ocr_snippet"]
    # FTS syntax characters in user input must not break the query.
    assert indexed_client.post(SEARCH, json={"ocr": 'grad* "AND (', "k": 5}).status_code == 200


def test_ocr_with_tesseract(indexed_client):
    import shutil

    if not shutil.which("tesseract"):
        pytest.skip("tesseract not installed")
    from app.indexer.ocr_engine import OCREngine

    ds = indexed_client.dataset
    slide = image_id_for(ds["slide"])
    text = OCREngine("tesseract").extract_text(Image.open(ds["slide"]))["raw_text"]
    _insert_ocr(slide, text)
    r = indexed_client.post(SEARCH, json={"ocr": "lecture", "k": 3})
    assert r.json()["results"][0]["image_id"] == slide


def test_sketch_red_circle_returns_red_image(indexed_client):
    canvas = Image.new("RGBA", (400, 300), (0, 0, 0, 0))
    ImageDraw.Draw(canvas).ellipse((120, 60, 280, 220), fill=(225, 25, 25, 255))
    r = indexed_client.post(SEARCH, json={"sketch_png": _png_data_url(canvas), "k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["modes"] == ["sketch"]
    assert set(body["results"][0]["scores"]) == {"semantic", "color"}
    assert any("red" in name for name in _paths(body)[:3]), _paths(body)


def test_blank_sketch_is_400(indexed_client):
    blank = _png_data_url(Image.new("RGBA", (64, 64), (0, 0, 0, 0)))
    assert indexed_client.post(SEARCH, json={"sketch_png": blank}).status_code == 400
    assert indexed_client.post(SEARCH, json={"sketch_png": "not-base64!!"}).status_code == 400


def test_upload_endpoint(indexed_client):
    ds = indexed_client.dataset
    data = ds["near_circle_blue"].read_bytes()
    r = indexed_client.post(
        f"{SEARCH}/upload", files={"file": ("q.jpg", data, "image/jpeg")}, data={"k": "4"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["modes"] == ["upload"]
    assert len(body["results"]) == 4
    top2 = {h["image_id"] for h in body["results"][:2]}
    assert {image_id_for(ds["near_circle_blue"]), image_id_for(ds["circle_blue"])} == top2

    bad = indexed_client.post(f"{SEARCH}/upload", files={"file": ("x.png", b"nope", "image/png")})
    assert bad.status_code == 400


def test_empty_query_is_400(indexed_client):
    assert indexed_client.post(SEARCH, json={}).status_code == 400
    assert indexed_client.post(SEARCH, json={"text": "   ", "ocr": ""}).status_code == 400
    assert indexed_client.post(SEARCH, json={"colors": ["blue"]}).status_code == 400


def test_hybrid_weights_and_filters(indexed_client):
    from app.db.schema import get_db

    ds = indexed_client.dataset
    slide = image_id_for(ds["slide"])
    _insert_ocr(slide, "GRADIENT DESCENT LECTURE")
    body = indexed_client.post(
        SEARCH, json={"text": "red", "ocr": "gradient", "colors": ["#DC1E1E"], "k": 20}
    ).json()
    assert body["modes"] == ["text", "color", "ocr"]
    assert set(body["results"][0]["scores"]) == {"semantic", "ocr", "color"}
    assert slide in [h["image_id"] for h in body["results"]]

    # OCR-dominant weights put the slide first.
    body = indexed_client.post(
        SEARCH, json={"text": "red", "ocr": "gradient", "weights": {"semantic": 0.0, "ocr": 1.0}, "k": 5}
    ).json()
    assert body["results"][0]["image_id"] == slide

    # Cluster filter: ignored before projections exist, applied afterwards.
    red, blue = image_id_for(ds["solid_red"]), image_id_for(ds["solid_blue"])
    unfiltered = indexed_client.post(SEARCH, json={"text": "red", "cluster_ids": [7], "k": 50}).json()
    assert len(unfiltered["results"]) > 2
    with get_db() as conn:
        conn.executemany(
            "INSERT INTO image_projections (image_id, method, dims, x, y, cluster_id) VALUES (?, 'umap', 2, 0, 0, ?)",
            [(red, 7), (blue, 8)],
        )
    filtered = indexed_client.post(SEARCH, json={"text": "red", "cluster_ids": [7], "k": 50}).json()
    assert [h["image_id"] for h in filtered["results"]] == [red]
    assert filtered["results"][0]["cluster_id"] == 7

    # Folder filter.
    assert indexed_client.post(SEARCH, json={"text": "red", "folder_id": 999}).json()["results"] == []


def test_rebuild_and_persistence(indexed_client):
    from app.search import faiss_index

    r = indexed_client.post(f"{SEARCH}/rebuild")
    assert r.status_code == 200
    body = r.json()
    n_images = len(indexed_client.dataset)
    assert body["n_vectors"] == n_images
    assert body["index_type"] == "IndexIDMap2(IndexFlatIP)"
    assert body["build_ms"] >= 0

    faiss_index.reset_index()
    loaded = faiss_index.get_index()  # comes back from disk, not rebuilt
    assert len(loaded) == n_images and loaded.build_ms == 0.0
    assert indexed_client.post(SEARCH, json={"text": "blue", "k": 3}).status_code == 200


# ── VectorIndex unit tests ────────────────────────────────────────────


def _random_unit(n: int, dim: int, seed: int = 0) -> np.ndarray:
    mat = np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)
    return mat / np.linalg.norm(mat, axis=1, keepdims=True)


def test_vector_index_allowed_ids():
    from app.search.faiss_index import VectorIndex

    mat = _random_unit(500, 64)
    ids = np.arange(1000, 1500, dtype=np.int64)
    index = VectorIndex(ids, mat)
    got, scores, _ = index.search(mat[3], 5)
    assert got[0, 0] == 1003 and scores[0, 0] == pytest.approx(1.0, abs=1e-5)
    allowed = [1100, 1200, 1300, 99999]
    got, scores, _ = index.search(mat[3], 10, allowed_ids=allowed)
    assert sorted(got[0].tolist()) == [1100, 1200, 1300]
    expected = (mat[[100, 200, 300]] @ mat[3]).max()
    assert scores[0, 0] == pytest.approx(expected, abs=1e-5)


def test_faiss_latency_5k_p50_under_1ms():
    from app.search.faiss_index import VectorIndex

    mat = _random_unit(5000, 512)
    index = VectorIndex(np.arange(5000, dtype=np.int64), mat)
    queries = _random_unit(300, 512, seed=1)
    for q in queries[:20]:  # warm-up
        index.search(q, 60)
    timings = []
    for q in queries:
        t0 = time.perf_counter()
        index.search(q, 60)
        timings.append((time.perf_counter() - t0) * 1000.0)
    p50 = float(np.median(timings))
    print(f"FAISS 5k x 512-d top-60 p50 = {p50:.3f} ms, p95 = {np.percentile(timings, 95):.3f} ms")
    assert p50 < 1.0


def test_filter_only_query_browses_folder(indexed_client):
    folders = indexed_client.get("/api/v1/folders").json()
    r = indexed_client.post("/api/v1/search", json={"folder_id": folders[0]["folder_id"], "k": 100})
    assert r.status_code == 200
    body = r.json()
    assert body["modes"] == ["browse"]
    assert len(body["results"]) == len(indexed_client.dataset)

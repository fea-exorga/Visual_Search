"""Milestone 3 — projection, clustering, labelling and the /galaxy API."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from app.projection.cluster import PALETTE, cluster_points, k_range
from app.projection.labels import assign_labels, label_clusters, top_keyword
from app.projection.reduce import normalize_coords, project

from .conftest import index_folder, make_dataset

FAST = {"n_epochs": 100}  # keep UMAP cheap in tests


def _blobs(n_per: int = 40, dim: int = 512, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Three well separated gaussian blobs of L2-normalised vectors."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((3, dim))
    mats, labels = [], []
    for c in range(3):
        mats.append(centers[c] + 0.15 * rng.standard_normal((n_per, dim)))
        labels += [c] * n_per
    mat = np.vstack(mats).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    return mat, np.array(labels)


@pytest.fixture(scope="module")
def blobs() -> tuple[np.ndarray, np.ndarray]:
    return _blobs()


# ── reduce ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", ["umap", "tsne", "pca"])
@pytest.mark.parametrize("dims", [2, 3])
def test_project_shape_and_range(blobs, method, dims):
    mat, _ = blobs
    out = project(mat, method, dims, **(FAST if method == "umap" else {}))
    assert out.shape == (len(mat), dims)
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
    assert out.min() >= -1.0 and out.max() <= 1.0
    # every axis is actually used (scaled by max abs → some |value| == 1)
    assert np.allclose(np.abs(out).max(axis=0), 1.0)


def test_project_deterministic(blobs):
    mat, _ = blobs
    a = project(mat, "umap", 2, **FAST)
    b = project(mat, "umap", 2, **FAST)
    np.testing.assert_allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 5])
@pytest.mark.parametrize("method", ["umap", "tsne"])
def test_project_tiny_inputs(method, n):
    mat, _ = _blobs(n_per=2)
    mat = mat[:n]
    out = project(mat, method, 3, **(FAST if method == "umap" else {}))
    assert out.shape == (n, 3)
    assert np.all(np.isfinite(out)) and np.all(np.abs(out) <= 1.0)


def test_project_rejects_bad_args(blobs):
    mat, _ = blobs
    with pytest.raises(ValueError):
        project(mat, "mds", 2)
    with pytest.raises(ValueError):
        project(mat, "pca", 4)


def test_normalize_constant_axis():
    out = normalize_coords(np.array([[1.0, 5.0], [3.0, 5.0]]))
    np.testing.assert_allclose(out, [[-1.0, 0.0], [1.0, 0.0]])


# ── cluster ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", ["umap", "tsne", "pca"])
def test_auto_k_recovers_blobs(blobs, method):
    mat, truth = blobs
    coords = project(mat, method, 2, **(FAST if method == "umap" else {}))
    res = cluster_points(coords)
    assert res.k == 3
    assert adjusted_rand_score(truth, res.labels) > 0.9
    assert res.sizes.sum() == len(mat)
    assert list(res.sizes) == sorted(res.sizes, reverse=True)  # ids ordered by size
    assert res.colors == list(PALETTE[:3])
    assert res.centroids.shape == (3, 2)


def test_fixed_k_and_edge_cases():
    rng = np.random.default_rng(1)
    pts = rng.uniform(-1, 1, (60, 2))
    assert cluster_points(pts, n_clusters=5).k == 5
    assert cluster_points(np.zeros((1, 2))).k == 1
    assert cluster_points(np.zeros((4, 3))).k == 1  # all identical → one cluster
    assert cluster_points(np.zeros((0, 2))).k == 0
    assert cluster_points(pts[:2], n_clusters=10).k == 2  # clamped to distinct points
    assert list(k_range(15)) == [2, 3, 4, 5]
    assert max(k_range(10_000)) == 24


# ── labels ───────────────────────────────────────────────────────────


class _VocabEncoder:
    """Text encoder whose prompts map onto one-hot axes, for exact label checks."""

    name = "vocab-test"
    dim = 8

    def encode_texts(self, texts):
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, t in enumerate(texts):
            out[i, 0 if "red" in t else 1 if "slide" in t else 2] = 1.0
        return out


def test_label_clusters_zero_shot_and_ocr_keyword():
    mat = np.zeros((4, 8), np.float32)
    mat[:2, 0] = 1.0  # cluster 0 → red
    mat[2:, 1] = 1.0  # cluster 1 → slide
    vocab = ("solid red", "lecture slide with text", "cat")
    names = label_clusters(mat, np.array([0, 0, 1, 1]), _VocabEncoder(),
                           ocr_texts=[None, "", "Gradient descent", "the gradient of loss"],
                           vocabulary=vocab)
    assert names == ["solid red", "lecture slide with text · gradient"]


def test_assign_labels_unique():
    vocab_emb = np.eye(3, dtype=np.float32)
    cents = np.array([[1, 0, 0], [0.9, 0, 0.2]], np.float32)  # both prefer prompt 0
    names = assign_labels(cents, vocab_emb, ["a", "b", "c"])
    assert names == ["a", "c"]


def test_top_keyword():
    assert top_keyword(["The loss and the LOSS", "learning loss rate 2024", None]) == "loss"
    assert top_keyword([None, "a an to 1234"]) is None


# ── API ──────────────────────────────────────────────────────────────


def test_galaxy_404_before_compute_and_validation(indexed_client):
    c = indexed_client
    assert c.get("/api/v1/galaxy?method=umap&dims=2").status_code == 404
    assert c.get("/api/v1/galaxy?method=bogus&dims=2").status_code == 400
    assert c.post("/api/v1/galaxy/compute", json={"method": "bogus"}).status_code == 400
    assert c.post("/api/v1/galaxy/compute", json={"dims": 4}).status_code == 400


def test_galaxy_409_without_images(client):
    assert client.post("/api/v1/galaxy/compute", json={}).status_code == 409


def _check_payload(payload: dict, method: str, dims: int, n: int) -> None:
    assert payload["method"] == method and payload["dims"] == dims
    assert payload["computed_at"] and payload["stale"] is False
    pts = payload["points"]
    assert len(pts) == n and len({p["image_id"] for p in pts}) == n
    cids = {cl["cluster_id"] for cl in payload["clusters"]}
    for p in pts:
        assert p["cluster_id"] in cids
        assert -1.0 <= p["x"] <= 1.0 and -1.0 <= p["y"] <= 1.0
        assert ("z" in p) == (dims == 3)
    assert sum(cl["size"] for cl in payload["clusters"]) == n
    for cl in payload["clusters"]:
        assert len(cl["centroid"]) == dims
        assert cl["color"].startswith("#") and len(cl["color"]) == 7
        assert isinstance(cl["label"], str) and cl["label"]


def test_galaxy_compute_and_get(indexed_client):
    c = indexed_client
    r = c.post("/api/v1/galaxy/compute", json={"method": "umap", "dims": 2})
    assert r.status_code == 200, r.text
    _check_payload(r.json(), "umap", 2, 15)

    r = c.post("/api/v1/galaxy/compute", json={"method": "tsne", "dims": 3, "perplexity": 5})
    assert r.status_code == 200, r.text
    _check_payload(r.json(), "tsne", 3, 15)

    got = c.get("/api/v1/galaxy?method=umap&dims=2").json()
    _check_payload(got, "umap", 2, 15)
    got3 = c.get("/api/v1/galaxy", params={"method": "tsne", "dims": 3}).json()
    _check_payload(got3, "tsne", 3, 15)
    assert c.get("/api/v1/galaxy?method=umap&dims=3").status_code == 404

    # recompute with a fixed k replaces (not appends) the stored rows
    r = c.post("/api/v1/galaxy/compute", json={"method": "umap", "dims": 2, "n_clusters": 2})
    _check_payload(r.json(), "umap", 2, 15)
    assert len(r.json()["clusters"]) == 2


def test_galaxy_separates_text_slide_from_solids(indexed_client):
    from .conftest import image_id_for

    c = indexed_client
    payload = c.post("/api/v1/galaxy/compute", json={"method": "pca", "dims": 2}).json()
    by_id = {p["image_id"]: p["cluster_id"] for p in payload["points"]}
    slide = by_id[image_id_for(c.dataset["slide"])]
    solids = {by_id[image_id_for(c.dataset[f"solid_{n}"])] for n in ("red", "blue", "green")}
    assert slide not in solids


def test_galaxy_stale_after_new_folder(indexed_client, tmp_path):
    c = indexed_client
    assert c.post("/api/v1/galaxy/compute", json={"method": "pca", "dims": 2}).status_code == 200
    assert c.get("/api/v1/galaxy?method=pca&dims=2").json()["stale"] is False

    index_folder(make_dataset(tmp_path / "more")["solid_red"].parent)
    got = c.get("/api/v1/galaxy?method=pca&dims=2").json()
    assert got["stale"] is True
    assert len(got["points"]) == 15  # still the old projection until recompute

    fresh = c.post("/api/v1/galaxy/compute", json={"method": "pca", "dims": 2}).json()
    assert fresh["stale"] is False and len(fresh["points"]) == 30


def test_distinctive_keywords_skip_library_wide_filler():
    from app.projection.labels import distinctive_keywords
    texts = ["works gradient loss", "works gradient", "works gradient", "works pizza menu",
             "works pizza", "works pizza"]
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert distinctive_keywords(texts, labels, 2) == ["gradient", "pizza"]

"""Milestone 5 — duplicate detection (exact SHA-256, near via pHash / cosine)."""

import numpy as np

from app.db.schema import get_db
from tests.conftest import image_id_for, index_folder


def _group_of(payload, image_id):
    for g in payload["groups"]:
        if any(i["image_id"] == image_id for i in g["images"]):
            return g
    return None


def test_hamming_pairs_vectorised():
    from app.dedup import hamming_pairs

    rng = np.random.default_rng(0)
    h = rng.integers(0, 2**63, size=300, dtype=np.uint64)
    h[10] = h[3] ^ np.uint64(0b1011)       # distance 3
    h[200] = h[150] ^ np.uint64(0xFF)      # distance 8
    i, j, d = hamming_pairs(h, 6, chunk=64)
    pairs = {(int(a), int(b)): int(c) for a, b, c in zip(i, j, d)}
    assert pairs.get((3, 10)) == 3
    assert (150, 200) not in pairs
    i, j, d = hamming_pairs(h, 8, chunk=7)
    assert {(3, 10), (150, 200)} <= {(int(a), int(b)) for a, b in zip(i, j)}


def test_union_find():
    from app.dedup import UnionFind

    uf = UnionFind(6)
    uf.union(0, 1); uf.union(1, 2); uf.union(4, 5)
    groups = sorted(sorted(g) for g in uf.groups().values())
    assert groups == [[0, 1, 2], [3], [4, 5]]


def test_hashes_stored_by_indexer(data_dir, dataset):
    index_folder(dataset["solid_red"].parent)
    with get_db() as conn:
        n = conn.execute("SELECT COUNT(*) c FROM images").fetchone()["c"]
        h = conn.execute("SELECT COUNT(*) c FROM image_hashes WHERE phash IS NOT NULL AND dhash IS NOT NULL").fetchone()["c"]
    assert h == n  # indexed images + the scanner-linked exact dup (via hook)


def test_exact_and_near_groups(data_dir, dataset):
    from app.dedup import DedupParams, find_duplicates

    index_folder(dataset["solid_red"].parent)
    red, dup = image_id_for(dataset["solid_red"]), image_id_for(dataset["dup_red"])
    blue, near = image_id_for(dataset["circle_blue"]), image_id_for(dataset["near_circle_blue"])

    res = find_duplicates()
    g = _group_of(res, red)
    assert g is not None and g["kind"] == "exact"
    assert {i["image_id"] for i in g["images"]} == {red, dup}
    assert all(i["similarity"] == 1.0 for i in g["images"])
    g = _group_of(res, blue)
    assert g is not None and g["kind"] == "near" and near in {i["image_id"] for i in g["images"]}
    # distinct solid colours / differently coloured circles are not duplicates
    assert _group_of(res, image_id_for(dataset["solid_blue"])) is None
    assert _group_of(res, image_id_for(dataset["circle_red"])) is None

    # pHash alone (cosine path disabled)
    res = find_duplicates(DedupParams(threshold=1.01, max_hamming=6))
    g = _group_of(res, blue)
    assert g is not None and near in {i["image_id"] for i in g["images"]}

    # cosine alone (pHash path disabled)
    res = find_duplicates(DedupParams(threshold=0.95, max_hamming=-1))
    g = _group_of(res, blue)
    assert g is not None and near in {i["image_id"] for i in g["images"]}
    sims = {i["image_id"]: i["similarity"] for i in g["images"]}
    assert sims[near] >= 0.95


def test_backfill_missing_hashes(data_dir, dataset):
    from app.dedup import backfill_hashes

    index_folder(dataset["solid_red"].parent)
    with get_db() as conn:
        conn.execute("DELETE FROM image_hashes")
    assert backfill_hashes() == len(dataset)
    assert backfill_hashes() == 0


def test_duplicates_api_shape(indexed_client):
    ds = indexed_client.dataset
    r = indexed_client.get("/api/v1/duplicates", params={"threshold": 0.95, "max_hamming": 6})
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"groups", "n_groups", "n_images"}
    assert body["n_groups"] == len(body["groups"]) >= 2
    assert body["n_images"] == sum(len(g["images"]) for g in body["groups"])
    for g in body["groups"]:
        assert g["kind"] in ("exact", "near") and isinstance(g["group_id"], int)
        assert len(g["images"]) >= 2
        for im in g["images"]:
            assert set(im) >= {"image_id", "file_path", "thumbnail_url", "file_size", "width", "height", "similarity"}
            assert 0.0 <= im["similarity"] <= 1.0
    red = image_id_for(ds["solid_red"])
    assert _group_of(body, red)["kind"] == "exact"
    assert indexed_client.get("/api/v1/duplicates", params={"threshold": 2}).status_code == 422

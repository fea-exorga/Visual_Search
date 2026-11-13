"""Exact + near-duplicate grouping.

* **exact** — identical SHA-256 (computed by the scanner).
* **near**  — perceptual hash (pHash) Hamming distance ≤ ``max_hamming``
  *or* embedding cosine ≥ ``threshold`` (FAISS ``range_search`` on an exact
  inner-product index over ``load_vectors()``).

Both signals over-match on their own: pHash is a grayscale 32×32 DCT, so it
is colour-blind (a red and a blue circle share a hash) and degenerate for
flat images; CLIP puts *different* slides of one deck, or two screenshots of
one app, at cosine 0.95-0.98, while a 5 % crop of the same photo can drop
to 0.90. Every near candidate is therefore verified at the pixel level
(:mod:`app.dedup.verify`): similar mean chromaticity and a high
brightness-invariant colour correlation of the thumbnails after a small
crop-alignment search. ``DedupParams(verify=False)`` gives the raw
"pHash ≤ max_hamming OR cosine ≥ threshold" rule.

Pairs are merged with union-find, so groups are the connected components of
the duplicate graph. All-pairs Hamming distances are computed with a
vectorised XOR + popcount over ``uint64`` (≈ 0.3 s for 10k images).
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from ..db.schema import get_db
from ..db.vectors import current_models, load_vectors
from .verify import verify_pairs

logger = logging.getLogger(__name__)

HASH_BITS = 64


@dataclass
class DedupParams:
    threshold: float = 0.95        # cosine similarity for embedding near-dups
    max_hamming: int = 6           # pHash Hamming distance for near-dups
    min_hash_bits: int = 6         # hashes with < this many 1 (or 0) bits are degenerate
    folder_id: int | None = None
    # ── verification of near candidates (see verify.py) ──
    verify: bool = True
    cosine_margin: float = 0.07    # also consider cosine ≥ threshold - margin (crops)
    max_chroma_dist: float = 0.03  # L1 distance of mean-palette chromaticity
    min_coarse_corr: float = 0.93  # 12 px colour correlation, best of 9 crop alignments
    min_pixel_corr: float = 0.96   # 64 px colour correlation at the refined alignment
    max_flat_color_dist: float = 30.0  # both images (nearly) flat → compare mean colours


# ── union-find ────────────────────────────────────────────────────────


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n)
        self.rank = np.zeros(n, dtype=np.int32)

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return int(root)

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def groups(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            out.setdefault(self.find(i), []).append(i)
        return out


# ── hashing ───────────────────────────────────────────────────────────


def hex_to_u64(hexes: list[str | None]) -> tuple[np.ndarray, np.ndarray]:
    """Parse 16-char hex hashes → (uint64 array, valid mask)."""
    vals = np.zeros(len(hexes), dtype=np.uint64)
    valid = np.zeros(len(hexes), dtype=bool)
    for i, h in enumerate(hexes):
        if h and len(h) == 16:
            try:
                vals[i] = np.uint64(int(h, 16))
                valid[i] = True
            except ValueError:
                pass
    return vals, valid


def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    b = x.view(np.uint8).reshape(*x.shape, 8)  # pragma: no cover - numpy < 2
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return table[b].sum(-1)


def hamming_pairs(hashes: np.ndarray, max_hamming: int, valid: np.ndarray | None = None,
                  chunk: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All pairs ``i < j`` with Hamming(hashes[i], hashes[j]) ≤ max_hamming.

    Returns ``(i, j, distance)``. Vectorised XOR + popcount in row chunks, so
    10k hashes need ~50M popcounts and a few tens of MB at a time.
    """
    hashes = np.asarray(hashes, dtype=np.uint64)
    n = len(hashes)
    idx = np.arange(n) if valid is None else np.flatnonzero(valid)
    h = hashes[idx]
    m = len(h)
    I, J, D = [], [], []
    for s in range(0, m, chunk):
        e = min(m, s + chunk)
        d = _popcount(h[s:e, None] ^ h[None, s:])  # only columns ≥ s (upper triangle)
        r, c = np.nonzero(d <= max_hamming)
        c = c + s
        keep = c > (r + s)
        r, c = r[keep] + s, c[keep]
        I.append(idx[r]); J.append(idx[c]); D.append(d[r - s, c - s])
    if not I:
        return (np.empty(0, np.int64),) * 2 + (np.empty(0, np.uint8),)
    return np.concatenate(I), np.concatenate(J), np.concatenate(D).astype(np.uint8)


def compute_hashes(file_path: str) -> tuple[str, str]:
    import imagehash
    from ..indexer.worker import WORK_MAX_SIDE, load_image

    img = load_image(file_path, WORK_MAX_SIDE)
    return str(imagehash.phash(img)), str(imagehash.dhash(img))


def backfill_hashes() -> int:
    """Store pHash/dHash for COMPLETED images that lack them. Returns #filled.

    Exact duplicates copy the hash of a same-SHA-256 sibling; the rest are
    computed in a thread pool.
    """
    with get_db() as conn:
        rows = conn.execute(
            """SELECT i.image_id, i.file_path, i.sha256 FROM images i
               LEFT JOIN image_hashes h ON h.image_id = i.image_id
               WHERE i.status = 'COMPLETED' AND h.phash IS NULL"""
        ).fetchall()
        if not rows:
            return 0
        copied = conn.execute(
            """INSERT OR REPLACE INTO image_hashes (image_id, phash, dhash)
               SELECT i.image_id, h2.phash, h2.dhash FROM images i
               JOIN images i2 ON i2.sha256 = i.sha256 AND i2.image_id != i.image_id
               JOIN image_hashes h2 ON h2.image_id = i2.image_id AND h2.phash IS NOT NULL
               LEFT JOIN image_hashes h ON h.image_id = i.image_id
               WHERE i.status = 'COMPLETED' AND h.phash IS NULL AND i.sha256 IS NOT NULL
               GROUP BY i.image_id"""
        ).rowcount
        rows = conn.execute(
            """SELECT i.image_id, i.file_path FROM images i
               LEFT JOIN image_hashes h ON h.image_id = i.image_id
               WHERE i.status = 'COMPLETED' AND h.phash IS NULL"""
        ).fetchall()

    def one(r):
        try:
            return (r["image_id"],) + compute_hashes(r["file_path"])
        except Exception as exc:
            logger.warning("hash failed for %s: %s", r["file_path"], exc)
            return None

    results = []
    if rows:
        with ThreadPoolExecutor(max_workers=max(1, os.cpu_count() or 1)) as pool:
            results = [x for x in pool.map(one, rows) if x is not None]
        with get_db() as conn:
            conn.executemany(
                """INSERT INTO image_hashes (image_id, phash, dhash) VALUES (?, ?, ?)
                   ON CONFLICT(image_id) DO UPDATE SET phash=excluded.phash, dhash=excluded.dhash""",
                results,
            )
    return max(0, copied) + len(results)


# ── grouping ──────────────────────────────────────────────────────────


def _load_meta(folder_id: int | None) -> list[dict]:
    sql = """SELECT i.image_id, i.file_path, i.sha256, i.file_size, i.width, i.height,
                    h.phash, h.dhash,
                    (SELECT SUM(c.r * c.weight) / SUM(c.weight) FROM image_colors c WHERE c.image_id = i.image_id) AS mr,
                    (SELECT SUM(c.g * c.weight) / SUM(c.weight) FROM image_colors c WHERE c.image_id = i.image_id) AS mg,
                    (SELECT SUM(c.b * c.weight) / SUM(c.weight) FROM image_colors c WHERE c.image_id = i.image_id) AS mb
             FROM images i LEFT JOIN image_hashes h ON h.image_id = i.image_id
             WHERE i.status = 'COMPLETED'"""
    params: list = []
    if folder_id is not None:
        sql += " AND i.folder_id = ?"
        params.append(folder_id)
    sql += " ORDER BY i.image_id"
    with get_db() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def _cosine_pairs(mat: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairs (i < j) with cosine ≥ threshold via FAISS range_search (exact IP)."""
    if len(mat) < 2:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)
    try:
        import faiss

        index = faiss.IndexFlatIP(mat.shape[1])
        index.add(mat)
        lims, D, I = index.range_search(mat, float(threshold) - 1e-6)
        rows = np.repeat(np.arange(len(mat)), np.diff(lims.astype(np.int64)))
    except ImportError:  # pragma: no cover - faiss is a hard dependency
        sims = mat @ mat.T
        rows, I = np.nonzero(sims >= threshold)
        D = sims[rows, I]
    keep = I > rows
    return rows[keep].astype(np.int64), I[keep].astype(np.int64), D[keep].astype(np.float32)


def find_duplicates(params: DedupParams | None = None, *, backfill: bool = True) -> dict:
    """Group duplicate images. Returns the PRD §6.4 ``/duplicates`` payload."""
    p = params or DedupParams()
    if backfill:
        backfill_hashes()
    meta = _load_meta(p.folder_id)
    n = len(meta)
    if n == 0:
        return {"groups": [], "n_groups": 0, "n_images": 0}
    ids = np.array([m["image_id"] for m in meta], dtype=np.int64)
    pos = {int(i): k for k, i in enumerate(ids)}

    # Embeddings of the dominant model, aligned with ``meta``.
    mat = None
    has_vec = np.zeros(n, dtype=bool)
    models = current_models()
    if models:
        vid, vmat = load_vectors(model=models[0][0])
        if len(vid):
            mat = np.zeros((n, vmat.shape[1]), dtype=np.float32)
            for i, v in zip(vid, vmat):
                k = pos.get(int(i))
                if k is not None:
                    mat[k] = v
                    has_vec[k] = True

    uf = UnionFind(n)

    # 1. exact — same SHA-256
    by_sha: dict[str, list[int]] = {}
    for k, m in enumerate(meta):
        if m["sha256"]:
            by_sha.setdefault(m["sha256"], []).append(k)
    for members in by_sha.values():
        for k in members[1:]:
            uf.union(members[0], k)

    # 2. near candidates — pHash Hamming (flat/degenerate hashes ignored)
    hashes, valid = hex_to_u64([m["phash"] for m in meta])
    bits = _popcount(hashes).astype(int)
    valid &= (bits >= p.min_hash_bits) & (bits <= HASH_BITS - p.min_hash_bits)
    hi, hj, _ = hamming_pairs(hashes, p.max_hamming, valid)
    cand_i, cand_j = [hi], [hj]

    # 3. near candidates — embedding cosine (FAISS range search). The search
    #    radius is widened by ``cosine_margin`` so crops / strong re-encodes
    #    (which CLIP moves to ~0.88-0.95) are still considered; every
    #    candidate below ``threshold`` must pass the same pixel verification.
    if mat is not None and has_vec.sum() >= 2 and p.threshold <= 1.0:  # > 1 disables cosine
        sub = np.flatnonzero(has_vec)
        radius = p.threshold - (p.cosine_margin if p.verify else 0.0)
        ci, cj, _ = _cosine_pairs(np.ascontiguousarray(mat[sub]), radius)
        cand_i.append(sub[ci]); cand_j.append(sub[cj])
    ci = np.concatenate(cand_i).astype(np.int64)
    cj = np.concatenate(cand_j).astype(np.int64)
    if len(ci):
        key = np.unique(np.minimum(ci, cj) * n + np.maximum(ci, cj))
        ci, cj = key // n, key % n
        # pairs already joined as exact duplicates need no verification
        same = np.array([meta[a]["sha256"] is not None and meta[a]["sha256"] == meta[b]["sha256"]
                         for a, b in zip(ci, cj)], dtype=bool)
        ci, cj = ci[~same], cj[~same]
    n_cand = int(len(ci))
    if n_cand and p.verify:
        ok = verify_pairs(ci, cj, meta, p)
    else:
        ok = np.ones(n_cand, dtype=bool)
    for a, b in zip(ci[ok], cj[ok]):
        uf.union(int(a), int(b))
    logger.info("dedup: %d images, %d near candidates, %d verified", n, n_cand, int(ok.sum()))

    # 4. assemble groups
    groups = [g for g in uf.groups().values() if len(g) > 1]
    groups.sort(key=lambda g: (-len(g), meta[g[0]]["image_id"]))
    out = []
    for gid, members in enumerate(groups, start=1):
        shas = {meta[k]["sha256"] for k in members}
        kind = "exact" if len(shas) == 1 and None not in shas else "near"
        sims = _member_similarity(members, meta, mat, has_vec, hashes)
        images = []
        for k, s in zip(members, sims):
            m = meta[k]
            images.append({
                "image_id": m["image_id"],
                "file_path": m["file_path"],
                "thumbnail_url": f"/api/v1/thumbnails/{m['image_id']}",
                "file_size": m["file_size"],
                "width": m["width"],
                "height": m["height"],
                "similarity": round(float(s), 4),
            })
        images.sort(key=lambda x: (-x["similarity"], x["image_id"]))
        out.append({"group_id": gid, "kind": kind, "images": images})
    return {"groups": out, "n_groups": len(out), "n_images": sum(len(g["images"]) for g in out)}


def _member_similarity(members: list[int], meta: list[dict], mat, has_vec, hashes) -> list[float]:
    """Max similarity of each member to another member of its group.

    1.0 for an exact (same SHA-256) twin, else embedding cosine, else
    ``1 - Hamming/64`` of the pHashes when no vectors exist.
    """
    m = np.asarray(members)
    k = len(m)
    sim = np.zeros((k, k))
    if mat is not None:
        sim = np.clip(mat[m] @ mat[m].T, -1.0, 1.0).astype(np.float64)
        nov = ~has_vec[m]
    else:
        nov = np.ones(k, dtype=bool)
    if nov.any():
        h = hashes[m]
        ham = 1.0 - _popcount(h[:, None] ^ h[None, :]).astype(np.float64) / HASH_BITS
        either = nov[:, None] | nov[None, :]
        sim = np.where(either, ham, sim)
    shas = [meta[i]["sha256"] for i in members]
    same = np.array([[a is not None and a == b for b in shas] for a in shas])
    sim = np.where(same, 1.0, sim)
    np.fill_diagonal(sim, -1.0)
    return np.clip(sim.max(axis=1), 0.0, 1.0).tolist()

"""Pixel-level verification of near-duplicate candidate pairs.

A candidate pair (from pHash or CLIP cosine) is accepted when

1. the mean palette chromaticity ``rgb / (r+g+b)`` of both images is close
   (brightness-invariant colour check), and
2. their thumbnails correlate after aligning a possible crop:

   * coarse — 12×12 RGB, per-channel mean removed, L2-normalised; the full
     view of one image against 9 symmetric crops (0 / 4.5 / 9 % per axis) of
     the other, both directions; best score ≥ ``min_coarse_corr``;
   * fine — 64×64 RGB at the best coarse alignment, refined by a two-level
     3×3 search (±2 %, then ±1 %);
     best score ≥ ``min_pixel_corr``.

Correlation of mean-removed pixels is invariant to brightness / contrast
scaling and robust to re-encoding and resizing, while text, layout and
colour differences between *different* images of the same template (slides
of one deck, screenshots of one app) push it down. Nearly flat images
(std < 2 grey levels) are compared by mean colour instead.

Signatures come from the 256-px thumbnails written by the scanner (falls
back to decoding the original), are computed only for images that appear in
a candidate pair, and are loaded on a thread pool.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from ..config import get_settings

if TYPE_CHECKING:  # pragma: no cover
    from .service import DedupParams

logger = logging.getLogger(__name__)

COARSE_PX = 12
COARSE_CROPS = (0.0, 0.045, 0.09)
FINE_PX = 64
FINE_STEP = 0.02
BASE_PX = 96          # thumbnails are reduced to this before cropping
FLAT_STD = 2.0


def _load_base(meta_row: dict) -> Image.Image | None:
    thumb = get_settings().THUMBNAILS_DIR / f"{meta_row['image_id']}.jpg"
    try:
        if thumb.exists():
            img = Image.open(thumb)
            img.draft("RGB", (BASE_PX, BASE_PX))
        else:
            from ..indexer.worker import load_image
            img = load_image(meta_row["file_path"], 256)
        img = img.convert("RGB")
        img.thumbnail((BASE_PX, BASE_PX), Image.BILINEAR)
        return img
    except Exception as exc:
        logger.debug("dedup: cannot load %s: %s", meta_row["file_path"], exc)
        return None


def _vec(img: Image.Image, cx: float, cy: float, px: int) -> tuple[np.ndarray, float]:
    """Mean-removed, L2-normalised RGB vector of the symmetric crop (cx, cy)."""
    w, h = img.size
    box = (w * cx, h * cy, w * (1 - cx), h * (1 - cy))
    a = np.asarray(img.resize((px, px), Image.BILINEAR, box=box), dtype=np.float32).reshape(-1, 3)
    a = a - a.mean(axis=0)
    std = float(np.sqrt((a * a).mean()))
    n = np.linalg.norm(a)
    v = (a / n).ravel() if n > 1e-6 else np.zeros(a.size, np.float32)
    return v, std


def _coarse(img: Image.Image) -> tuple[np.ndarray, float]:
    vs, std0 = [], 0.0
    for cy in COARSE_CROPS:
        for cx in COARSE_CROPS:
            v, s = _vec(img, cx, cy, COARSE_PX)
            if cx == 0 and cy == 0:
                std0 = s
            vs.append(v)
    return np.stack(vs), std0


# Pair-score cache: (db, id_a, sha_a, id_b, sha_b) → (coarse, fine | None,
# flat_color_dist | None). Scores do not depend on DedupParams, so repeated
# /duplicates calls (UI refresh, threshold slider) only verify new pairs.
_PAIR_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_PAIR_CACHE_MAX = 300_000
_cache_lock = threading.Lock()
_NOT_FLAT = -1.0


def clear_cache() -> None:
    with _cache_lock:
        _PAIR_CACHE.clear()


def _fine_score(base_c: Image.Image, base_f: Image.Image, cx0: float, cy0: float) -> float:
    """Best 64-px correlation around the coarse alignment (2-level 3×3 search)."""
    ref, _ = _vec(base_f, 0.0, 0.0, FINE_PX)
    best, bx, by = -1.0, cx0, cy0
    seen = set()
    for step in (FINE_STEP, FINE_STEP / 2):
        cx_c, cy_c = bx, by
        for dy in (-step, 0.0, step):
            for dx in (-step, 0.0, step):
                cx, cy = round(max(0.0, cx_c + dx), 4), round(max(0.0, cy_c + dy), 4)
                if (cx, cy) in seen:
                    continue
                seen.add((cx, cy))
                v, _ = _vec(base_c, cx, cy, FINE_PX)
                score = float(v @ ref)
                if score > best:
                    best, bx, by = score, cx, cy
    return best


def verify_pairs(ci: np.ndarray, cj: np.ndarray, meta: list[dict], p: "DedupParams") -> np.ndarray:
    """Boolean mask over candidate pairs ``(ci[k], cj[k])`` (indices into meta)."""
    n_pairs = len(ci)
    ok = np.zeros(n_pairs, dtype=bool)
    if n_pairs == 0:
        return ok
    t0 = time.perf_counter()

    # 1. chromaticity of the mean palette colour (cheap, vectorised)
    col = np.array([[m.get("mr"), m.get("mg"), m.get("mb")] for m in meta], dtype=object)
    has_col = np.array([c[0] is not None for c in col])
    colf = np.where(has_col[:, None], col, 0).astype(np.float64)
    chroma = colf / np.maximum(colf.sum(axis=1, keepdims=True), 1.0)
    cdist = np.abs(chroma[ci] - chroma[cj]).sum(axis=1)
    alive = ~(has_col[ci] & has_col[cj]) | (cdist <= p.max_chroma_dist)
    idx = np.flatnonzero(alive)
    if len(idx) == 0:
        return ok

    db = str(get_settings().DB_PATH)
    keys = {t: (db, meta[ci[t]]["image_id"], meta[ci[t]]["sha256"], meta[cj[t]]["image_id"], meta[cj[t]]["sha256"])
            for t in idx}
    with _cache_lock:
        cached = {t: _PAIR_CACHE.get(k) for t, k in keys.items()}

    def needs_work(t) -> bool:
        c = cached[t]
        return c is None or (c[2] == _NOT_FLAT and c[1] is None and c[0] >= p.min_coarse_corr)

    todo = [t for t in idx if needs_work(t)]

    # 2. signatures for images in pairs not answered by the cache
    workers = max(1, os.cpu_count() or 1)
    involved = sorted({int(ci[t]) for t in todo} | {int(cj[t]) for t in todo})
    bases: dict[int, Image.Image | None] = {}
    sig: dict[int, np.ndarray] = {}
    std: dict[int, float] = {}
    if involved:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            bases = dict(zip(involved, pool.map(lambda k: _load_base(meta[k]), involved)))
            good = [k for k in involved if bases[k] is not None]
            for k, (sg, sd) in zip(good, pool.map(lambda k: _coarse(bases[k]), good, chunksize=64)):
                sig[k], std[k] = sg, sd
    t1 = time.perf_counter()

    # 3. coarse alignment search (+ flat-image shortcut)
    fine_jobs = []
    new: dict[int, tuple] = {}
    for t in todo:
        a, b = int(ci[t]), int(cj[t])
        if a not in sig or b not in sig:
            new[t] = (1.0, 1.0, _NOT_FLAT)  # cannot verify (no thumbnail) → trust candidate
            continue
        if std[a] < FLAT_STD or std[b] < FLAT_STD:
            both = std[a] < FLAT_STD and std[b] < FLAT_STD and has_col[a] and has_col[b]
            new[t] = (0.0, None, float(np.linalg.norm(colf[a] - colf[b])) if both else 1e9)
            continue
        s_ab = sig[b] @ sig[a][0]  # crops of b vs full a
        s_ba = sig[a] @ sig[b][0]  # crops of a vs full b
        if s_ab.max() >= s_ba.max():
            best, cropped, full, k = float(s_ab.max()), b, a, int(s_ab.argmax())
        else:
            best, cropped, full, k = float(s_ba.max()), a, b, int(s_ba.argmax())
        new[t] = (best, None, _NOT_FLAT)
        if best >= p.min_coarse_corr:
            n = len(COARSE_CROPS)
            fine_jobs.append((t, cropped, full, COARSE_CROPS[k % n], COARSE_CROPS[k // n]))
    t2 = time.perf_counter()

    # 4. fine check at the best alignment
    if fine_jobs:
        def fine(job):
            t, cropped, full, cx0, cy0 = job
            return t, _fine_score(bases[cropped], bases[full], cx0, cy0)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for t, score in pool.map(fine, fine_jobs, chunksize=32):
                new[t] = (new[t][0], score, _NOT_FLAT)
    t3 = time.perf_counter()

    with _cache_lock:
        for t, v in new.items():
            _PAIR_CACHE[keys[t]] = v
            cached[t] = v
        while len(_PAIR_CACHE) > _PAIR_CACHE_MAX:
            _PAIR_CACHE.popitem(last=False)

    for t in idx:
        coarse, fine_s, flat = cached[t]
        if flat != _NOT_FLAT:
            ok[t] = flat <= p.max_flat_color_dist
        else:
            ok[t] = coarse >= p.min_coarse_corr and fine_s is not None and fine_s >= p.min_pixel_corr
    logger.info("dedup verify: %d candidates, %d after colour, %d uncached, %d fine checks, %d accepted "
                "(signatures %.2fs, coarse %.2fs, fine %.2fs)", n_pairs, len(idx), len(todo), len(fine_jobs),
                int(ok.sum()), t1 - t0, t2 - t1, t3 - t2)
    return ok

"""Colour-palette search in CIE Lab space.

Every indexed image stores up to five KMeans swatches (``image_colors``) with
their pixel share (``weight``). A query is one or more swatches, each with a
relative importance. For every query swatch we look for the best matching
image swatch, where a match is good when the colours are perceptually close
(small CIE76 ΔE) *and* the image swatch covers a meaningful part of the image:

    sim(q, p)   = max(0, 1 - ΔE(q, p) / DELTA_E_MAX)
    cover(p)    = sqrt(min(1, weight_p / FULL_COVERAGE))
    match(q)    = max_p sim(q, p) · cover(p)
    score(img)  = Σ_q w_q · match(q) / Σ_q w_q          ∈ [0, 1]

All images are scored at once with numpy over palettes loaded by one query.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ..db.schema import get_db

DELTA_E_MAX = 60.0  # ΔE at which two colours no longer count as similar
FULL_COVERAGE = 0.3  # a swatch covering ≥ 30 % of the image counts fully
MAX_SWATCHES = 5

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")


@dataclass(frozen=True)
class QuerySwatch:
    """A query colour and its relative importance."""

    rgb: tuple[int, int, int]
    weight: float = 1.0


def hex_to_rgb(hex_code: str) -> tuple[int, int, int]:
    """Parse ``#RRGGBB`` (leading ``#`` optional); raises ``ValueError``."""
    m = _HEX_RE.match(hex_code.strip())
    if not m:
        raise ValueError(f"invalid colour {hex_code!r}; expected #RRGGBB")
    v = int(m.group(1), 16)
    return (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB values in [0, 255] (shape ``(..., 3)``) to CIE Lab (D65)."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]]
    )
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


@dataclass(frozen=True)
class Palettes:
    """Palettes of many images as padded arrays."""

    ids: np.ndarray  # (N,) int64
    lab: np.ndarray  # (N, MAX_SWATCHES, 3) float64
    weight: np.ndarray  # (N, MAX_SWATCHES) float64, 0 for missing swatches


def load_palettes(image_ids: np.ndarray | list[int] | None = None) -> Palettes:
    """Load palettes of COMPLETED images (optionally only ``image_ids``) in one query."""
    sql = """SELECT c.image_id, c.rank, c.r, c.g, c.b, c.weight
             FROM image_colors c JOIN images i ON i.image_id = c.image_id
             WHERE i.status = 'COMPLETED'"""
    params: list[int] = []
    if image_ids is not None:
        params = [int(i) for i in image_ids]
        if not params:
            return Palettes(np.empty(0, np.int64), np.empty((0, MAX_SWATCHES, 3)), np.empty((0, MAX_SWATCHES)))
        sql += f" AND c.image_id IN ({','.join('?' * len(params))})"
    with get_db() as conn:
        rows = np.array(
            [(r["image_id"], r["rank"], r["r"], r["g"], r["b"], r["weight"]) for r in conn.execute(sql, params)],
            dtype=np.float64,
        ).reshape(-1, 6)
    ids, inverse = np.unique(rows[:, 0].astype(np.int64), return_inverse=True)
    slot = np.clip(rows[:, 1].astype(np.int64) - 1, 0, MAX_SWATCHES - 1)
    rgb = np.zeros((len(ids), MAX_SWATCHES, 3))
    weight = np.zeros((len(ids), MAX_SWATCHES))
    rgb[inverse, slot] = rows[:, 2:5]
    weight[inverse, slot] = rows[:, 5]
    return Palettes(ids, rgb_to_lab(rgb), weight)


def score_palettes(query: list[QuerySwatch], palettes: Palettes) -> np.ndarray:
    """Return a colour-match score in [0, 1] for every image in ``palettes``."""
    if not query or len(palettes.ids) == 0:
        return np.zeros(len(palettes.ids), dtype=np.float64)
    q_lab = rgb_to_lab(np.array([s.rgb for s in query], dtype=np.float64))  # (Q, 3)
    q_w = np.array([max(s.weight, 0.0) for s in query], dtype=np.float64)
    if q_w.sum() <= 0:
        q_w = np.ones_like(q_w)
    # ΔE76 between every query swatch and every image swatch: (N, Q, S)
    delta = np.linalg.norm(palettes.lab[:, None, :, :] - q_lab[None, :, None, :], axis=-1)
    sim = np.clip(1.0 - delta / DELTA_E_MAX, 0.0, 1.0)
    cover = np.sqrt(np.minimum(1.0, palettes.weight / FULL_COVERAGE))[:, None, :]
    match = (sim * cover).max(axis=2)  # (N, Q)
    return np.clip(match @ q_w / q_w.sum(), 0.0, 1.0)

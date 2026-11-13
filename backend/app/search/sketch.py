"""Sketch-pad queries: decode, normalise and embed a hand-drawn PNG.

Sketches are thin strokes on a mostly empty canvas, which CLIP sees very
differently from photos. To help it we composite onto white, crop to the
strokes' bounding box (plus a margin) padded to a square, and average the
embedding of the sketch with that of a thickened, slightly blurred copy.
The stroke colours are also extracted so the colour scorer can use them as
a weak hint.
"""

from __future__ import annotations

import base64
import binascii
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter, UnidentifiedImageError

from ..embeddings import get_encoder
from ..embeddings.base import l2_normalize
from .colors import QuerySwatch

WHITE_THRESHOLD = 235  # pixels with every channel above this are background
MARGIN = 0.1  # crop margin as a fraction of the stroke bbox's longer side
MAX_STROKE_COLORS = 3
MIN_COLOR_SHARE = 0.05  # ignore stroke colours covering < 5 % of stroke pixels
_QUANT = 32  # colour quantisation step for stroke colour histogram


@dataclass(frozen=True)
class SketchQuery:
    """A prepared sketch: its embedding and dominant stroke colours."""

    vector: np.ndarray  # (dim,) float32, L2-normalised
    colors: list[QuerySwatch]


def decode_image(data: str) -> Image.Image:
    """Decode a ``data:image/...;base64,`` URL or bare base64 string."""
    payload = data.split(",", 1)[1] if data.lstrip().startswith("data:") else data
    try:
        raw = base64.b64decode(payload.strip(), validate=False)
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (binascii.Error, UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("sketch_png is not a valid base64-encoded image") from exc
    return img


def to_rgb_on_white(img: Image.Image) -> Image.Image:
    """Flatten transparency onto a white background."""
    rgba = img.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(canvas, rgba).convert("RGB")


def crop_to_strokes(img: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """Crop to the stroke bbox with a margin, padded to a white square.

    Returns the cropped image and the RGB values of all stroke pixels.
    Raises ``ValueError`` for a blank canvas.
    """
    arr = np.asarray(img)
    mask = (arr < WHITE_THRESHOLD).any(axis=2)
    if not mask.any():
        raise ValueError("sketch is empty")
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = int(max(x1 - x0, y1 - y0) * (1 + 2 * MARGIN)) + 1
    square = Image.new("RGB", (side, side), "white")
    square.paste(img.crop((x0, y0, x1, y1)), ((side - (x1 - x0)) // 2, (side - (y1 - y0)) // 2))
    return square, arr[mask]


def stroke_colors(pixels: np.ndarray) -> list[QuerySwatch]:
    """Up to :data:`MAX_STROKE_COLORS` dominant stroke colours, weighted by share."""
    q = (pixels.astype(np.int64) // _QUANT).reshape(-1, 3)
    keys = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
    uniq, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    total = counts.sum()
    out: list[QuerySwatch] = []
    for idx in np.argsort(-counts)[:MAX_STROKE_COLORS]:
        share = counts[idx] / total
        if share < MIN_COLOR_SHARE:
            break
        mean = pixels[inverse == idx].mean(axis=0)
        out.append(QuerySwatch(rgb=tuple(int(round(c)) for c in mean), weight=float(share)))
    return out


def prepare_sketch(data: str) -> SketchQuery:
    """Decode, normalise and embed a sketch (data URL or base64 PNG)."""
    img = to_rgb_on_white(decode_image(data))
    cropped, pixels = crop_to_strokes(img)
    thick = cropped.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.GaussianBlur(1))
    vecs = get_encoder().encode_images([cropped, thick])
    vector = l2_normalize(vecs.mean(axis=0))[0]
    return SketchQuery(vector=vector, colors=stroke_colors(pixels))

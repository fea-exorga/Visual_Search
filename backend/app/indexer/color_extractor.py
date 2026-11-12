"""Dominant color palette extraction using KMeans clustering."""

import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

from ..config import get_settings


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB values to #RRGGBB hex string."""
    return f"#{r:02X}{g:02X}{b:02X}"


def extract_color_palette(image: Image.Image, n_colors: int = 5) -> list[dict]:
    """
    Extract dominant color palette from an image using MiniBatchKMeans.

    Args:
        image: PIL Image to analyze.
        n_colors: Number of dominant colors to extract.

    Returns:
        List of dicts with keys: rank, hex_code, r, g, b, weight.
    """
    # Resize to 64x64 for fast clustering
    resized = image.convert("RGB").resize((64, 64), Image.LANCZOS)

    # Reshape to (4096, 3) pixel array
    pixels = np.array(resized, dtype=np.float64).reshape(-1, 3)

    # Fit MiniBatchKMeans
    kmeans = MiniBatchKMeans(
        n_clusters=n_colors,
        n_init="auto",
        random_state=42,
        batch_size=256,
    )
    kmeans.fit(pixels)

    # Cluster centers and labels
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    total_pixels = len(labels)

    # Compute weight per cluster
    cluster_info = []
    for idx in range(n_colors):
        count = int(np.sum(labels == idx))
        weight = count / total_pixels
        r, g, b = int(round(centers[idx][0])), int(round(centers[idx][1])), int(round(centers[idx][2]))
        cluster_info.append({
            "r": r,
            "g": g,
            "b": b,
            "hex_code": rgb_to_hex(r, g, b),
            "weight": round(weight, 6),
        })

    # Sort by weight descending
    cluster_info.sort(key=lambda c: c["weight"], reverse=True)

    # Assign ranks 1..n_colors
    palette = []
    for rank, info in enumerate(cluster_info, start=1):
        palette.append({
            "rank": rank,
            "hex_code": info["hex_code"],
            "r": info["r"],
            "g": info["g"],
            "b": info["b"],
            "weight": info["weight"],
        })

    return palette


def save_color_palette(image_id: int, palette: list[dict]) -> None:
    """
    Persist a color palette to the database and mark the image as COLOR_DONE.

    Args:
        image_id: Primary key of the image row.
        palette: Output of extract_color_palette().
    """
    from ..db.schema import get_db

    with get_db() as db:
        for color in palette:
            db.execute(
                """
                INSERT OR REPLACE INTO image_colors
                    (image_id, rank, hex_code, r, g, b, weight)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    image_id,
                    color["rank"],
                    color["hex_code"],
                    color["r"],
                    color["g"],
                    color["b"],
                    color["weight"],
                ),
            )

        db.execute(
            "UPDATE images SET status = 'COLOR_DONE', updated_at = CURRENT_TIMESTAMP WHERE image_id = ?",
            (image_id,),
        )
        db.commit()

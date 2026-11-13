"""Milestone 3 — manifold projection (UMAP / t-SNE / PCA), theme clustering
and zero-shot cluster labelling for the galaxy explorer.

Public API::

    from app.projection import compute_galaxy, get_galaxy, project, cluster_points
"""

from __future__ import annotations

from .cluster import PALETTE, ClusterResult, cluster_points
from .reduce import METHODS, project
from .service import GalaxyError, compute_galaxy, get_galaxy

__all__ = [
    "METHODS",
    "PALETTE",
    "ClusterResult",
    "GalaxyError",
    "cluster_points",
    "compute_galaxy",
    "get_galaxy",
    "project",
]

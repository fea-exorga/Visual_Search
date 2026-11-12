"""Batched CLIP/SigLIP vision embedding generator."""

import logging
import os

import numpy as np
import torch
from PIL import Image

from ..config import get_settings
from ..db.schema import get_db

logger = logging.getLogger(__name__)


class VisionEmbedder:
    """Generate L2-normalised vision embeddings using SigLIP / CLIP models."""

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or self.get_device()
        self.embedding_dim: int = get_settings().EMBEDDING_DIM
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------
    # Device detection
    # ------------------------------------------------------------------

    @staticmethod
    def get_device() -> str:
        """Return the best available device: cuda > mps > cpu."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load model and processor if not yet loaded."""
        if self._model is not None:
            return

        from transformers import AutoModel, AutoProcessor

        logger.info(
            "Loading vision model %s on %s …", self.model_name, self.device
        )
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()
        self._model.to(self.device)

        # Infer embedding dim from model config if available
        if hasattr(self._model.config, "hidden_size"):
            self.embedding_dim = self._model.config.hidden_size

        logger.info(
            "Model loaded — embedding_dim=%d, device=%s",
            self.embedding_dim,
            self.device,
        )

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec, axis=-1, keepdims=True)
        norm = np.where(norm == 0, 1, norm)
        return (vec / norm).astype(np.float32)

    def embed_single(self, image: Image.Image) -> np.ndarray:
        """
        Embed a single PIL Image.

        Returns:
            L2-normalised float32 numpy array of shape (dim,).
        """
        self._ensure_loaded()

        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self._model(**inputs)

        # Use pooler_output if available, otherwise mean-pool last_hidden_state
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            vec = outputs.pooler_output[0].cpu().numpy()
        else:
            vec = outputs.last_hidden_state.mean(dim=1)[0].cpu().numpy()

        return self._l2_normalize(vec)

    def embed_batch(self, images: list[Image.Image]) -> np.ndarray:
        """
        Embed a batch of PIL Images.

        Returns:
            L2-normalised float32 numpy array of shape (N, dim).
        """
        if not images:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        self._ensure_loaded()

        inputs = self._processor(
            images=images, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self._model(**inputs)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            vecs = outputs.pooler_output.cpu().numpy()
        else:
            vecs = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        return self._l2_normalize(vecs)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_embeddings(
        self, image_ids: list[int], embeddings: np.ndarray
    ) -> None:
        """
        Append embeddings to the memory-mapped .npy store and update the DB.

        Args:
            image_ids: List of image primary keys (len N).
            embeddings: (N, dim) float32 array of L2-normalised vectors.
        """
        settings = get_settings()
        emb_path = settings.EMBEDDINGS_PATH

        # Load or create the embedding matrix
        if os.path.exists(emb_path):
            existing = np.load(emb_path, mmap_mode="r")
            start_row = existing.shape[0]
            new_matrix = np.vstack([np.array(existing), embeddings])
        else:
            start_row = 0
            new_matrix = embeddings

        np.save(emb_path, new_matrix)

        # Update DB rows
        with get_db() as db:
            for offset, image_id in enumerate(image_ids):
                row_id = start_row + offset
                db.execute(
                    """
                    UPDATE images
                    SET embedding_row_id = ?,
                        status = 'EMBEDDING_DONE',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE image_id = ?
                    """,
                    (row_id, image_id),
                )
            db.commit()

        logger.info(
            "Saved %d embeddings (rows %d–%d) to %s",
            len(image_ids),
            start_row,
            start_row + len(image_ids) - 1,
            emb_path,
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_embedder: VisionEmbedder | None = None


def get_embedder() -> VisionEmbedder:
    """Return the module-level VisionEmbedder singleton."""
    global _embedder
    if _embedder is None:
        _embedder = VisionEmbedder()
    return _embedder

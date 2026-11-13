"""Optional transformers backend (CLIP or SigLIP) — requires ``torch``."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .base import l2_normalize


class TorchHFEncoder:
    def __init__(self, model_name: str = "google/siglip-base-patch16-224") -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).eval().to(self.device)
        cfg = self._model.config
        self.dim = int(getattr(cfg, "projection_dim", None) or cfg.text_config.hidden_size)
        self.name = f"{model_name.split('/')[-1]}/torch"

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, self.dim), dtype=np.float32)
        inputs = self._processor(images=[i.convert("RGB") for i in images], return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            feats = self._model.get_image_features(**inputs)
        return l2_normalize(feats.float().cpu().numpy())

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        inputs = self._processor(text=list(texts), return_tensors="pt", padding="max_length", truncation=True).to(self.device)
        with self._torch.inference_mode():
            feats = self._model.get_text_features(**inputs)
        return l2_normalize(feats.float().cpu().numpy())

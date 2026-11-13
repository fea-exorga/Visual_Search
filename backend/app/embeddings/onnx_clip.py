"""Torch-free CLIP encoder running on ONNX Runtime.

Loads separately exported vision / text towers (``image_embeds`` /
``text_embeds`` outputs, already projected into the shared space) in
fp32, fp16 or int8 precision.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .base import l2_normalize
from .download import PRECISION_FILES, ensure_model

logger = logging.getLogger(__name__)

_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


class OnnxClipEncoder:
    def __init__(
        self,
        precision: str = "fp32",
        repo_id: str | None = None,
        num_threads: int | None = None,
        model_path: Path | None = None,
    ) -> None:
        import onnxruntime as ort

        self.precision = precision
        mdir = Path(model_path) if model_path else ensure_model(precision, repo_id)
        vision_file, text_file = PRECISION_FILES[precision]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = num_threads or max(1, os.cpu_count() or 1)
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self._vision = ort.InferenceSession(str(mdir / vision_file), opts, providers=providers)
        self._text = ort.InferenceSession(str(mdir / text_file), opts, providers=providers)

        self._vision_in = self._vision.get_inputs()[0]
        self._vision_dtype = np.float16 if "float16" in self._vision_in.type else np.float32
        self._text_inputs = {i.name for i in self._text.get_inputs()}

        cfg = json.loads((mdir / "preprocessor_config.json").read_text())
        size = cfg.get("crop_size", 224)
        self.image_size = size["height"] if isinstance(size, dict) else int(size)

        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(str(mdir / "tokenizer.json"))
        self._tok.enable_truncation(max_length=77)
        pad_id = self._tok.token_to_id("<|endoftext|>")
        self._tok.enable_padding(pad_id=pad_id, pad_token="<|endoftext|>")

        model_cfg = json.loads((mdir / "config.json").read_text())
        self.dim = int(model_cfg.get("projection_dim", 512))
        base = (repo_id or mdir.name).split("__")[-1].split("/")[-1]
        self.name = f"{base}/onnx-{precision}"
        logger.info("ONNX CLIP ready: %s (dim=%d, providers=%s)", self.name, self.dim, providers)

    # ── preprocessing ───────────────────────────────────────────────────
    def preprocess(self, img: Image.Image) -> np.ndarray:
        """Resize shortest side → crop centre → normalise. Returns (3, H, W)."""
        s = self.image_size
        img = img.convert("RGB")
        w, h = img.size
        scale = s / min(w, h)
        nw, nh = max(s, round(w * scale)), max(s, round(h * scale))
        img = img.resize((nw, nh), Image.BICUBIC)
        left, top = (nw - s) // 2, (nh - s) // 2
        img = img.crop((left, top, left + s, top + s))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - _CLIP_MEAN) / _CLIP_STD
        return arr.transpose(2, 0, 1)

    # ── public API ──────────────────────────────────────────────────────
    def encode_pixel_batch(self, pixels: np.ndarray) -> np.ndarray:
        """Encode an already-preprocessed (N, 3, H, W) batch."""
        out = self._vision.run(["image_embeds"], {self._vision_in.name: pixels.astype(self._vision_dtype)})[0]
        return l2_normalize(out.astype(np.float32))

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, self.dim), dtype=np.float32)
        return self.encode_pixel_batch(np.stack([self.preprocess(im) for im in images]))

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        enc = self._tok.encode_batch(list(texts))
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        feeds = {"input_ids": ids}
        if "attention_mask" in self._text_inputs:
            feeds["attention_mask"] = np.array([e.attention_mask for e in enc], dtype=np.int64)
        out = self._text.run(["text_embeds"], feeds)[0]
        return l2_normalize(out.astype(np.float32))

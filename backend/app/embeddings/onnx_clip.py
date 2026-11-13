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
# (x/255 - mean)/std == x*_SCALE - _OFFSET
_SCALE = (1.0 / (255.0 * _CLIP_STD)).astype(np.float32)
_OFFSET = (_CLIP_MEAN / _CLIP_STD).astype(np.float32)


class OnnxClipEncoder:
    def __init__(
        self,
        precision: str = "fp32",
        repo_id: str | None = None,
        num_threads: int | None = None,
        model_path: Path | None = None,
        inter_op_threads: int | None = None,
        execution_mode: str | None = None,
        allow_spinning: bool | None = None,
    ) -> None:
        import onnxruntime as ort

        self.precision = precision
        mdir = Path(model_path) if model_path else ensure_model(precision, repo_id)
        vision_file, text_file = PRECISION_FILES[precision]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        env_threads = int(os.environ.get("VS_ONNX_THREADS", "0") or 0)
        self.num_threads = num_threads or env_threads or max(1, os.cpu_count() or 1)
        opts.intra_op_num_threads = self.num_threads
        inter = inter_op_threads or int(os.environ.get("VS_ONNX_INTER_THREADS", "0") or 0)
        if inter:
            opts.inter_op_num_threads = inter
        mode = (execution_mode or os.environ.get("VS_ONNX_EXEC_MODE", "sequential")).lower()
        opts.execution_mode = (ort.ExecutionMode.ORT_PARALLEL if mode == "parallel"
                               else ort.ExecutionMode.ORT_SEQUENTIAL)
        if allow_spinning is None:
            env_spin = os.environ.get("VS_ONNX_SPIN")
            allow_spinning = None if env_spin is None else env_spin not in ("0", "false", "no")
        if allow_spinning is not None:
            # Spinning intra-op threads burn CPU between ops; turning it off
            # leaves cores to the indexer's decode/OCR thread pool.
            opts.add_session_config_entry("session.intra_op.allow_spinning", "1" if allow_spinning else "0")
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
    def _resize_crop(self, img: Image.Image) -> Image.Image:
        """Shortest side → ``image_size`` (bicubic) then centre crop.

        Large inputs are first shrunk cheaply: JPEGs that are not decoded yet
        use ``draft`` (DCT-domain scaling), everything else ``reduce`` by an
        integer factor via ``reducing_gap`` before the final bicubic pass.
        """
        s = self.image_size
        if img.format == "JPEG" and img.tile:  # not decoded yet
            try:
                img.draft("RGB", (s * 2, s * 2))
            except Exception:
                pass
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        scale = s / min(w, h)
        nw, nh = max(s, round(w * scale)), max(s, round(h * scale))
        if (nw, nh) != (w, h):
            img = img.resize((nw, nh), Image.BICUBIC, reducing_gap=3.0 if scale < 0.5 else None)
        left, top = (nw - s) // 2, (nh - s) // 2
        if (left, top, nw, nh) != (0, 0, s, s):
            img = img.crop((left, top, left + s, top + s))
        return img

    def preprocess_uint8(self, img: Image.Image) -> np.ndarray:
        """(H, W, 3) uint8 crop — cheap to stack, normalised per batch."""
        return np.asarray(self._resize_crop(img), dtype=np.uint8)

    def normalize_batch(self, batch_u8: np.ndarray) -> np.ndarray:
        """(N, H, W, 3) uint8 → (N, 3, H, W) model-dtype, CLIP-normalised."""
        x = batch_u8.astype(np.float32)
        x *= _SCALE
        x -= _OFFSET
        return np.ascontiguousarray(x.transpose(0, 3, 1, 2), dtype=self._vision_dtype)

    def preprocess(self, img: Image.Image) -> np.ndarray:
        """Resize shortest side → crop centre → normalise. Returns (3, H, W)."""
        arr = np.asarray(self._resize_crop(img), dtype=np.float32)
        arr = arr * _SCALE - _OFFSET
        return arr.transpose(2, 0, 1)

    # ── public API ──────────────────────────────────────────────────────
    def encode_pixel_batch(self, pixels: np.ndarray) -> np.ndarray:
        """Encode an already-preprocessed (N, 3, H, W) batch."""
        if pixels.dtype != self._vision_dtype:
            pixels = pixels.astype(self._vision_dtype)
        out = self._vision.run(["image_embeds"], {self._vision_in.name: pixels})[0]
        return l2_normalize(out.astype(np.float32))

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, self.dim), dtype=np.float32)
        batch = np.stack([self.preprocess_uint8(im) for im in images])
        return self.encode_pixel_batch(self.normalize_batch(batch))

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

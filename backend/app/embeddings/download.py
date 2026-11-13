"""Fetch pre-exported ONNX CLIP weights (fp32 / fp16 / int8) from the HF Hub.

Usage::

    python -m app.embeddings.download            # default model, all precisions
    python -m app.embeddings.download --precision fp32
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from ..config import get_settings

logger = logging.getLogger(__name__)

# precision → (vision file, text file) inside the repo's onnx/ folder
PRECISION_FILES = {
    "fp32": ("vision_model.onnx", "text_model.onnx"),
    "fp16": ("vision_model_fp16.onnx", "text_model_fp16.onnx"),
    "int8": ("vision_model_quantized.onnx", "text_model_quantized.onnx"),
}
AUX_FILES = ["tokenizer.json", "preprocessor_config.json", "config.json"]


def model_dir(repo_id: str | None = None) -> Path:
    s = get_settings()
    repo_id = repo_id or s.ONNX_REPO_ID
    return s.MODELS_DIR / repo_id.replace("/", "__")


def ensure_model(precision: str | None = None, repo_id: str | None = None) -> Path:
    """Download (if missing) the files for one precision and return the model dir."""
    from huggingface_hub import hf_hub_download

    s = get_settings()
    precision = precision or s.ONNX_PRECISION
    repo_id = repo_id or s.ONNX_REPO_ID
    target = model_dir(repo_id)
    target.mkdir(parents=True, exist_ok=True)

    wanted = list(AUX_FILES) + [f"onnx/{f}" for f in PRECISION_FILES[precision]]
    for rel in wanted:
        dest = target / Path(rel).name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        logger.info("Downloading %s/%s …", repo_id, rel)
        src = hf_hub_download(repo_id=repo_id, filename=rel)
        shutil.copyfile(src, dest)
    return target


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=[*PRECISION_FILES, "all"], default="all")
    ap.add_argument("--repo-id", default=None)
    args = ap.parse_args()
    precisions = list(PRECISION_FILES) if args.precision == "all" else [args.precision]
    for p in precisions:
        path = ensure_model(p, args.repo_id)
    print(path)


if __name__ == "__main__":
    main()

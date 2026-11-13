"""Vision-language encoders (CLIP / SigLIP) exposed through a common interface."""

from .base import Encoder
from .registry import get_encoder, reset_encoder

__all__ = ["Encoder", "get_encoder", "reset_encoder"]

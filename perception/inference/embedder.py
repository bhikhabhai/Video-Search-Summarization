"""CLIP ViT-B/32 embeddings (via open_clip) for images and text.

Returns unit-normalised float32 vectors of shape (768,), so cosine similarity
reduces to a dot product everywhere downstream. Model loading is lazy; the
pure :func:`l2_normalise` helper is importable without pulling in torch.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PIL import Image

from perception.config import get_device, get_settings

logger = logging.getLogger(__name__)

EMBED_DIM = 768  # ViT-L/14 output dimension


def l2_normalise(vec: np.ndarray) -> np.ndarray:
    """Project a vector onto the unit sphere (float32).

    A zero vector is returned unchanged (as float32) rather than dividing by
    zero — callers treat it as "matches nothing".
    """
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


class Embedder:
    """Lazy CLIP wrapper producing unit-norm float32 vectors of shape (768,)."""

    def __init__(self) -> None:
        settings = get_settings()
        self.model_name = settings.clip_model
        self.pretrained = settings.clip_pretrained
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import open_clip
        import torch

        device = get_device()
        logger.info("Loading CLIP %s (%s) on %s", self.model_name, self.pretrained, device)
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        model = model.to(device).eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        self._device = device
        self._torch = torch

    def embed_image(self, image: Image.Image) -> np.ndarray:
        return self.embed_images([image])[0]

    def embed_images(self, images: list[Image.Image]) -> list[np.ndarray]:
        """Embed a batch of PIL images. Returns unit-norm (768,) float32 arrays."""
        self._load()
        torch = self._torch
        tensors = torch.stack(
            [self._preprocess(img.convert("RGB")) for img in images]
        ).to(self._device)
        with torch.no_grad():
            features = self._model.encode_image(tensors)
        return [l2_normalise(f.cpu().numpy()) for f in features]

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of strings. Returns unit-norm (768,) float32 arrays."""
        self._load()
        torch = self._torch
        tokens = self._tokenizer(texts).to(self._device)
        with torch.no_grad():
            features = self._model.encode_text(tokens)
        return [l2_normalise(f.cpu().numpy()) for f in features]


if __name__ == "__main__":
    import argparse

    from perception.config import configure_logging

    configure_logging()
    parser = argparse.ArgumentParser(description="Embed an image and a few probe texts")
    parser.add_argument("image", help="path to an image file")
    args = parser.parse_args()

    embedder = Embedder()
    img_vec = embedder.embed_image(Image.open(args.image).convert("RGB"))
    print(f"image embedding: shape={img_vec.shape} dtype={img_vec.dtype} "
          f"norm={np.linalg.norm(img_vec):.4f}")

    probes = ["a person", "a car on a road", "a dog", "an empty room"]
    for text, vec in zip(probes, embedder.embed_texts(probes)):
        print(f"  sim('{text}') = {float(img_vec @ vec):.3f}")

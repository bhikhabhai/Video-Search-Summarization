"""Dedicated OCR for reliable text / number-plate reading.

Runs independently of the VLM captioner, so plate numbers and signage are
extracted by a purpose-built model rather than relying on the caption VLM
(which reads text inconsistently).

Backends (``OCR_BACKEND``):

* ``none``   — disabled; the pipeline keeps whatever OCR the VLM caption found.
* ``paddle`` — local **PaddleOCR** (needs ``paddleocr`` + ``paddlepaddle``,
  imported lazily so the dependency is only required when this backend is on).
* ``ollama`` — send the frame to an Ollama vision model with an OCR-only
  prompt; reuses the caption/main Ollama server unless ``OCR_OLLAMA_BASE_URL``
  is set.

All backends return a list of text strings per frame (deduplicated, order
preserved). Lazy loading: nothing is imported/loaded until the first read.
"""

from __future__ import annotations

import base64
import io
import logging
import re
from typing import Any

import numpy as np
from PIL import Image

from perception.config import get_settings

logger = logging.getLogger(__name__)

_OCR_PROMPT = (
    "Read ALL legible text in this image, including vehicle number plates, "
    "signs, and labels. Return each distinct piece of text on its own line, "
    "exactly as written. If there is no legible text, return nothing."
)

# Plate-ish tokens: keep alphanumerics, drop stray punctuation/whitespace runs.
_CLEAN_RE = re.compile(r"\s+")


class OCRReader:
    """Config-selectable OCR over keyframes (PaddleOCR or Ollama vision)."""

    def __init__(self) -> None:
        settings = get_settings()
        self.backend = settings.ocr_backend.lower()
        self.languages = settings.ocr_languages
        self.min_confidence = settings.ocr_min_confidence
        url = (
            settings.ocr_ollama_base_url
            or settings.ollama_caption_base_url
            or settings.ollama_base_url
        )
        self.ollama_url = url.rstrip("/")
        self.ollama_model = settings.ocr_ollama_model
        self._concurrency = max(1, settings.ollama_concurrency)
        self._paddle: Any = None

    @property
    def enabled(self) -> bool:
        return self.backend in ("paddle", "ollama")

    # ── Lazy loaders ───────────────────────────────────────────

    def _load_paddle(self) -> None:
        if self._paddle is not None:
            return
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "OCR_BACKEND=paddle requires PaddleOCR. Install it with:\n"
                "  pip install paddleocr paddlepaddle"
            ) from exc
        logger.info("Loading PaddleOCR (lang=%s)", self.languages)
        # angle classification helps rotated plates/signs.
        self._paddle = PaddleOCR(use_angle_cls=True, lang=self.languages, show_log=False)

    # ── Public API ─────────────────────────────────────────────

    def read(self, image: Image.Image) -> list[str]:
        """Return the distinct text strings found in one frame."""
        if not self.enabled:
            return []
        try:
            if self.backend == "paddle":
                return self._read_paddle(image)
            return self._read_ollama(image)
        except Exception:
            logger.exception("OCR failed on a frame; returning no text")
            return []

    def read_batch(self, images: list[Image.Image]) -> list[list[str]]:
        """OCR every frame; returns one list of strings per input image.

        The Ollama backend is network-bound, so its requests run concurrently;
        local PaddleOCR runs sequentially (it already uses the CPU fully).
        """
        if not self.enabled:
            return [[] for _ in images]
        total = len(images)
        if self.backend == "ollama" and self._concurrency > 1 and total > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=self._concurrency) as pool:
                return list(pool.map(self.read, images))
        results: list[list[str]] = []
        for i, img in enumerate(images):
            logger.info("OCR frame %d/%d (%d remaining)", i + 1, total, total - i - 1)
            results.append(self.read(img))
        return results

    # ── Backends ───────────────────────────────────────────────

    def _read_paddle(self, image: Image.Image) -> list[str]:
        self._load_paddle()
        # PaddleOCR expects a BGR ndarray.
        bgr = np.array(image.convert("RGB"))[:, :, ::-1]
        result = self._paddle.ocr(bgr, cls=True)
        texts: list[str] = []
        # PaddleOCR 2.x: result is [ [ [box, (text, score)], ... ] ] (per image).
        for page in result or []:
            for line in page or []:
                try:
                    text, score = line[1][0], float(line[1][1])
                except (IndexError, TypeError, ValueError):
                    continue
                if score >= self.min_confidence and text.strip():
                    texts.append(text.strip())
        return _dedupe(texts)

    def _read_ollama(self, image: Image.Image) -> list[str]:
        import httpx

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode()
        payload = {
            "model": self.ollama_model,
            "messages": [{"role": "user", "content": _OCR_PROMPT, "images": [b64]}],
            "stream": False,
            "options": {"temperature": 0},
        }
        resp = httpx.post(f"{self.ollama_url}/api/chat", json=payload, timeout=180)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Ollama OCR request to {self.ollama_url} for model "
                f"'{self.ollama_model}' failed ({resp.status_code}): {resp.text}"
            )
        content = resp.json()["message"]["content"]
        lines = [_CLEAN_RE.sub(" ", ln).strip() for ln in content.splitlines()]
        return _dedupe([ln for ln in lines if ln])


def _dedupe(items: list[str]) -> list[str]:
    """Case-insensitive dedupe, preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


if __name__ == "__main__":
    import argparse

    from perception.config import configure_logging

    configure_logging()
    parser = argparse.ArgumentParser(description="Run OCR on an image with the configured backend")
    parser.add_argument("image", help="path to an image file")
    args = parser.parse_args()

    reader = OCRReader()
    print(f"backend={reader.backend}")
    for text in reader.read(Image.open(args.image).convert("RGB")):
        print(f"  {text}")

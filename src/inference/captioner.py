"""VLM dense captioning.

Backends (chosen once, at first use — never at import time), controlled by
``CAPTION_BACKEND``:

* ``qwen`` (default): **Qwen2.5-VL-3B-Instruct** — an instruction VLM that in a
  single prompt describes the scene, counts and characterises people, names
  interactions (handshake, handover, etc.), lists objects/vehicles, and reads
  legible text / number plates (OCR). On a capable GPU it loads in 4-bit; on
  CPU it runs in float32 (correct but slow — expect ~15-30 s/frame).
* ``blip``: BLIP-2 (opt-2.7b, 4-bit) on a GPU with >= ``VRAM_THRESHOLD_GB``,
  else BLIP-base (~1 GB, CPU-friendly but weak — generic captions, no actions).
* ``auto``: Qwen when a capable GPU is present, otherwise BLIP-base.

If the preferred backend fails to load (missing deps, OOM), it falls back to
BLIP-base so ingestion never hard-fails on captioning.

Caption deduplication: consecutive keyframes often show the same scene. If the
CLIP cosine similarity between the new caption and the previous one exceeds
``CAPTION_DEDUP_THRESHOLD`` (default 0.92), the previous caption is reused.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import numpy as np
from PIL import Image

from src.config import get_device, get_settings, get_vram_gb
from src.inference.embedder import Embedder
from src.models import Detection

logger = logging.getLogger(__name__)


# Single instruction that pulls out everything we care about per frame.
_QWEN_CAPTION_INSTRUCTION = (
    "You are analysing a single frame from a surveillance video. "
    "Describe it in 1-3 concise sentences, stating concretely:\n"
    "- how many people are visible and what each is doing;\n"
    "- any interaction between people (e.g. shaking hands, handing over an "
    "object, fighting, embracing, talking);\n"
    "- notable objects or vehicles;\n"
    "- any clearly legible text or vehicle number plates (read them out).\n"
    "Only state what is visibly certain — do not speculate."
)


class Captioner:
    """Lazy VLM wrapper producing one dense caption string per frame."""

    def __init__(self, embedder: Embedder | None = None) -> None:
        settings = get_settings()
        self.backend_pref = settings.caption_backend.lower()
        self.qwen_name = settings.qwen_vl_model
        self.max_new_tokens = settings.caption_max_new_tokens
        self.qwen_min_pixels = settings.qwen_min_pixels
        self.qwen_max_pixels = settings.qwen_max_pixels
        self.blip2_name = settings.blip2_model
        self.blip_base_name = settings.blip_base_model
        self.vram_threshold_gb = settings.vram_threshold_gb
        self.dedup_threshold = settings.caption_dedup_threshold
        # Shared with the pipeline's embedder so CLIP loads only once.
        self._embedder = embedder or Embedder()
        self._model: Any = None
        self._processor: Any = None
        self._device = "cpu"
        # "qwen" | "blip2" | "blip"
        self._backend = ""

    # ── Model loading ──────────────────────────────────────────

    def _load(self) -> None:
        if self._model is not None:
            return
        device = get_device()
        vram = get_vram_gb()
        capable_gpu = device == "cuda" and vram >= self.vram_threshold_gb

        if self.backend_pref in ("qwen", "auto"):
            if self.backend_pref == "auto" and not capable_gpu:
                logger.info("caption_backend=auto on CPU/small GPU: using BLIP base")
                self._load_blip_base(device)
                return
            try:
                self._load_qwen(device, capable_gpu)
                return
            except Exception:
                logger.exception("Qwen2.5-VL load failed; falling back to BLIP")

        # Explicit "blip" backend, or a Qwen fallback.
        if capable_gpu:
            try:
                self._load_blip2()
                return
            except Exception:
                logger.exception("BLIP-2 load failed; falling back to BLIP base")
        else:
            logger.info(
                "VRAM %.1f GB < %.1f GB threshold (or CPU-only): using BLIP base",
                vram, self.vram_threshold_gb,
            )
        self._load_blip_base(device)

    def _load_qwen(self, device: str, capable_gpu: bool) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        logger.info("Loading Qwen2.5-VL (%s)", self.qwen_name)
        self._processor = AutoProcessor.from_pretrained(
            self.qwen_name,
            min_pixels=self.qwen_min_pixels,
            max_pixels=self.qwen_max_pixels,
        )

        load_kwargs: dict[str, Any] = {}
        if capable_gpu:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = "auto"
            self._device = "cuda"
        elif device == "cuda":
            load_kwargs["torch_dtype"] = torch.float16
            self._device = "cuda"
        else:
            load_kwargs["torch_dtype"] = torch.float32
            self._device = "cpu"
            logger.warning(
                "Qwen2.5-VL on CPU is slow (~15-30 s/frame). Consider a GPU, a "
                "smaller MAX_KEYFRAMES, or CAPTION_BACKEND=blip for faster ingest."
            )

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.qwen_name, **load_kwargs
        )
        if "device_map" not in load_kwargs:
            self._model = self._model.to(self._device)
        self._model.eval()
        self._backend = "qwen"

    def _load_blip2(self) -> None:
        import torch
        from transformers import Blip2ForConditionalGeneration, Blip2Processor, BitsAndBytesConfig

        logger.info("Loading BLIP-2 (%s) in 4-bit", self.blip2_name)
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        self._processor = Blip2Processor.from_pretrained(self.blip2_name)
        self._model = Blip2ForConditionalGeneration.from_pretrained(
            self.blip2_name, quantization_config=quant, device_map="auto"
        )
        self._device = "cuda"
        self._backend = "blip2"

    def _load_blip_base(self, device: str) -> None:
        from transformers import BlipForConditionalGeneration, BlipProcessor

        logger.info("Loading BLIP base (%s) on %s", self.blip_base_name, device)
        self._processor = BlipProcessor.from_pretrained(self.blip_base_name)
        self._model = BlipForConditionalGeneration.from_pretrained(self.blip_base_name)
        self._model = self._model.to(device).eval()
        self._device = device
        self._backend = "blip"

    # ── Captioning ─────────────────────────────────────────────

    @staticmethod
    def _build_hint(detections: list[Detection]) -> str:
        """Format detections into a readable label string, e.g. "2 persons, 1 car"."""
        if not detections:
            return ""
        counts = Counter(d.label for d in detections)
        parts = []
        for label in sorted(counts):
            n = counts[label]
            plural = label if n == 1 or label.endswith("s") else label + "s"
            parts.append(f"{n} {plural}")
        return ", ".join(parts)

    def _caption_qwen(self, image: Image.Image, hint: str) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        prompt = _QWEN_CAPTION_INSTRUCTION
        if hint:
            prompt += f"\n\nAn object detector reported: {hint}."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.convert("RGB")},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            generated = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        text_out = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return " ".join(text_out.split()).strip()

    def _caption_blip(self, image: Image.Image, hint: str = "") -> str:
        import torch

        rgb = image.convert("RGB")
        if self._backend == "blip2":
            text = (
                "Question: Describe this scene in detail, including what the people are doing. "
                f"Detected objects: {hint}. Answer:"
                if hint
                else "Question: Describe this scene in detail, including what the people are doing. Answer:"
            )
            inputs = self._processor(images=rgb, text=text, return_tensors="pt")
        else:
            text = "In this security camera scene," if hint else ""
            inputs = (
                self._processor(images=rgb, text=text, return_tensors="pt")
                if text
                else self._processor(images=rgb, return_tensors="pt")
            )

        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=60)
        caption = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        if self._backend == "blip2" and "Answer:" in caption:
            caption = caption.split("Answer:", 1)[-1]
        return caption.strip()

    def _caption_one(self, image: Image.Image, hint: str = "") -> str:
        if self._backend == "qwen":
            return self._caption_qwen(image, hint)
        return self._caption_blip(image, hint)

    def caption_frames(
        self,
        images: list[Image.Image],
        detections: list[list[Detection]] | None = None,
    ) -> list[str]:
        """Caption every frame, deduplicating near-identical consecutive captions.

        When ``detections`` are supplied, detected object labels/counts are fed
        to the VLM as grounding so captions reference what was actually found.
        Failures on individual frames are logged and yield an empty caption
        rather than aborting the batch.
        """
        self._load()
        captions: list[str] = []
        prev_caption = ""
        prev_vec: np.ndarray | None = None

        for i, image in enumerate(images):
            hint = self._build_hint(detections[i]) if detections else ""
            try:
                caption = self._caption_one(image, hint=hint)
            except Exception:
                logger.exception("Captioning failed on frame %d; using empty caption", i)
                captions.append("")
                continue

            if caption and prev_vec is not None:
                try:
                    vec = self._embedder.embed_text(caption)
                    similarity = float(vec @ prev_vec)
                    if similarity > self.dedup_threshold:
                        logger.debug(
                            "Frame %d caption ~ previous (cos=%.3f); reusing '%s'",
                            i, similarity, prev_caption,
                        )
                        captions.append(prev_caption)
                        continue
                    prev_vec = vec
                except Exception:
                    logger.exception("Caption dedup embedding failed; keeping caption")
                    prev_vec = None
            elif caption:
                try:
                    prev_vec = self._embedder.embed_text(caption)
                except Exception:
                    prev_vec = None

            prev_caption = caption
            captions.append(caption)

        return captions

    def caption(self, image: Image.Image) -> str:
        """Convenience wrapper for a single image."""
        return self.caption_frames([image])[0]

    def caption_actions(
        self,
        images: list[Image.Image],
        detections: list[list[Detection]] | None = None,
    ) -> list[list[str]]:
        """Return per-frame action descriptions.

        Qwen2.5-VL already folds actions/interactions into its main caption, so
        this returns empties for the Qwen backend (no slow second pass). For the
        BLIP backends it runs a focused VQA-style pass on frames with 2+ people.
        """
        self._load()

        if self._backend == "qwen":
            # Actions are captured in the primary caption already.
            return [[] for _ in images]

        results: list[list[str]] = []
        for i, image in enumerate(images):
            person_count = 0
            if detections:
                person_count = sum(
                    1 for d in detections[i] if d.label.lower() == "person"
                )
            if person_count < 2 and detections is not None:
                results.append([])
                continue
            try:
                action = self._caption_action_one(image)
                results.append([action] if action else [])
            except Exception:
                logger.exception("Action captioning failed on frame %d", i)
                results.append([])
        return results

    def _caption_action_one(self, image: Image.Image) -> str:
        """Run one VQA-style caption focused on what people are doing (BLIP)."""
        import torch

        rgb = image.convert("RGB")
        if self._backend == "blip2":
            text = (
                "Question: What are the people doing? "
                "Are they shaking hands, fighting, embracing, or interacting in any way? "
                "Answer:"
            )
            inputs = self._processor(images=rgb, text=text, return_tensors="pt")
        else:
            text = "The people in this scene are"
            inputs = self._processor(images=rgb, text=text, return_tensors="pt")

        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=40)
        answer = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        if self._backend == "blip2" and "Answer:" in answer:
            answer = answer.split("Answer:", 1)[-1]
        return answer.strip()


if __name__ == "__main__":
    import argparse

    from src.config import configure_logging

    configure_logging()
    parser = argparse.ArgumentParser(description="Caption a single image with the active VLM")
    parser.add_argument("image", help="path to an image file")
    args = parser.parse_args()

    captioner = Captioner()
    print(captioner.caption(Image.open(args.image).convert("RGB")))

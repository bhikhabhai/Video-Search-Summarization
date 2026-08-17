"""Weapon detection via a fine-tuned YOLOv11m (handgun / long_gun / knife).

An optional component (like OCR and Pose): when ``WEAPON_ENABLED`` is set, each
keyframe is run through the weapon model and any hits become:
  * extra detections (with boxes) merged into the frame — so "gun"/"knife"
    queries draw them,
  * a per-frame ``weapon detected: <class>`` event — so it appears in the
    event timeline, Q&A, summary, and gets highlighted on retrieved frames,
  * a grounding hint fed to the caption VLM.

Runs lazily on ``WEAPON_DEVICE`` (default auto → GPU if present). On a tiny GPU
already holding the caption VLM, set ``WEAPON_DEVICE=cpu`` — YOLO is fast on CPU.

Honest limits: a weapon detector flags candidates for review, not ground truth.
Run at a high confidence to suppress false alarms (phones/tools as guns), and
expect concealed/partial weapons to be missed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image

    from perception.models import Detection

logger = logging.getLogger(__name__)


class WeaponDetector:
    """Lazy YOLOv11m weapon detector producing per-frame weapon detections."""

    def __init__(self) -> None:
        from perception.config import get_settings

        s = get_settings()
        self.enabled = s.weapon_enabled
        self.model_name = s.weapon_model
        self.confidence = s.weapon_confidence
        self.imgsz = s.weapon_imgsz
        self.device_pref = s.weapon_device.lower()
        self.batch_size = s.detector_batch_size
        self._model: Any = None
        self._device = "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO

        from perception.config import get_device

        self._device = get_device() if self.device_pref == "auto" else self.device_pref
        logger.info("Loading weapon model %s on %s", self.model_name, self._device)
        self._model = YOLO(self.model_name)
        self._model.to(self._device)

    def detect_batch(self, images: list["Image.Image"]) -> list[list["Detection"]]:
        """Return weapon detections per frame ([] when disabled or none found)."""
        if not self.enabled:
            return [[] for _ in images]
        from perception.models import Detection

        self._load()
        out: list[list[Detection]] = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size]
            try:
                results = self._model(
                    batch, conf=self.confidence, imgsz=self.imgsz,
                    device=self._device, verbose=False,
                )
            except Exception:
                logger.exception("Weapon detection failed for a batch; skipping")
                out.extend([[] for _ in batch])
                continue

            for image, res in zip(batch, results):
                dets: list[Detection] = []
                w, h = image.size
                boxes = getattr(res, "boxes", None)
                if boxes is not None:
                    for k, b in enumerate(boxes):
                        label = self._model.names[int(b.cls[0])]
                        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                        dets.append(
                            Detection(
                                label=label,
                                confidence=round(float(b.conf[0]), 3),
                                bbox=(round(x1 / w, 4), round(y1 / h, 4),
                                      round(x2 / w, 4), round(y2 / h, 4)),
                                # Unique within the frame so events can point at it.
                                track_id=f"{label}_{k}",
                            )
                        )
                out.append(dets)
        return out


if __name__ == "__main__":
    import argparse

    from PIL import Image

    from perception.config import configure_logging

    configure_logging()
    parser = argparse.ArgumentParser(description="Run the weapon detector on an image")
    parser.add_argument("image")
    args = parser.parse_args()

    det = WeaponDetector()
    det.enabled = True  # force-on for the CLI
    for d in det.detect_batch([Image.open(args.image).convert("RGB")])[0]:
        print(f"  {d.label:<10} conf={d.confidence:.2f} bbox={d.bbox}")

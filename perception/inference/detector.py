"""YOLOv8n object detection wrapper.

Design notes:
* The model is loaded lazily on first use, never at import time.
* Frames are processed in batches of ``detector_batch_size`` (default 8) to
  maximise throughput on both CPU and GPU.
* A class-name filter (env var ``DETECTOR_CLASSES``, comma-separated COCO
  names) restricts which labels are kept; empty means all 80 classes.
* Every batch is wrapped in try/except: a failing batch yields empty
  detections for its frames instead of aborting the whole video.
"""

from __future__ import annotations

import logging
from typing import Any

from PIL import Image

from perception.config import get_device, get_settings
from perception.models import Detection

logger = logging.getLogger(__name__)


class Detector:
    """Batched YOLOv8n inference with lazy loading and graceful degradation."""

    def __init__(self) -> None:
        settings = get_settings()
        self.model_name = settings.detector_model
        self.confidence = settings.detector_confidence
        self.agnostic_nms = settings.agnostic_nms
        self.nms_iou = settings.nms_iou
        self.batch_size = settings.detector_batch_size
        self.class_filter: set[str] = {
            name.strip().lower()
            for name in settings.detector_classes.split(",")
            if name.strip()
        }
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            device = get_device()
            logger.info("Loading %s on %s", self.model_name, device)
            self._model = YOLO(self.model_name)
            self._model.to(device)
        return self._model

    def _keep(self, label: str) -> bool:
        return not self.class_filter or label.lower() in self.class_filter

    def detect_batch(self, images: list[Image.Image]) -> list[list[Detection]]:
        """Run detection over all images, batched in groups of ``batch_size``.

        Returns one (possibly empty) list of :class:`Detection` per input
        image, in the same order.
        """
        model = self._load()
        all_results: list[list[Detection]] = []

        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size]
            try:
                outputs = model(batch, conf=self.confidence, iou=self.nms_iou,
                                agnostic_nms=self.agnostic_nms, verbose=False)
            except Exception:
                logger.exception(
                    "Detection failed for frames %d-%d; returning empty detections",
                    start, start + len(batch) - 1,
                )
                all_results.extend([[] for _ in batch])
                continue

            for image, output in zip(batch, outputs):
                detections: list[Detection] = []
                width, height = image.size
                boxes = getattr(output, "boxes", None)
                if boxes is not None:
                    for box in boxes:
                        label = model.names[int(box.cls[0])]
                        if not self._keep(label):
                            continue
                        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                        detections.append(
                            Detection(
                                label=label,
                                confidence=round(float(box.conf[0]), 3),
                                bbox=(
                                    round(x1 / width, 4),
                                    round(y1 / height, 4),
                                    round(x2 / width, 4),
                                    round(y2 / height, 4),
                                ),
                            )
                        )
                all_results.append(detections)

        return all_results

    def detect(self, image: Image.Image) -> list[Detection]:
        """Convenience wrapper for a single image."""
        return self.detect_batch([image])[0]


if __name__ == "__main__":
    import argparse

    from perception.config import configure_logging

    configure_logging()
    parser = argparse.ArgumentParser(description="Run YOLOv8n on a single image")
    parser.add_argument("image", help="path to an image file")
    args = parser.parse_args()

    detector = Detector()
    for det in detector.detect(Image.open(args.image).convert("RGB")):
        print(f"  {det.label:<15} conf={det.confidence:.2f} bbox={det.bbox}")

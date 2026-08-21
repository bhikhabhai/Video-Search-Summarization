"""Pose-based action/event detection (fall / fight / running / crowding).

Two clearly-separated parts:

* :class:`PoseEstimator` — wraps YOLOv8-pose to turn each frame into a list of
  :class:`Person` keypoint sets (lazy-loaded; same ``ultralytics`` package as
  the object detector, so no extra dependency).
* :func:`detect_events` and the ``is_*`` predicates — a **pure function of
  keypoints**, so the rule logic is unit-testable with synthetic poses and no
  model weights.

Important honesty about limits: we only have sparse keyframes, not continuous
video or object tracks. So single-frame **fall** detection is reliable (body
geometry), while **fight** and **running** are pose heuristics (proximity +
raised/extended arms; wide stance + forward lean) and are best-effort. Proper
temporal actions need Phase 1c tracking + Phase 3 video models.

COCO-17 keypoint order is assumed (YOLO-pose default).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from perception.config import get_settings

logger = logging.getLogger(__name__)

# COCO-17 keypoint indices.
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# Rule thresholds (tunable; normalised [0,1] coords unless noted).
FALL_ASPECT_RATIO = 1.3       # bbox width/height above this => lying down
FALL_TORSO_ANGLE_DEG = 55.0   # torso tilt from vertical above this => fallen
CLOSE_CENTER_DIST = 0.20      # normalised centre distance => people "close"
RUN_STANCE_RATIO = 1.6        # ankle spread / hip width above this => wide stance
RUN_LEAN_RATIO = 0.25         # torso horizontal offset / height => forward lean


@dataclass
class Person:
    """One detected person: normalised keypoints (NaN where low-confidence)."""

    keypoints: np.ndarray  # shape (17, 2), normalised [0,1], NaN = missing
    bbox: tuple[float, float, float, float]  # normalised xyxy

    def kp(self, idx: int) -> tuple[float, float] | None:
        x, y = self.keypoints[idx]
        if np.isnan(x) or np.isnan(y):
            return None
        return float(x), float(y)

    def midpoint(self, i: int, j: int) -> tuple[float, float] | None:
        pts = [p for p in (self.kp(i), self.kp(j)) if p is not None]
        if not pts:
            return None
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def bbox_aspect(self) -> float:
        x1, y1, x2, y2 = self.bbox
        h = max(y2 - y1, 1e-6)
        return (x2 - x1) / h


# ── Pure rule engine (unit-testable) ──────────────────────────


def torso_angle_deg(person: Person) -> float | None:
    """Angle of the shoulder→hip vector from vertical (0=upright, 90=horizontal)."""
    shoulder = person.midpoint(L_SHOULDER, R_SHOULDER)
    hip = person.midpoint(L_HIP, R_HIP)
    if shoulder is None or hip is None:
        return None
    dx = hip[0] - shoulder[0]
    dy = hip[1] - shoulder[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def is_fall(person: Person) -> bool:
    """Body horizontal — via a wide bbox or a near-horizontal torso."""
    if person.bbox_aspect > FALL_ASPECT_RATIO:
        return True
    angle = torso_angle_deg(person)
    return angle is not None and angle > FALL_TORSO_ANGLE_DEG


def has_raised_arm(person: Person) -> bool:
    """A wrist above its shoulder (raised/extended arm — strike, wave, surrender)."""
    for wrist_idx, shoulder_idx in ((L_WRIST, L_SHOULDER), (R_WRIST, R_SHOULDER)):
        wrist = person.kp(wrist_idx)
        shoulder = person.kp(shoulder_idx)
        # y grows downward, so "above" means a smaller y.
        if wrist and shoulder and wrist[1] < shoulder[1]:
            return True
    return False


def is_running(person: Person) -> bool:
    """Heuristic: wide stride + forward torso lean (approximate, single-frame)."""
    l_ank, r_ank = person.kp(L_ANKLE), person.kp(R_ANKLE)
    l_hip, r_hip = person.kp(L_HIP), person.kp(R_HIP)
    shoulder = person.midpoint(L_SHOULDER, R_SHOULDER)
    hip = person.midpoint(L_HIP, R_HIP)
    if not (l_ank and r_ank and l_hip and r_hip and shoulder and hip):
        return False
    hip_width = abs(l_hip[0] - r_hip[0])
    ankle_spread = abs(l_ank[0] - r_ank[0])
    if hip_width < 1e-6:
        return False
    wide_stance = ankle_spread > RUN_STANCE_RATIO * hip_width
    body_height = max(abs(hip[1] - shoulder[1]), 1e-6)
    forward_lean = abs(shoulder[0] - hip[0]) > RUN_LEAN_RATIO * body_height
    return wide_stance and forward_lean


def _center_distance(a: Person, b: Person) -> float:
    ax, ay = a.center
    bx, by = b.center
    return math.hypot(ax - bx, ay - by)


def detect_events(persons: list[Person], crowd_threshold: int = 5) -> list[str]:
    """Scene-level events for one frame from its detected people."""
    events: set[str] = set()

    for person in persons:
        if is_fall(person):
            events.add("fall")
        if is_running(person):
            events.add("running")

    # Fight: two nearby people with at least one raised/extended arm.
    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            if _center_distance(persons[i], persons[j]) < CLOSE_CENTER_DIST and (
                has_raised_arm(persons[i]) or has_raised_arm(persons[j])
            ):
                events.add("fighting")

    if len(persons) >= crowd_threshold:
        events.add("crowding")

    return sorted(events)


# ── YOLO-pose wrapper ──────────────────────────────────────────


class PoseEstimator:
    """Lazy YOLOv8-pose wrapper producing :class:`Person` sets per frame."""

    def __init__(self) -> None:
        settings = get_settings()
        self.enabled = settings.pose_enabled
        self.model_name = settings.pose_model
        self.confidence = settings.pose_confidence
        self.kp_confidence = settings.pose_kp_confidence
        self.crowd_threshold = settings.crowd_person_threshold
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO

        from perception.config import get_device

        logger.info("Loading YOLO-pose (%s)", self.model_name)
        self._model = YOLO(self.model_name)
        self._device = get_device()

    def unload(self) -> None:
        """Release the pose model from VRAM so the caption VLM gets the memory.

        Lazy-reloaded on the next call — lets pose run 'at tracker time' and hand
        its GPU budget to Qwen before captioning, instead of coexisting with it.
        """
        if self._model is None:
            return
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def estimate(self, image: Image.Image) -> list[Person]:
        return self.estimate_batch([image])[0]

    def estimate_batch(self, images: list[Image.Image]) -> list[list[Person]]:
        """Return the detected people (with keypoints) for each frame."""
        if not self.enabled:
            return [[] for _ in images]
        self._load()
        try:
            results = self._model(
                images, verbose=False, conf=self.confidence, device=self._device
            )
        except Exception:
            logger.exception("Pose estimation failed; returning no poses")
            return [[] for _ in images]

        out: list[list[Person]] = []
        for res in results:
            people: list[Person] = []
            kps = getattr(res, "keypoints", None)
            boxes = getattr(res, "boxes", None)
            if kps is None or boxes is None or kps.xyn is None:
                out.append(people)
                continue
            xyn = kps.xyn.cpu().numpy()  # (n, 17, 2) normalised
            confs = kps.conf.cpu().numpy() if kps.conf is not None else None
            xyxyn = boxes.xyxyn.cpu().numpy()  # (n, 4) normalised
            for i in range(len(xyn)):
                pts = xyn[i].astype(float).copy()
                if confs is not None:
                    pts[confs[i] < self.kp_confidence] = np.nan
                else:
                    # No per-keypoint conf: treat exact (0,0) as missing.
                    pts[(pts[:, 0] == 0) & (pts[:, 1] == 0)] = np.nan
                people.append(Person(keypoints=pts, bbox=tuple(xyxyn[i].tolist())))
            out.append(people)
        return out

    def events_batch(self, images: list[Image.Image]) -> list[list[str]]:
        """Convenience: pose-estimate then apply the rule engine per frame."""
        return [
            detect_events(persons, self.crowd_threshold)
            for persons in self.estimate_batch(images)
        ]


if __name__ == "__main__":
    import argparse

    from perception.config import configure_logging

    configure_logging()
    parser = argparse.ArgumentParser(description="Detect pose-based events in an image")
    parser.add_argument("image")
    args = parser.parse_args()

    estimator = PoseEstimator()
    estimator.enabled = True  # force-on for the CLI
    persons = estimator.estimate(Image.open(args.image).convert("RGB"))
    print(f"{len(persons)} person(s)")
    print("events:", detect_events(persons, estimator.crowd_threshold))

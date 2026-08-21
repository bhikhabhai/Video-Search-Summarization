"""Temporal (multi-frame) pose action detection.

Single-frame pose can only guess posture; *motion* needs a sequence. This runs
YOLO-pose on every frame where a person is present, associates each skeleton to
its existing track id (by IoU with the tracker's person boxes — so no new
tracker is needed), buffers a short keypoint history per track, and fires
actions from the temporal signal:

  per-track : fall, running, sudden movement (anomaly)
  pairwise  : handshake, possible fight

Fired events use the same ``{"id": <track label>, "phrase": <text>}`` shape as
:class:`EventDeriver`, so they merge straight into the track pass's event stream
(keyframe trigger + timeline + drawing + caption injection) with no extra wiring.

Two clearly-separated parts, mirroring pose.py:
  * :class:`TemporalActionDetector` — a **pure function of pose sequences** (no
    model), unit-tested in ``__main__`` with synthetic keypoints.
  * :class:`PoseActionRunner` — adds the YOLO-pose model + IoU association.

Honest limits: on overhead/CCTV views yolov8n-pose keypoints are noisy, so
these rules are best-effort. Fall (body geometry) is the most robust; the
pairwise ones are heuristics. Thresholds are module constants — tune per camera.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image

    from perception.inference.pose import Person

logger = logging.getLogger(__name__)

# ── Tunable thresholds (normalised [0,1] coords; speeds are per second) ──
WINDOW_SEC = 1.5          # keypoint history kept per track
COOLDOWN_SEC = 3.0        # don't re-fire the same action for the same subject
FALL_ANGLE_DEG = 55.0     # torso tilt from vertical => fallen
FALL_UPRIGHT_DEG = 30.0   # must have been at least this upright just before
RUN_SPEED = 0.22          # sustained centroid speed (frac of frame / s) => running
ANOMALY_SPEED = 0.55      # instantaneous centroid speed spike => sudden movement
PAIR_CENTER_DIST = 0.22   # two people this close (centre-centre) are "interacting"
HANDSHAKE_WRIST_DIST = 0.07   # wrists this close + low speed + held => handshake
HANDSHAKE_HOLD_FRAMES = 2     # frames the wrists must stay close
FIGHT_LIMB_SPEED = 0.5    # fast wrist motion while close => possible fight


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def _centroid_speed(buf: "deque") -> float:
    """Mean centre-to-centre speed (per second) across the buffered window."""
    if len(buf) < 2:
        return 0.0
    speeds = []
    prev_ts, prev_p = buf[0]
    for ts, p in list(buf)[1:]:
        dt = ts - prev_ts
        if dt > 1e-6:
            cx, cy = p.center
            px, py = prev_p.center
            speeds.append(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 / dt)
        prev_ts, prev_p = ts, p
    return sum(speeds) / len(speeds) if speeds else 0.0


def _peak_speed(buf: "deque") -> float:
    if len(buf) < 2:
        return 0.0
    peak = 0.0
    prev_ts, prev_p = buf[0]
    for ts, p in list(buf)[1:]:
        dt = ts - prev_ts
        if dt > 1e-6:
            cx, cy = p.center
            px, py = prev_p.center
            peak = max(peak, ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 / dt)
        prev_ts, prev_p = ts, p
    return peak


def _wrists(person: "Person") -> list[tuple]:
    from perception.inference.pose import L_WRIST, R_WRIST

    return [w for w in (person.kp(L_WRIST), person.kp(R_WRIST)) if w is not None]


def _min_wrist_dist(a: "Person", b: "Person") -> float:
    wa, wb = _wrists(a), _wrists(b)
    if not wa or not wb:
        return 1e9
    return min(((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5 for p in wa for q in wb)


class TemporalActionDetector:
    """Per-track pose buffers + temporal rules. Pure (no model); testable."""

    def __init__(self, window_sec: float = WINDOW_SEC, cooldown_sec: float = COOLDOWN_SEC) -> None:
        self.window = window_sec
        self.cooldown = cooldown_sec
        self._buf: dict[str, deque] = defaultdict(deque)
        self._last_fired: dict[tuple, float] = {}

    def _fire(self, key: tuple, ts: float) -> bool:
        """Debounce: True the first time / after cooldown for this (action, subject)."""
        last = self._last_fired.get(key)
        if last is not None and (ts - last) < self.cooldown:
            return False
        self._last_fired[key] = ts
        return True

    def update(self, ts: float, people: dict[str, "Person"]) -> list[dict]:
        """Feed one frame's ``{track_id: Person}``; return fired action events."""
        for tid, person in people.items():
            buf = self._buf[tid]
            buf.append((ts, person))
            while buf and (ts - buf[0][0]) > self.window:
                buf.popleft()

        out: list[dict] = []

        # ── Per-track rules ────────────────────────────────────
        for tid, person in people.items():
            buf = self._buf[tid]
            from perception.inference.pose import is_fall, torso_angle_deg

            # Fall: currently down, but was upright within the window.
            if is_fall(person):
                was_upright = any(
                    (a := torso_angle_deg(p)) is not None and a < FALL_UPRIGHT_DEG
                    for _, p in buf
                )
                if was_upright and self._fire(("fall", tid), ts):
                    out.append({"id": tid, "phrase": f"{tid} fell"})

            speed = _centroid_speed(buf)
            if speed > RUN_SPEED and self._fire(("running", tid), ts):
                out.append({"id": tid, "phrase": f"{tid} running"})
            elif _peak_speed(buf) > ANOMALY_SPEED and self._fire(("sudden", tid), ts):
                out.append({"id": tid, "phrase": f"{tid} sudden movement"})

        # ── Pairwise rules ─────────────────────────────────────
        ids = sorted(people)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = people[ids[i]], people[ids[j]]
                ca, cb = a.center, b.center
                if ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5 > PAIR_CENTER_DIST:
                    continue
                pair = tuple(sorted((ids[i], ids[j])))
                wrist_d = _min_wrist_dist(a, b)
                fast = (_peak_speed(self._buf[ids[i]]) > FIGHT_LIMB_SPEED
                        or _peak_speed(self._buf[ids[j]]) > FIGHT_LIMB_SPEED)
                if wrist_d < HANDSHAKE_WRIST_DIST and not fast:
                    if self._fire(("handshake", pair), ts):
                        out.append({"id": ids[i],
                                    "phrase": f"handshake between {ids[i]} and {ids[j]}"})
                elif fast and self._fire(("fight", pair), ts):
                    out.append({"id": ids[i],
                                "phrase": f"possible fight between {ids[i]} and {ids[j]}"})
        return out


class PoseActionRunner:
    """YOLO-pose per person-frame + IoU association + temporal detection."""

    def __init__(self) -> None:
        from perception.config import get_settings
        from perception.inference.pose import PoseEstimator

        s = get_settings()
        self.enabled = s.pose_action_enabled
        self.pose = PoseEstimator()
        self.pose.enabled = True  # gated by pose_action_enabled here, not POSE_ENABLED
        self.detector = TemporalActionDetector()

    def step(self, frame_bgr, ts: float, person_boxes: dict[str, tuple]) -> list[dict]:
        """One frame: ``person_boxes`` = {track label: normalised xyxy}. Returns events."""
        if not self.enabled or not person_boxes:
            return []
        import cv2
        from PIL import Image

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            people = self.pose.estimate(Image.fromarray(rgb))
        except Exception:
            logger.exception("Pose estimation failed on a frame; skipping actions")
            return []

        # Associate each skeleton to a track id by best IoU (>0.3).
        matched: dict[str, Any] = {}
        for tid, box in person_boxes.items():
            best, best_iou = None, 0.3
            for person in people:
                iou = _iou(person.bbox, box)
                if iou > best_iou:
                    best, best_iou = person, iou
            if best is not None:
                matched[tid] = best
        return self.detector.update(ts, matched)

    def unload(self) -> None:
        self.pose.unload()


if __name__ == "__main__":
    # Self-check: a synthetic upright→horizontal sequence must fire "fell".
    import numpy as np

    from perception.inference.pose import L_HIP, L_SHOULDER, Person, R_HIP, R_SHOULDER

    def make(angle_horizontal: bool) -> Person:
        kp = np.full((17, 2), np.nan)
        # shoulders high, hips low = upright; shoulders/hips side-by-side = fallen.
        if angle_horizontal:
            kp[L_SHOULDER] = [0.40, 0.50]; kp[R_SHOULDER] = [0.42, 0.50]
            kp[L_HIP] = [0.60, 0.51]; kp[R_HIP] = [0.62, 0.51]   # torso horizontal
            bbox = (0.4, 0.48, 0.62, 0.56)  # wide/short => lying
        else:
            kp[L_SHOULDER] = [0.50, 0.30]; kp[R_SHOULDER] = [0.52, 0.30]
            kp[L_HIP] = [0.50, 0.60]; kp[R_HIP] = [0.52, 0.60]   # torso vertical
            bbox = (0.48, 0.28, 0.54, 0.62)  # tall
        return Person(keypoints=kp, bbox=bbox)

    d = TemporalActionDetector()
    assert d.update(0.0, {"person_1": make(False)}) == []          # upright, nothing
    assert d.update(0.3, {"person_1": make(False)}) == []
    fired = d.update(0.6, {"person_1": make(True)})                 # now horizontal
    assert any("fell" in e["phrase"] for e in fired), fired
    # Debounced: firing again immediately does nothing.
    assert d.update(0.7, {"person_1": make(True)}) == []
    print("TemporalActionDetector self-check OK:", fired)

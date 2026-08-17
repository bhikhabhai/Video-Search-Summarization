"""Whole-video multi-object tracking (YOLO + ByteTrack) → keyframes + counts.

One pass over *every* frame of the video (no frame skipping) does three jobs:

1. **Precise unique counts** — ByteTrack assigns a stable id per object, so the
   number of distinct ids per class is the true count of unique people /
   vehicles (not per-frame detections, which double-count across frames).
2. **Track-driven keyframes** — a keyframe is emitted when the set of active
   track ids changes (someone/something enters or leaves) or periodically while
   objects persist. This beats scene-change detection on fixed cameras, where
   the background never changes but subjects come and go.
3. **Per-keyframe detections** — taken straight from the track pass, so no
   separate detection step is needed.

The keyframe-selection rule (:class:`KeyframePicker`) is a pure function of
``(timestamp, active_ids)`` and is unit-tested in ``__main__`` without any video.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from perception.models import Detection, ExtractedFrame, VideoMeta

logger = logging.getLogger(__name__)


class KeyframePicker:
    """Decide, frame by frame, whether the current frame is a keyframe.

    Emits when the active track-id set changes (respecting a min gap) or when
    ``period_sec`` has elapsed since the last keyframe (a time-based safety net
    so long stretches are never skipped).
    """

    def __init__(self, period_sec: float, min_gap_sec: float) -> None:
        self.period = period_sec
        self.min_gap = min_gap_sec
        self._last_ids: frozenset[int] | None = None
        self._last_ts = -math.inf

    def emit(self, ts: float, ids: frozenset) -> bool:
        changed = ids != self._last_ids
        gap_ok = (ts - self._last_ts) >= self.min_gap
        periodic = (ts - self._last_ts) >= self.period
        if (changed and gap_ok) or periodic:
            self._last_ids = ids
            self._last_ts = ts
            return True
        return False

    def mark(self, ts: float, ids: frozenset) -> None:
        """Record a keyframe forced by something else (e.g. an event)."""
        self._last_ids = ids
        self._last_ts = ts


# ── Events ─────────────────────────────────────────────────────

# type code -> phrase template ({zone}/{dir} filled in). Used only when an IVA
# event has no pre-rendered "label".
_IVA_EVENT_PHRASES = {
    "TW": "crossed {zone} tripwire{dir}",
    "INTRUSION": "entered restricted {zone}",
    "ZONE_ENTER": "entered {zone}",
    "ZONE_EXIT": "left {zone}",
    "LOITER": "loitering in {zone}",
    "CROWD": "crowd in {zone}",
    "WRONG_DIR": "moving the wrong way in {zone}",
    "ABANDONED": "left an unattended object in {zone}",
    "REMOVED": "removed an object from {zone}",
    "ILLEGAL_PARK": "parked illegally in {zone}",
    "PK+": "parked in {zone}",
    "PK-": "left parking {zone}",
    "STOPPED": "stopped in {zone}",
    "FALL": "fell", "FIGHT": "fighting", "RUN": "running",
    "FIRE": "fire", "SMOKE": "smoke", "PPE_MISSING": "missing PPE",
}


def render_iva_event(ev: dict, label_of) -> dict:
    """Render one IVA event to ``{"id": <track label>, "phrase": <text>}``.

    ``id`` is the triggering object's track label (or "" for scene-level events),
    so a box can be highlighted for it. Uses ``ev['label']`` verbatim if present.
    """
    who = label_of(ev["id"]) if ev.get("id") else ""
    if ev.get("label"):
        return {"id": who, "phrase": str(ev["label"]).strip()}
    code = str(ev.get("type", "")).upper()
    zone = str(ev.get("zone", "")).strip() or "the area"
    d = str(ev.get("dir", "")).strip()
    dir_txt = f" ({d})" if d else ""
    phrase = _IVA_EVENT_PHRASES.get(code, code.lower() or "event")
    phrase = phrase.format(zone=zone, dir=dir_txt)
    return {"id": who, "phrase": f"{who} {phrase}".strip()}


class EventDeriver:
    """Derive entered/left/loitering events from per-frame active-id sets.

    No geometry needed — purely from track lifecycle, so it works for both the
    IVA path and our own tracker. ``label_of``/``t_start_of`` are callables the
    caller supplies (the two loaders store tracks differently).
    """

    def __init__(self, loiter_sec: float) -> None:
        self.loiter_sec = loiter_sec
        self._prev: frozenset = frozenset()
        self._loitered: set = set()

    def step(self, ts, active, label_of, t_start_of) -> list[dict]:
        """Return events as ``[{"id": <track label>, "phrase": <text>}, ...]``."""
        active = set(active)
        out: list[dict] = []
        for oid in sorted(active - self._prev):
            out.append({"id": label_of(oid), "phrase": f"{label_of(oid)} entered"})
        for oid in sorted(self._prev - active):
            out.append({"id": label_of(oid), "phrase": f"{label_of(oid)} left"})
        for oid in sorted(active):
            if oid not in self._loitered and (ts - t_start_of(oid)) >= self.loiter_sec:
                self._loitered.add(oid)
                dwell = ts - t_start_of(oid)
                out.append({"id": label_of(oid),
                            "phrase": f"{label_of(oid)} loitering ({dwell:.0f}s)"})
        self._prev = frozenset(active)
        return out


def _summarise(tracks: dict[int, dict], names: dict[int, str]) -> dict:
    """Build the per-video track summary: unique counts + one row per track."""
    counts: Counter[str] = Counter()
    rows: list[dict] = []
    for tid, info in tracks.items():
        cls = info["cls"].most_common(1)[0][0]  # majority class over the track's life
        counts[cls] += 1
        rows.append(
            {
                "id": f"{cls}_{tid}",
                "cls": cls,
                "t_start": round(info["t_start"], 2),
                "t_end": round(info["t_end"], 2),
                "frames": info["frames"],
            }
        )
    rows.sort(key=lambda r: r["t_start"])
    return {"counts": dict(counts), "tracks": rows}


class VideoTracker:
    """Run YOLO+ByteTrack over a whole video; yield keyframes, detections, counts."""

    def __init__(self) -> None:
        from perception.config import get_settings

        s = get_settings()
        self.model_name = s.detector_model
        self.confidence = s.detector_confidence
        self.classes = [int(c) for c in s.track_classes.split(",") if c.strip()]
        self.agnostic_nms = s.agnostic_nms
        self.nms_iou = s.nms_iou
        self.imgsz = s.track_imgsz
        self.loiter_sec = s.track_loiter_sec
        # Absolute path so it resolves regardless of working directory.
        self.tracker_cfg = str(Path(__file__).with_name(s.tracker_config))
        self.period = s.track_keyframe_period_sec
        self.min_gap = s.track_min_gap_sec
        self.max_keyframes = s.max_keyframes
        self.save_video = s.save_track_video
        self.video_out_dir = s.data_dir / "tracks_video"
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            from perception.config import get_device

            device = get_device()
            logger.info("Loading tracker %s on %s", self.model_name, device)
            self._model = YOLO(self.model_name)
            self._model.to(device)
        return self._model

    def process(
        self, meta: VideoMeta
    ) -> tuple[list[ExtractedFrame], list[list[Detection]], list[str], list[list[dict]], dict]:
        """Track every frame; return (keyframes, detections, facts, events, summary)."""
        import cv2
        from PIL import Image

        from perception.models import Detection, ExtractedFrame

        model = self._load()
        fps = meta.fps or 25.0
        picker = KeyframePicker(self.period, self.min_gap)
        deriver = EventDeriver(self.loiter_sec)
        tracks: dict[int, dict] = {}
        keyframes: list[ExtractedFrame] = []
        kf_dets: list[list[Detection]] = []
        kf_facts: list[str] = []
        kf_events: list[list[dict]] = []
        video_events: list[dict] = []

        def _label(tid: int) -> str:
            return f"{tracks[tid]['cls'].most_common(1)[0][0]}_{tid}"

        logger.info(
            "Tracking %s over every frame (classes=%s, fps=%.2f)",
            meta.source, self.classes, fps,
        )
        results = model.track(
            source=meta.source,
            stream=True,               # frame-by-frame, low memory
            tracker=self.tracker_cfg,  # BoT-SORT+ReID (or ByteTrack) config
            classes=self.classes,
            conf=self.confidence,
            imgsz=self.imgsz,                # higher res -> reliable small/distant detection
            iou=self.nms_iou,                # merge overlapping boxes on one object
            agnostic_nms=self.agnostic_nms,  # one vehicle -> one box (not car+truck)
            verbose=False,
        )

        writer = None
        out_path = self.video_out_dir / f"{meta.video_id}.mp4"
        if self.save_video:
            self.video_out_dir.mkdir(parents=True, exist_ok=True)

        try:
            for i, r in enumerate(results):
                ts = i / fps
                if self.save_video:
                    annotated = r.plot()  # BGR frame with boxes + track IDs
                    if writer is None:
                        vh, vw = annotated.shape[:2]
                        writer = cv2.VideoWriter(
                            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                            fps, (vw, vh),
                        )
                    writer.write(annotated)

                active: set[int] = set()
                dets: list[Detection] = []
                boxes = getattr(r, "boxes", None)
                if boxes is not None and boxes.id is not None:
                    h, w = r.orig_shape
                    for b in boxes:
                        tid = int(b.id[0])
                        cls_name = model.names[int(b.cls[0])]
                        active.add(tid)
                        info = tracks.setdefault(
                            tid, {"cls": Counter(), "t_start": ts, "t_end": ts, "frames": 0}
                        )
                        info["cls"][cls_name] += 1
                        info["t_end"] = ts
                        info["frames"] += 1
                        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                        dets.append(
                            Detection(
                                label=cls_name,
                                confidence=round(float(b.conf[0]), 3),
                                bbox=(round(x1 / w, 4), round(y1 / h, 4),
                                      round(x2 / w, 4), round(y2 / h, 4)),
                                track_id=_label(tid),
                            )
                        )

                # Derived events (entered/left/loitering) from track lifecycle.
                events = deriver.step(
                    ts, active, _label, lambda t: tracks[t]["t_start"]
                )
                for e in events:
                    video_events.append({"t": round(ts, 2), "event": e["phrase"]})

                is_kf = picker.emit(ts, frozenset(active))
                if events and not is_kf:
                    is_kf = True
                    picker.mark(ts, frozenset(active))

                if is_kf and len(keyframes) < self.max_keyframes:
                    rgb = cv2.cvtColor(r.orig_img, cv2.COLOR_BGR2RGB)
                    keyframes.append(
                        ExtractedFrame(
                            frame_idx=i,
                            timestamp_sec=round(ts, 3),
                            pil_image=Image.fromarray(rgb),
                            diff_score=0.0,
                        )
                    )
                    kf_dets.append(dets)
                    kf_events.append(events)
                    # Track facts for this keyframe: which objects are present
                    # and since when — grounds the VLM caption in temporal state.
                    facts = "; ".join(
                        f"{_label(t)} (present since {tracks[t]['t_start']:.0f}s)"
                        for t in sorted(active)
                    )
                    if events:
                        facts += ("; " if facts else "") + "EVENTS: " + \
                            "; ".join(e["phrase"] for e in events)
                    kf_facts.append(facts)
        finally:
            if writer is not None:
                writer.release()
                logger.info("Wrote tracking video -> %s", out_path)

        summary = _summarise(tracks, model.names)
        summary["events"] = video_events
        logger.info(
            "Tracked %d unique objects -> %s; %d keyframes; %d events",
            len(tracks), summary["counts"], len(keyframes), len(video_events),
        )
        return keyframes, kf_dets, kf_facts, kf_events, summary


if __name__ == "__main__":
    # Self-check for the keyframe-selection rule (no video needed).
    p = KeyframePicker(period_sec=2.0, min_gap_sec=0.5)
    # First frame with an object -> emit (id-set changed from None).
    assert p.emit(0.0, frozenset({1})) is True
    # Same ids, within min_gap -> no emit.
    assert p.emit(0.2, frozenset({1})) is False
    # New id appears after min_gap -> emit.
    assert p.emit(0.6, frozenset({1, 2})) is True
    # No change but period elapsed -> periodic emit.
    assert p.emit(2.7, frozenset({1, 2})) is True
    # Object leaves shortly after -> change, but within min_gap -> no emit.
    assert p.emit(2.8, frozenset({1})) is False
    print("KeyframePicker self-check OK")

    # EventDeriver returns {id, phrase} dicts; ids let us tie events to boxes.
    d = EventDeriver(loiter_sec=10.0)
    tstart = {"a": 0.0, "b": 2.0}
    lbl = lambda x: f"person_{x}"
    e0 = d.step(0.0, {"a"}, lbl, lambda o: tstart[o])
    assert e0 == [{"id": "person_a", "phrase": "person_a entered"}], e0
    d.step(2.0, {"a", "b"}, lbl, lambda o: tstart[o])
    e2 = d.step(12.0, {"b"}, lbl, lambda o: tstart[o])
    assert any(x["phrase"].endswith("left") and x["id"] == "person_a" for x in e2), e2
    # IVA renderer returns {id, phrase}, id = the triggering track label.
    r = render_iva_event({"type": "TW", "id": "x", "zone": "Main Gate", "dir": "A->B"},
                         lambda i: "person_P2")
    assert r["id"] == "person_P2" and "tripwire" in r["phrase"], r
    print("EventDeriver + render_iva_event self-check OK")

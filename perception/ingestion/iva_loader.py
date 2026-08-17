"""Ingest a video using an external IVA application's per-frame JSON output.

The IVA already performs detection, tracking (stable ids), and LPR, so this
loader *replaces* our YOLO+ByteTrack pass: it reads the IVA ``meta.jsonl``
(one JSON object per frame) and produces the exact same tuple as
:meth:`VideoTracker.process` — ``(keyframes, detections, track_facts, summary)``
— so the rest of ingestion (VLM caption, embed, store) is unchanged.

Expected per-frame schema (one JSON object per line)::

    {"seg":1, "frame":1, "ts_ms":0.0,
     "objects":[{"id":"..._P0000","cls":"person","bbox":[x1,y1,x2,y2],"score":0.9},
                {"id":"..._V0005","cls":"vehicle","bbox":[...],"score":0.87,
                 "lpr":{"plate":"GJ66PL0451","ocr_score":0.92}}],
     "events":[]}

bbox is absolute pixels in the recording's resolution (probed from the video).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import TYPE_CHECKING, Any

from perception.ingestion.tracker import EventDeriver, KeyframePicker, render_iva_event

if TYPE_CHECKING:
    from perception.models import Detection, ExtractedFrame, VideoMeta

logger = logging.getLogger(__name__)


def _short_id(obj_id: str) -> str:
    """Readable id suffix, e.g. '23072026..._P0000' -> 'P0000'."""
    return obj_id.rsplit("_", 1)[-1] or obj_id


def _summarise(tracks: dict[str, dict]) -> dict:
    """Unique counts per class + one row per track (with plate when known)."""
    counts: Counter[str] = Counter()
    rows: list[dict] = []
    for oid, info in tracks.items():
        cls = info["cls"].most_common(1)[0][0]
        counts[cls] += 1
        row = {
            "id": f"{cls}_{_short_id(oid)}",
            "cls": cls,
            "t_start": round(info["t_start"], 2),
            "t_end": round(info["t_end"], 2),
            "frames": info["frames"],
        }
        if info["plate"]:
            row["plate"] = info["plate"]
        rows.append(row)
    rows.sort(key=lambda r: r["t_start"])
    return {"counts": dict(counts), "tracks": rows}


def parse_manifest_regions(manifest_path: str, w: float, h: float) -> list[dict]:
    """Extract active analytics regions from an IVA manifest.jsonl config line.

    Returns a list of normalised regions to draw, e.g.::

        [{"kind": "line", "name": "Tripwire", "points": [[x1,y1],[x2,y2]]},
         {"kind": "polygon", "name": "Zone 1", "points": [[x,y], ...]}]

    Coordinates are normalised to [0,1] by the recording resolution (w, h), so
    they line up with the stored detection boxes on any thumbnail size.
    """
    w = w or 1.0
    h = h or 1.0

    def npt(p):
        return [round(float(p[0]) / w, 4), round(float(p[1]) / h, 4)]

    config = None
    try:
        with open(manifest_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("type") == "config":
                    config = rec
                    break
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read IVA manifest %s", manifest_path)
        return []
    if not config:
        return []

    regions: list[dict] = []
    for key, val in config.items():
        if key == "type" or not isinstance(val, dict) or not val.get("active"):
            continue
        line = val.get("line")
        if isinstance(line, dict) and "a" in line and "b" in line:
            regions.append({"kind": "line", "name": key.capitalize(),
                            "points": [npt(line["a"]), npt(line["b"])]})
        for zone in val.get("zones", []) or []:
            poly = zone.get("polygon")
            if poly:
                regions.append({"kind": "polygon", "name": zone.get("name", key),
                                "points": [npt(p) for p in poly]})
        poly = val.get("polygon")
        if poly:
            regions.append({"kind": "polygon", "name": key.capitalize(),
                            "points": [npt(p) for p in poly]})
    logger.info("Parsed %d analytics region(s) from manifest", len(regions))
    return regions


class IVALoader:
    """Turn an IVA meta.jsonl + its video into keyframes/detections/facts/summary."""

    def __init__(self) -> None:
        from perception.config import get_settings

        s = get_settings()
        self.period = s.track_keyframe_period_sec
        self.min_gap = s.track_min_gap_sec
        self.max_keyframes = s.max_keyframes
        self.loiter_sec = s.track_loiter_sec

    def process(
        self, meta: "VideoMeta", meta_jsonl_path: str, manifest_path: str | None = None
    ) -> tuple[list["ExtractedFrame"], list[list["Detection"]], list[str], list[list[dict]], dict]:
        import cv2
        from PIL import Image

        from perception.models import Detection, ExtractedFrame

        w = float(meta.width or 1)
        h = float(meta.height or 1)
        regions = parse_manifest_regions(manifest_path, w, h) if manifest_path else []
        picker = KeyframePicker(self.period, self.min_gap)
        deriver = EventDeriver(self.loiter_sec)
        tracks: dict[str, dict] = {}
        keyframes: list[ExtractedFrame] = []
        kf_dets: list[list[Detection]] = []
        kf_facts: list[str] = []
        kf_events: list[list[dict]] = []
        video_events: list[dict] = []

        def _label(oid: str) -> str:
            return f"{tracks[oid]['cls'].most_common(1)[0][0]}_{_short_id(oid)}"

        cap = cv2.VideoCapture(meta.source)
        logger.info("IVA ingest: %s + %s", meta.source, meta_jsonl_path)
        try:
            with open(meta_jsonl_path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    ts = float(rec.get("ts_ms", 0)) / 1000.0
                    objs = rec.get("objects", [])

                    active: set[str] = set()
                    dets: list[Detection] = []
                    for o in objs:
                        oid = str(o["id"])
                        cls = str(o["cls"])
                        active.add(oid)
                        info = tracks.setdefault(
                            oid, {"cls": Counter(), "t_start": ts, "t_end": ts, "frames": 0, "plate": ""}
                        )
                        info["cls"][cls] += 1
                        info["t_end"] = ts
                        info["frames"] += 1
                        plate = (o.get("lpr") or {}).get("plate")
                        if plate:
                            info["plate"] = plate
                        x1, y1, x2, y2 = o["bbox"]
                        dets.append(
                            Detection(
                                label=cls,
                                confidence=round(float(o.get("score", 0.0)), 3),
                                bbox=(round(x1 / w, 4), round(y1 / h, 4),
                                      round(x2 / w, 4), round(y2 / h, 4)),
                                track_id=_label(oid),
                            )
                        )

                    # Events: derived (entered/left/loiter) + IVA's own events.
                    # Each is {"id": <track label>, "phrase": <text>}.
                    events = deriver.step(ts, active, _label, lambda o: tracks[o]["t_start"])
                    events += [render_iva_event(ev, _label) for ev in rec.get("events", [])]
                    for e in events:
                        video_events.append({"t": round(ts, 2), "event": e["phrase"]})

                    # Advance the decoder one frame (assumes one meta line per frame).
                    grabbed = cap.grab()
                    is_kf = picker.emit(ts, frozenset(active))
                    if events and not is_kf:
                        is_kf = True
                        picker.mark(ts, frozenset(active))
                    if is_kf and len(keyframes) < self.max_keyframes:
                        ok, bgr = cap.retrieve() if grabbed else (False, None)
                        if not ok:
                            continue
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        keyframes.append(
                            ExtractedFrame(
                                frame_idx=int(rec.get("frame", len(keyframes))),
                                timestamp_sec=round(ts, 3),
                                pil_image=Image.fromarray(rgb),
                                diff_score=0.0,
                            )
                        )
                        kf_dets.append(dets)
                        kf_events.append(events)
                        facts = self._facts(active, tracks)
                        if events:
                            facts += ("; " if facts else "") + "EVENTS: " + \
                                "; ".join(e["phrase"] for e in events)
                        kf_facts.append(facts)
        finally:
            cap.release()

        summary = _summarise(tracks)
        summary["events"] = video_events
        summary["regions"] = regions
        logger.info(
            "IVA: %d tracked objects -> %s; %d keyframes; %d events",
            len(tracks), summary["counts"], len(keyframes), len(video_events),
        )
        return keyframes, kf_dets, kf_facts, kf_events, summary

    @staticmethod
    def _facts(active: set[str], tracks: dict[str, dict]) -> str:
        """Grounding line for a keyframe: ids present, since when, with plates."""
        parts = []
        for oid in sorted(active):
            info = tracks[oid]
            cls = info["cls"].most_common(1)[0][0]
            label = f"{cls}_{_short_id(oid)} (present since {info['t_start']:.0f}s"
            if info["plate"]:
                label += f", plate {info['plate']}"
            parts.append(label + ")")
        return "; ".join(parts)

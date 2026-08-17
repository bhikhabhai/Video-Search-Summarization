"""Per-video tracking summary, stored as one JSON file per video.

Tracks are *per-video* data (unique counts, one row per tracked object), a
different grain from the per-frame records in ChromaDB — so they live in their
own file, ``data/tracks/{video_id}.json``, rather than being duplicated onto
every frame's metadata.
"""

from __future__ import annotations

import json
import logging
import threading

from perception.config import get_settings

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()


def _path(video_id: str):
    d = get_settings().data_dir / "tracks"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{video_id}.json"


def write_tracks(video_id: str, summary: dict) -> None:
    """Persist ``{"counts": {...}, "tracks": [...]}`` for one video."""
    with _LOCK:
        _path(video_id).write_text(json.dumps(summary, indent=2))
    logger.info("Wrote track summary for %s: %s", video_id, summary.get("counts"))


def read_tracks(video_id: str) -> dict | None:
    """Return the stored summary, or None if this video was never tracked."""
    path = _path(video_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        logger.warning("Corrupt track file for %s; ignoring", video_id)
        return None


def delete_tracks(video_id: str) -> None:
    """Remove a video's track file (best-effort)."""
    with _LOCK:
        _path(video_id).unlink(missing_ok=True)

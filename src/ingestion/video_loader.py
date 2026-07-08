"""Probe video sources (local files or RTSP URLs) with FFmpeg.

The loader is deliberately thin: it validates that the source is decodable,
extracts duration / fps / resolution, and enforces the upload limits from
config. Frame decoding lives in :mod:`src.ingestion.frame_extractor`.
"""

from __future__ import annotations

import logging
import uuid
from fractions import Fraction
from pathlib import Path

import ffmpeg

from src.config import get_settings
from src.models import VideoMeta

logger = logging.getLogger(__name__)


class VideoLoadError(RuntimeError):
    """Raised when a source cannot be probed or violates ingest limits."""


def _is_rtsp(source: str) -> bool:
    return source.lower().startswith(("rtsp://", "rtsps://"))


def _parse_fps(stream: dict[str, str]) -> float:
    """Parse fps from an ffprobe stream dict, preferring avg_frame_rate."""
    for key in ("avg_frame_rate", "r_frame_rate"):
        raw = stream.get(key, "")
        try:
            value = float(Fraction(raw))
            if value > 0:
                return value
        except (ValueError, ZeroDivisionError):
            continue
    return 0.0


def load_video(source: str, video_id: str | None = None) -> VideoMeta:
    """Probe ``source`` and return a validated :class:`VideoMeta`.

    Args:
        source: Path to a local video file, or an RTSP URL.
        video_id: Optional stable id; a fresh UUID hex is generated otherwise.

    Raises:
        VideoLoadError: if the source is missing, not decodable, or exceeds
            the configured duration/size limits (limits only apply to files).
    """
    settings = get_settings()
    rtsp = _is_rtsp(source)
    size_bytes = 0
    filename = source

    if not rtsp:
        path = Path(source)
        if not path.is_file():
            raise VideoLoadError(f"Video file not found: {source}")
        size_bytes = path.stat().st_size
        filename = path.name
        if size_bytes > settings.max_video_mb * 1024 * 1024:
            raise VideoLoadError(
                f"Video is {size_bytes / 1024 / 1024:.1f} MB; "
                f"limit is {settings.max_video_mb:.0f} MB"
            )

    try:
        probe = ffmpeg.probe(source)
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else str(exc)
        raise VideoLoadError(f"FFmpeg could not probe {source}: {stderr[-500:]}") from exc

    video_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
    if not video_streams:
        raise VideoLoadError(f"No video stream found in {source}")
    stream = video_streams[0]

    duration = float(stream.get("duration") or probe.get("format", {}).get("duration") or 0.0)
    if not rtsp:
        if duration <= 0:
            raise VideoLoadError(f"Could not determine duration of {source}")
        if duration > settings.max_video_seconds:
            raise VideoLoadError(
                f"Video is {duration:.0f}s long; limit is {settings.max_video_seconds:.0f}s"
            )

    meta = VideoMeta(
        video_id=video_id or uuid.uuid4().hex[:12],
        source=source,
        duration_sec=duration,
        fps=_parse_fps(stream),
        width=int(stream.get("width", 0)),
        height=int(stream.get("height", 0)),
        size_bytes=size_bytes,
        filename=filename,
    )
    logger.info(
        "Loaded video %s: %.1fs @ %.2f fps, %dx%d (%s)",
        meta.video_id, meta.duration_sec, meta.fps, meta.width, meta.height, meta.filename,
    )
    return meta


if __name__ == "__main__":
    import argparse
    import sys

    from src.config import configure_logging

    configure_logging()
    parser = argparse.ArgumentParser(description="Probe a video file or RTSP URL")
    parser.add_argument("source", help="video path or rtsp:// URL")
    args = parser.parse_args()
    try:
        print(load_video(args.source).model_dump_json(indent=2))
    except VideoLoadError as err:
        print(f"error: {err}", file=sys.stderr)
        sys.exit(1)

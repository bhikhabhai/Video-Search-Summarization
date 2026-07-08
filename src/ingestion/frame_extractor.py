"""Intelligent keyframe extraction via dual-threshold scene-change detection.

Why dual-threshold instead of a single metric?
==============================================

Each metric alone has a characteristic failure mode:

* **Absolute frame differencing** (mean |I_t - I_{t-1}| on grayscale) fires on
  *any* pixel motion. A camera pan, a walking person, or flickering lights all
  produce large diffs even though the *scene* has not changed — using it alone
  over-samples motion-heavy footage.

* **Histogram comparison** (chi-squared distance between colour histograms)
  ignores spatial layout entirely. It is robust to motion, but blind to cuts
  between two scenes that happen to share a similar palette (e.g. two
  different rooms with beige walls) — using it alone under-samples.

Requiring **both** metrics to exceed their thresholds means a frame is only
declared a scene cut when the image changed structurally (large pixel diff)
*and* its global colour statistics shifted (large histogram distance). This
suppresses the false positives of each metric with the strengths of the
other, which in practice yields far cleaner keyframe sets than either metric
alone or a naive uniform sampler.

A ``min_scene_gap_sec`` cool-down additionally prevents burst-sampling during
rapid cut sequences (music videos, trailers), which would otherwise flood the
downstream captioning/embedding stages with near-duplicate work.

The detection core (:meth:`FrameExtractor.extract_from_arrays`) is a pure
function of numpy arrays, so it is unit-testable without FFmpeg or real video
files. :meth:`FrameExtractor.extract` wraps it with an FFmpeg decode pipe.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Iterator

import cv2
import ffmpeg
import numpy as np
from PIL import Image

from src.config import get_settings
from src.models import ExtractedFrame, VideoMeta

logger = logging.getLogger(__name__)

# Analysis happens on a small grayscale/HSV proxy for speed; keyframes keep
# the full decoded resolution.
_ANALYSIS_SIZE = (160, 96)  # (w, h)
_HIST_BINS = 32


def _gray_proxy(rgb: np.ndarray) -> np.ndarray:
    """Small grayscale version of an RGB frame used for pixel differencing."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, _ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)


def _colour_hist(rgb: np.ndarray) -> np.ndarray:
    """Normalised hue+saturation histogram of an RGB frame."""
    small = cv2.resize(rgb, _ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [_HIST_BINS, _HIST_BINS], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist


def frame_metrics(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    prev_hist: np.ndarray,
    hist: np.ndarray,
) -> tuple[float, float]:
    """Return ``(diff_score, hist_score)`` between two consecutive frames.

    ``diff_score`` is the mean absolute grayscale difference normalised to
    [0, 1]; ``hist_score`` is the chi-squared distance between the two
    L1-normalised colour histograms (0 = identical, larger = more different).
    """
    diff_score = float(np.mean(cv2.absdiff(prev_gray, gray))) / 255.0
    hist_score = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR))
    return diff_score, hist_score


class FrameExtractor:
    """Scene-change based keyframe extractor with dual-threshold detection."""

    def __init__(
        self,
        diff_threshold: float | None = None,
        hist_threshold: float | None = None,
        min_scene_gap_sec: float | None = None,
        max_gap_sec: float | None = None,
        analysis_fps: float | None = None,
        max_keyframes: int | None = None,
    ) -> None:
        settings = get_settings()
        self.diff_threshold = diff_threshold if diff_threshold is not None else settings.diff_threshold
        self.hist_threshold = hist_threshold if hist_threshold is not None else settings.hist_threshold
        self.min_scene_gap_sec = (
            min_scene_gap_sec if min_scene_gap_sec is not None else settings.min_scene_gap_sec
        )
        self.max_gap_sec = max_gap_sec if max_gap_sec is not None else settings.max_gap_sec
        self.analysis_fps = analysis_fps if analysis_fps is not None else settings.analysis_fps
        self.max_keyframes = max_keyframes if max_keyframes is not None else settings.max_keyframes
        self.analysis_width = settings.analysis_width

    # ── Detection core (FFmpeg-free, unit-testable) ────────────

    def extract_from_arrays(
        self, frames: Iterable[tuple[int, float, np.ndarray]]
    ) -> list[ExtractedFrame]:
        """Run dual-threshold cut detection over ``(idx, timestamp, rgb)`` tuples.

        A frame becomes a keyframe when ANY of these is true:
        - It is the very first frame (always kept).
        - BOTH the pixel-diff score AND the histogram chi-squared score exceed
          their thresholds AND at least ``min_scene_gap_sec`` has elapsed
          (scene-cut path).
        - ``max_gap_sec`` seconds have passed since the last keyframe with no
          cut detected (time-based fallback — ensures static-camera activity
          videos, e.g. a balcony with people entering/exiting, are still
          densely sampled even when the background never changes).
        """
        keyframes: list[ExtractedFrame] = []
        prev_gray: np.ndarray | None = None
        prev_hist: np.ndarray | None = None
        last_key_ts = -math.inf

        for frame_idx, timestamp, rgb in frames:
            gray = _gray_proxy(rgb)
            hist = _colour_hist(rgb)

            if prev_gray is None:
                is_cut, score = True, 1.0
            else:
                diff_score, hist_score = frame_metrics(prev_gray, gray, prev_hist, hist)
                scene_cut = (
                    diff_score > self.diff_threshold
                    and hist_score > self.hist_threshold
                    and (timestamp - last_key_ts) >= self.min_scene_gap_sec
                )
                time_gap = (timestamp - last_key_ts) >= self.max_gap_sec
                is_cut = scene_cut or time_gap
                score = diff_score

            if is_cut:
                keyframes.append(
                    ExtractedFrame(
                        frame_idx=frame_idx,
                        timestamp_sec=round(timestamp, 3),
                        pil_image=Image.fromarray(rgb),
                        diff_score=round(score, 4),
                    )
                )
                last_key_ts = timestamp
                if len(keyframes) >= self.max_keyframes:
                    logger.warning("Hit max_keyframes=%d; stopping early", self.max_keyframes)
                    break

            prev_gray, prev_hist = gray, hist

        logger.info("Selected %d keyframes", len(keyframes))
        return keyframes

    # ── FFmpeg decode wrapper ──────────────────────────────────

    def _decode_frames(self, meta: VideoMeta) -> Iterator[tuple[int, float, np.ndarray]]:
        """Decode ``meta.source`` at ``analysis_fps`` into RGB numpy frames."""
        width = min(self.analysis_width, meta.width) if meta.width else self.analysis_width
        # FFmpeg requires even dimensions; -2 keeps the aspect ratio.
        scale_h = -2
        stream = (
            ffmpeg.input(meta.source)
            .filter("fps", fps=self.analysis_fps)
            .filter("scale", width, scale_h)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24", loglevel="error")
        )
        process = stream.run_async(pipe_stdout=True)

        # Compute the actual decoded height (aspect-preserving, rounded to even).
        if meta.width and meta.height:
            height = int(round(meta.height * width / meta.width / 2) * 2)
        else:  # pragma: no cover - RTSP without probe dims
            raise RuntimeError("Cannot decode without known source dimensions")

        frame_bytes = width * height * 3
        idx = 0
        try:
            while True:
                chunk = process.stdout.read(frame_bytes)
                if len(chunk) < frame_bytes:
                    break
                rgb = np.frombuffer(chunk, dtype=np.uint8).reshape((height, width, 3))
                timestamp = idx / self.analysis_fps
                yield idx, timestamp, rgb
                idx += 1
        finally:
            process.stdout.close()
            process.wait()

    def extract(self, meta: VideoMeta) -> list[ExtractedFrame]:
        """Decode the video and return its keyframes."""
        logger.info(
            "Extracting keyframes from %s (diff>%.2f AND chi2>%.2f, gap>=%.1fs, max_gap=%.1fs)",
            meta.source, self.diff_threshold, self.hist_threshold,
            self.min_scene_gap_sec, self.max_gap_sec,
        )
        return self.extract_from_arrays(self._decode_frames(meta))


# ── Visualisation helper ───────────────────────────────────────


def save_contact_sheet(
    frames: list[ExtractedFrame],
    out_path: str,
    columns: int = 5,
    thumb_width: int = 240,
) -> str:
    """Save a grid of extracted keyframes (with timestamps) as one PNG.

    Handy for eyeballing threshold tuning: run the extractor with different
    settings and compare sheets side by side.
    """
    if not frames:
        raise ValueError("No frames to render")

    thumbs: list[Image.Image] = []
    for frame in frames:
        img = frame.pil_image.copy()
        ratio = thumb_width / img.width
        img = img.resize((thumb_width, max(1, int(img.height * ratio))))
        thumbs.append(img)

    cell_h = max(t.height for t in thumbs) + 22  # room for the timestamp label
    rows = math.ceil(len(thumbs) / columns)
    sheet = Image.new("RGB", (columns * thumb_width, rows * cell_h), (18, 18, 18))

    from PIL import ImageDraw

    draw = ImageDraw.Draw(sheet)
    for i, (frame, thumb) in enumerate(zip(frames, thumbs)):
        x = (i % columns) * thumb_width
        y = (i // columns) * cell_h
        sheet.paste(thumb, (x, y))
        label = f"t={frame.timestamp_sec:.1f}s  d={frame.diff_score:.2f}"
        draw.text((x + 4, y + thumb.height + 4), label, fill=(230, 230, 230))

    sheet.save(out_path)
    logger.info("Contact sheet with %d frames written to %s", len(frames), out_path)
    return out_path


if __name__ == "__main__":
    import argparse

    from src.config import configure_logging
    from src.ingestion.video_loader import load_video

    configure_logging()
    parser = argparse.ArgumentParser(description="Extract keyframes and save a contact sheet")
    parser.add_argument("video", help="path to a video file")
    parser.add_argument("--out", default="contact_sheet.png", help="output PNG path")
    parser.add_argument("--diff", type=float, default=None, help="diff threshold override")
    parser.add_argument("--hist", type=float, default=None, help="histogram threshold override")
    args = parser.parse_args()

    video_meta = load_video(args.video)
    extractor = FrameExtractor(diff_threshold=args.diff, hist_threshold=args.hist)
    extracted = extractor.extract(video_meta)
    for kf in extracted:
        print(f"  frame {kf.frame_idx:5d}  t={kf.timestamp_sec:7.2f}s  diff={kf.diff_score:.3f}")
    if extracted:
        save_contact_sheet(extracted, args.out)
        print(f"contact sheet -> {args.out}")

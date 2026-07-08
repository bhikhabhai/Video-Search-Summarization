"""Unit tests for the dual-threshold scene-change detector.

The detector core operates on ``(idx, timestamp, rgb array)`` tuples, so we
can feed synthetic frames without FFmpeg or real video files.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ingestion.frame_extractor import FrameExtractor, frame_metrics, _colour_hist, _gray_proxy

WIDTH, HEIGHT = 320, 180
FPS = 6.0


def solid_frame(rgb: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:, :] = rgb
    return frame


def noisy_frame(rgb: tuple[int, int, int], seed: int) -> np.ndarray:
    """A solid frame with mild noise, simulating sensor grain (not a cut)."""
    rng = np.random.default_rng(seed)
    base = solid_frame(rgb).astype(np.int16)
    noise = rng.integers(-6, 7, size=base.shape, dtype=np.int16)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def as_stream(frames: list[np.ndarray]) -> list[tuple[int, float, np.ndarray]]:
    return [(i, i / FPS, frame) for i, frame in enumerate(frames)]


@pytest.fixture()
def extractor() -> FrameExtractor:
    # max_gap_sec=inf disables the time-based fallback so these tests isolate the
    # dual-threshold scene-cut logic. The fallback has its own test below.
    return FrameExtractor(
        diff_threshold=0.12,
        hist_threshold=0.35,
        min_scene_gap_sec=1.5,
        max_gap_sec=float("inf"),
        analysis_fps=FPS,
        max_keyframes=100,
    )


class TestFirstFrame:
    def test_first_frame_always_kept(self, extractor: FrameExtractor) -> None:
        keyframes = extractor.extract_from_arrays(as_stream([solid_frame((50, 80, 120))]))
        assert len(keyframes) == 1
        assert keyframes[0].frame_idx == 0
        assert keyframes[0].timestamp_sec == 0.0

    def test_empty_stream(self, extractor: FrameExtractor) -> None:
        assert extractor.extract_from_arrays([]) == []


class TestStaticScene:
    def test_identical_frames_produce_one_keyframe(self, extractor: FrameExtractor) -> None:
        frames = [solid_frame((90, 90, 90))] * 30
        keyframes = extractor.extract_from_arrays(as_stream(frames))
        assert len(keyframes) == 1

    def test_sensor_noise_is_not_a_cut(self, extractor: FrameExtractor) -> None:
        frames = [noisy_frame((120, 100, 80), seed=i) for i in range(30)]
        keyframes = extractor.extract_from_arrays(as_stream(frames))
        assert len(keyframes) == 1


class TestSceneCuts:
    def test_abrupt_scene_change_detected(self, extractor: FrameExtractor) -> None:
        # 3 seconds of blue-ish, then 3 seconds of red-ish: exactly one cut.
        scene_a = [solid_frame((20, 40, 200))] * int(3 * FPS)
        scene_b = [solid_frame((200, 40, 20))] * int(3 * FPS)
        keyframes = extractor.extract_from_arrays(as_stream(scene_a + scene_b))
        assert len(keyframes) == 2
        assert keyframes[1].frame_idx == len(scene_a)
        # Timestamp of the cut should land at the scene boundary.
        assert keyframes[1].timestamp_sec == pytest.approx(len(scene_a) / FPS)

    def test_multiple_distinct_scenes(self, extractor: FrameExtractor) -> None:
        colours = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30)]
        frames: list[np.ndarray] = []
        for colour in colours:
            frames.extend([solid_frame(colour)] * int(2 * FPS))  # 2s per scene > gap
        keyframes = extractor.extract_from_arrays(as_stream(frames))
        assert len(keyframes) == len(colours)


class TestDualThreshold:
    def test_brightness_only_change_requires_both_metrics(self) -> None:
        """A pure luminance shift trips the pixel diff but the colour histogram
        (hue/saturation) barely moves — dual-threshold must NOT fire."""
        extractor = FrameExtractor(
            diff_threshold=0.10, hist_threshold=0.35, min_scene_gap_sec=0.0,
            max_gap_sec=float("inf"), analysis_fps=FPS,
        )
        dark = [solid_frame((60, 60, 60))] * 10
        bright = [solid_frame((160, 160, 160))] * 10  # same grey hue, brighter
        keyframes = extractor.extract_from_arrays(as_stream(dark + bright))
        # Sanity: the diff metric alone WOULD have fired.
        diff, hist = frame_metrics(
            _gray_proxy(dark[0]), _gray_proxy(bright[0]),
            _colour_hist(dark[0]), _colour_hist(bright[0]),
        )
        assert diff > 0.10
        assert len(keyframes) == 1  # but the combined detector held back

    def test_metrics_are_zero_for_identical_frames(self) -> None:
        frame = solid_frame((10, 200, 90))
        diff, hist = frame_metrics(
            _gray_proxy(frame), _gray_proxy(frame),
            _colour_hist(frame), _colour_hist(frame),
        )
        assert diff == pytest.approx(0.0)
        assert hist == pytest.approx(0.0)


class TestMinSceneGap:
    def test_rapid_cuts_are_suppressed(self, extractor: FrameExtractor) -> None:
        """Cuts every 0.5 s with a 1.5 s gap: bursts must be thinned out."""
        colours = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30),
                   (30, 200, 200), (200, 30, 200)]
        frames: list[np.ndarray] = []
        for colour in colours:
            frames.extend([solid_frame(colour)] * 3)  # 3 frames = 0.5s per "scene"
        keyframes = extractor.extract_from_arrays(as_stream(frames))
        # 6 scenes over 3s with a 1.5s cool-down: at most 1 + floor(3/1.5) kept.
        assert 1 <= len(keyframes) <= 3
        # And every kept pair honours the gap.
        timestamps = [kf.timestamp_sec for kf in keyframes]
        gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
        assert all(gap >= 1.5 for gap in gaps)


class TestMaxKeyframes:
    def test_hard_cap(self) -> None:
        extractor = FrameExtractor(
            diff_threshold=0.01, hist_threshold=0.01, min_scene_gap_sec=0.0,
            analysis_fps=FPS, max_keyframes=5,
        )
        rng = np.random.default_rng(0)
        frames = [
            rng.integers(0, 255, size=(HEIGHT, WIDTH, 3), dtype=np.uint8).astype(np.uint8)
            for _ in range(50)
        ]
        keyframes = extractor.extract_from_arrays(as_stream(frames))
        assert len(keyframes) == 5


class TestMaxGapFallback:
    def test_static_scene_sampled_by_time(self) -> None:
        """A static camera (no scene cut) is still sampled every max_gap_sec so
        activity in an unchanging background isn't missed."""
        extractor = FrameExtractor(
            diff_threshold=0.12, hist_threshold=0.35, min_scene_gap_sec=1.0,
            max_gap_sec=3.0, analysis_fps=FPS, max_keyframes=100,
        )
        # 12 s of an identical frame -> forced keyframes at t=0, 3, 6, 9.
        frames = [solid_frame((90, 90, 90))] * int(12 * FPS)
        keyframes = extractor.extract_from_arrays(as_stream(frames))
        assert len(keyframes) == 4
        timestamps = [kf.timestamp_sec for kf in keyframes]
        gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
        assert all(gap == pytest.approx(3.0) for gap in gaps)

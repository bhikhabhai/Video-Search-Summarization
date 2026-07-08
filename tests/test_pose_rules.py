"""Unit tests for the pose-based action rule engine (no models required).

Keypoints are COCO-17, normalised [0,1], with y growing downward (image coords).
We build synthetic people to exercise fall / fight / running / crowding rules.
"""

from __future__ import annotations

import numpy as np

from src.inference.pose import (
    L_ANKLE,
    L_HIP,
    L_SHOULDER,
    L_WRIST,
    NOSE,
    R_ANKLE,
    R_HIP,
    R_SHOULDER,
    R_WRIST,
    Person,
    detect_events,
    has_raised_arm,
    is_fall,
    is_running,
)


def _blank() -> np.ndarray:
    return np.full((17, 2), np.nan, dtype=float)


def upright_person(cx: float = 0.5, top: float = 0.1, bottom: float = 0.9) -> Person:
    """A standing person: shoulders above hips above ankles, tall bbox."""
    kp = _blank()
    mid = bottom - (bottom - top)
    kp[NOSE] = [cx, top]
    kp[L_SHOULDER] = [cx - 0.05, top + 0.15]
    kp[R_SHOULDER] = [cx + 0.05, top + 0.15]
    kp[L_HIP] = [cx - 0.04, top + 0.45]
    kp[R_HIP] = [cx + 0.04, top + 0.45]
    kp[L_ANKLE] = [cx - 0.04, bottom]
    kp[R_ANKLE] = [cx + 0.04, bottom]
    # Tall bbox (aspect < 1).
    return Person(keypoints=kp, bbox=(cx - 0.06, top, cx + 0.06, bottom))


class TestFall:
    def test_upright_is_not_a_fall(self):
        assert not is_fall(upright_person())

    def test_wide_bbox_is_a_fall(self):
        p = upright_person()
        # Lying down => wide, short bbox.
        p.bbox = (0.1, 0.55, 0.9, 0.7)
        assert is_fall(p)

    def test_horizontal_torso_is_a_fall(self):
        kp = _blank()
        # Shoulders and hips at (almost) the same height, far apart in x => horizontal.
        kp[L_SHOULDER] = [0.2, 0.5]
        kp[R_SHOULDER] = [0.2, 0.52]
        kp[L_HIP] = [0.7, 0.5]
        kp[R_HIP] = [0.7, 0.52]
        p = Person(keypoints=kp, bbox=(0.2, 0.45, 0.7, 0.6))  # wide too, but torso alone suffices
        assert is_fall(p)


class TestRaisedArm:
    def test_wrist_above_shoulder(self):
        p = upright_person()
        p.keypoints[L_WRIST] = [0.44, 0.05]      # above the shoulder (smaller y)
        p.keypoints[L_SHOULDER] = [0.45, 0.25]
        assert has_raised_arm(p)

    def test_arms_down(self):
        p = upright_person()
        p.keypoints[L_WRIST] = [0.44, 0.6]       # below shoulder
        p.keypoints[R_WRIST] = [0.56, 0.6]
        assert not has_raised_arm(p)


class TestRunning:
    def test_wide_stance_with_lean(self):
        kp = _blank()
        # Forward lean: shoulders offset in x from hips; wide ankle spread.
        kp[L_SHOULDER] = [0.60, 0.3]
        kp[R_SHOULDER] = [0.62, 0.3]
        kp[L_HIP] = [0.49, 0.5]
        kp[R_HIP] = [0.51, 0.5]
        kp[L_ANKLE] = [0.40, 0.9]
        kp[R_ANKLE] = [0.62, 0.9]   # spread (0.22) >> hip width (0.02)
        p = Person(keypoints=kp, bbox=(0.4, 0.3, 0.62, 0.9))
        assert is_running(p)

    def test_standing_still_is_not_running(self):
        assert not is_running(upright_person())


class TestSceneEvents:
    def test_fall_event(self):
        p = upright_person()
        p.bbox = (0.1, 0.55, 0.9, 0.7)
        assert "fall" in detect_events([p])

    def test_fighting_two_close_with_raised_arm(self):
        a = upright_person(cx=0.48)
        b = upright_person(cx=0.52)
        a.keypoints[L_WRIST] = [0.47, 0.05]      # a raises an arm
        a.keypoints[L_SHOULDER] = [0.46, 0.25]
        assert "fighting" in detect_events([a, b])

    def test_far_apart_is_not_fighting(self):
        a = upright_person(cx=0.15)
        b = upright_person(cx=0.85)
        a.keypoints[L_WRIST] = [0.14, 0.05]
        a.keypoints[L_SHOULDER] = [0.15, 0.25]
        assert "fighting" not in detect_events([a, b])

    def test_crowding(self):
        crowd = [upright_person(cx=0.1 * i) for i in range(1, 7)]
        assert "crowding" in detect_events(crowd, crowd_threshold=5)

    def test_quiet_scene_has_no_events(self):
        assert detect_events([upright_person()]) == []

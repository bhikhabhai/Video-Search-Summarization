"""Unit tests for exhaustive-recall query helpers (no models required)."""

from __future__ import annotations

from perception.models import RetrievedFrame
from retrieval.api.main import _enumeration_candidates, _merge_frames


def _frame(idx: int, ts: float, caption: str = "", labels=None, ocr="", events=None):
    return RetrievedFrame(
        video_id="v1",
        frame_idx=idx,
        timestamp_sec=ts,
        caption=caption,
        detection_labels=labels or [],
        ocr_text=ocr,
        events=events or [],
    )


class TestEnumerationCandidates:
    def test_finds_all_frames_with_a_rare_term(self):
        frames = [
            _frame(0, 1.0, "a person walking", labels=["person"]),
            _frame(1, 5.0, "a person with a backpack", labels=["person", "backpack"]),
            _frame(2, 9.0, "two people", labels=["person"]),
            _frame(3, 47.0, "a green bag on a person", labels=["person", "backpack"]),
        ]
        out = _enumeration_candidates(frames, "find the time where person has backpack")
        # Both backpack frames returned, chronological — not just the first.
        assert [f.frame_idx for f in out] == [1, 3]

    def test_ubiquitous_term_does_not_pull_everything(self):
        # "person" is in every frame → it must NOT act as a filter on its own.
        frames = [_frame(i, float(i), "a person", labels=["person"]) for i in range(5)]
        assert _enumeration_candidates(frames, "where is a person") == []

    def test_no_specific_terms_returns_empty(self):
        frames = [_frame(0, 1.0, "a person", labels=["person"])]
        assert _enumeration_candidates(frames, "what is happening") == []

    def test_matches_ocr_and_events(self):
        frames = [
            _frame(0, 2.0, "a car", labels=["car"], ocr="GJ01AB1234"),
            _frame(1, 8.0, "people", labels=["person"], events=["fall"]),
        ]
        assert [f.frame_idx for f in _enumeration_candidates(frames, "gj01ab1234")] == [0]
        assert [f.frame_idx for f in _enumeration_candidates(frames, "any fall events?")] == [1]

    def test_tolerates_misspelling(self):
        frames = [
            _frame(0, 1.0, "a person walking", labels=["person"]),
            _frame(1, 11.5, "a person with a backpack", labels=["person", "backpack"]),
            _frame(2, 47.0, "a person and a backpack", labels=["person", "backpack"]),
        ]
        # "bag pak" (typo, split) should still find both backpack frames.
        out = _enumeration_candidates(frames, "persion with bag pak")
        assert [f.frame_idx for f in out] == [1, 2]

    def test_tolerates_split_word(self):
        frames = [
            _frame(0, 1.0, "empty street", labels=["car"]),
            _frame(1, 5.0, "someone with a backpack", labels=["backpack"]),
        ]
        # "back pack" as two words → joined → backpack.
        assert [f.frame_idx for f in _enumeration_candidates(frames, "find a back pack")] == [1]


class TestMergeFrames:
    def test_dedupes_and_sorts_chronologically(self):
        a = [_frame(3, 47.0), _frame(1, 5.0)]      # semantic hits (unordered)
        b = [_frame(1, 5.0), _frame(2, 9.0)]       # enumeration hits (overlap on idx 1)
        out = _merge_frames(a, b)
        assert [f.frame_idx for f in out] == [1, 2, 3]   # deduped, time-sorted

    def test_respects_cap(self):
        many = [_frame(i, float(i)) for i in range(50)]
        assert len(_merge_frames(many, [], cap=10)) == 10

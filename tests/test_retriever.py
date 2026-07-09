"""Integration-style tests for hybrid retrieval using injected fakes.

The fakes implement just the surface :class:`HybridRetriever` touches, with a
tiny hand-built "embedding space" (3-dim vectors), so the fusion + re-ranking
logic is exercised end-to-end without ChromaDB or CLIP weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from perception.inference.embedder import l2_normalise
from perception.models import RetrievedFrame
from retrieval.rag.retriever import HybridRetriever

VIDEO_ID = "vid001"

# Toy semantic space: axis 0 ~ "person/jacket", axis 1 ~ "car/road", axis 2 ~ "dog/park".
TEXT_SPACE: dict[str, np.ndarray] = {
    "a person wearing a red jacket": np.array([1.0, 0.1, 0.0]),
    "red jacket person": np.array([0.95, 0.1, 0.05]),
    "a car driving on the road": np.array([0.1, 1.0, 0.0]),
    "a dog playing in the park": np.array([0.0, 0.1, 1.0]),
    "<no caption>": np.array([0.0, 0.0, 0.0]),
}

FRAME_IMAGE_VECS: dict[int, np.ndarray] = {
    0: np.array([0.9, 0.2, 0.0]),   # person in jacket
    1: np.array([0.1, 0.95, 0.0]),  # car
    2: np.array([0.0, 0.1, 0.9]),   # dog
}


class FakeEmbedder:
    """Maps known phrases to the toy space; unknown text falls back to a
    bag-of-axis-words heuristic so arbitrary queries still get a vector."""

    def embed_text(self, text: str) -> np.ndarray:
        if text in TEXT_SPACE:
            return l2_normalise(TEXT_SPACE[text])
        lowered = text.lower()
        vec = np.array(
            [
                float(any(w in lowered for w in ("person", "jacket", "red"))),
                float(any(w in lowered for w in ("car", "road", "vehicle"))),
                float(any(w in lowered for w in ("dog", "park", "puppy"))),
            ]
        )
        return l2_normalise(vec if vec.any() else np.array([0.34, 0.33, 0.33]))

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed_text(t) for t in texts]


class FakeStore:
    """In-memory stand-in for VectorStore over three fixed frames."""

    def __init__(self) -> None:
        self.frames = [
            RetrievedFrame(
                video_id=VIDEO_ID, frame_idx=0, timestamp_sec=2.0,
                caption="a person wearing a red jacket",
                detection_labels=["person"],
            ),
            RetrievedFrame(
                video_id=VIDEO_ID, frame_idx=1, timestamp_sec=10.0,
                caption="a car driving on the road",
                detection_labels=["car"],
            ),
            RetrievedFrame(
                video_id=VIDEO_ID, frame_idx=2, timestamp_sec=20.0,
                caption="a dog playing in the park",
                detection_labels=["dog"],
            ),
        ]

    def get_by_video_id(self, video_id: str) -> list[RetrievedFrame]:
        return [f.model_copy() for f in self.frames] if video_id == VIDEO_ID else []

    def search_by_text(self, query, embedder, video_id, top_k=5):
        query_vec = embedder.embed_text(query)
        hits = []
        for frame in self.get_by_video_id(video_id):
            img_vec = l2_normalise(FRAME_IMAGE_VECS[frame.frame_idx])
            frame.score = float(query_vec @ img_vec)
            hits.append(frame)
        hits.sort(key=lambda f: f.score, reverse=True)
        return hits[:top_k]


@pytest.fixture()
def retriever() -> HybridRetriever:
    return HybridRetriever(store=FakeStore(), embedder=FakeEmbedder())  # type: ignore[arg-type]


class TestHybridRetrieval:
    def test_semantic_query_ranks_correct_frame_first(self, retriever: HybridRetriever) -> None:
        results = retriever.search(VIDEO_ID, "when does a person appear in a red jacket?")
        assert results
        assert results[0].frame_idx == 0
        assert results[0].timestamp_sec == 2.0

    def test_each_subject_maps_to_its_frame(self, retriever: HybridRetriever) -> None:
        for query, expected_idx in [
            ("red jacket person", 0),
            ("a car driving on the road", 1),
            ("a dog playing in the park", 2),
        ]:
            results = retriever.search(VIDEO_ID, query)
            assert results[0].frame_idx == expected_idx, query

    def test_scores_are_sorted_descending(self, retriever: HybridRetriever) -> None:
        results = retriever.search(VIDEO_ID, "car")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self, retriever: HybridRetriever) -> None:
        assert len(retriever.search(VIDEO_ID, "dog", top_k=1)) == 1
        assert len(retriever.search(VIDEO_ID, "dog", top_k=2)) == 2

    def test_unknown_video_returns_empty(self, retriever: HybridRetriever) -> None:
        assert retriever.search("nope", "anything") == []

    def test_bm25_channel_contributes(self, retriever: HybridRetriever) -> None:
        """A lexical-only cue ('park') present in exactly one caption must
        pull that frame to the top even with an ambiguous CLIP vector."""
        results = retriever.search(VIDEO_ID, "park")
        assert results[0].frame_idx == 2

    def test_detection_labels_are_searchable(self, retriever: HybridRetriever) -> None:
        """Detection labels are folded into the BM25 corpus."""
        results = retriever.search(VIDEO_ID, "vehicle car")
        assert results[0].frame_idx == 1
